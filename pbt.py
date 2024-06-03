import ray
import argparse
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from mymodel import Model
from fnn import Net
from model_train import train, valid
from data_prep import dataset_creator
import numpy as np
import os
import torch
from ray.tune.trial import ExportFormat
from fix_seed import seed_torch
import time
import setproctitle

setproctitle.setproctitle("liwg-job")


parser = argparse.ArgumentParser()
parser.add_argument("-n", '--num', type=int, default=2, help='no. of models')
parser.add_argument("-t", '--tep', type=int, default=4, help='total epochs')
parser.add_argument("-i", '--iep', type=int, default=2, help='inner epochs')
parser.add_argument("-s", '--seed', type=int, default=2, help='random seed')
args = parser.parse_args()

num_models = args.num
total_epochs = args.tep
in_epochs = args.iep
seed = args.seed
synch = True

seed_torch(seed)
scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=in_epochs,
        hyperparam_mutations={
            # distribution for resampling
            'lr': tune.loguniform(1e-4, 0.1),
            'mom': tune.uniform(0, 0.99),
            'p': tune.uniform(0., 1.),
            'alpha': tune.loguniform(1e-5, 0.1),
#           'batch_size': tune.choice([8, 16, 32, 64, 128])
            'batch_size': [8, 16, 32, 64, 128]
        },
#       log_config=True,
        synch=synch,
    )

def within_bounds(lr, mom, p, alpha):
    if lr < 1e-4: lr = 1e-4
    if lr > 0.1: lr = 0.1
    if mom < 0: mom = 0
    if mom > 0.99: mom = 0.99
    if p < 0: p = 0.
    if p > 1: p = 1
    if alpha < 1e-5: alpha = 1e-5
    if alpha > 0.1: alpha = 0.1
    return lr, mom, p, alpha


def objective(config, checkpoint_dir=None):
#   seed_torch()
    step = 0
    lr = config.get("lr")
    mom = config.get("mom")
    p = config.get("p")
    alpha = config.get("alpha")
    batch_size = config.get("batch_size")
    
    lr, mom, p, alpha = within_bounds(lr, mom, p, alpha)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    data_dir = "/home/zjlab/liwg/mydata/fashion_mnist/"
    train_loader, valid_loader, test_loader = dataset_creator(use_cuda, data_dir, seed,
                                                                        l_downld=False,
                                                                        batch_size=batch_size)
    input_size = 784 
    hidden_size = 100
    num_classes = 10
    # Initialization
    model = Net(input_size, hidden_size, num_classes, p, alpha).to(device)
#   optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2), eps=eps)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom)
#   if step == 0:
#       torch.save(model.state_dict(),'./weight.pt')
    # If checkpoint_dir is not None, then we are resuming from a checkpoint.
    # Load model state and iteration step from checkpoint.
    if checkpoint_dir is not None:
        print("Loading from checkpoint.")
        path = os.path.join(checkpoint_dir, "checkpoint.pt")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        step = checkpoint["step"]

    while True:
        train_dict = train(model, device, train_loader, optimizer)
        results = valid(model, device, valid_loader)
        acc = results['valid accuracy']
        loss = results['valid loss']
        if step % in_epochs == 0:
            # Every 5 steps, checkpoint our current state.
            # First get the checkpoint directory from tune.
            with tune.checkpoint_dir(step=step) as checkpoint_dir:
                # Then create a checkpoint file in this directory.
                path = os.path.join(checkpoint_dir, "checkpoint.pt")
                # Save state to checkpoint file.
                # No need to save optimizer for SGD.
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "mean_accuracy": acc,
                        "mean_loss": loss,
                        },
                    path,
                )
        step += 1
        tune.report(mean_accuracy=acc, mean_loss=loss, train_acc=train_dict['train accuracy'])
    # results = model.train_steps(epochs=100, e_stop=early_stopping)
#    print('after inner training,')
#    print(results)
#    print('test accuracy: ', model.test_steps())


search_space = {
    'lr': tune.loguniform(1e-4, 0.1),
    'mom': tune.uniform(0, 0.99),
    'p': tune.uniform(0., 1.),
    'alpha': tune.loguniform(1e-5, 0.1),
#   'batch_size': tune.qlograndint(3, 7, 1, 2)
    'batch_size': tune.choice([8, 16, 32, 64, 128])
}

ray.init(num_cpus=num_models, num_gpus=1)#, object_store_memory=1073741824*2)
#runtime_env = {"working_dir": "./"}
#ray.init(address='10.101.104.44:6379', runtime_env=runtime_env)

since = time.time()
analysis = tune.run(
        objective,
        name="exp_pbt",
        scheduler=scheduler,
        metric="mean_accuracy",
        mode="max",
        verbose=1,
        stop={
            "training_iteration": total_epochs,
        }, #stopper,
        resources_per_trial={"cpu": 0.8, "gpu": 0.1},
#       local_dir='/data/nfs/user/liwg/pfbo/pbt_test/',
        local_dir='./',
#       sync_config=tune.SyncConfig(syncer=None),  # Disable syncing        
        export_formats=[ExportFormat.MODEL],
        # keep_checkpoints_num=4,
        # checkpoint_score_attr="mean_accuracy",
        num_samples=num_models,
        config=search_space,
        reuse_actors=True,
#       resume='AUTO',
    )
time_elapsed = time.time() - since
print('-' * 30)
print('训练用时： {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

ray.shutdown()

best_config = analysis.best_config
best_checkpoint = torch.load(str(analysis.best_checkpoint)+'checkpoint.pt')
print('best validation accuracy:', analysis.best_result['mean_accuracy'])
hp = [None]*len(search_space)
hp[0] = best_config['lr']
hp[1] = best_config['mom']
hp[2] = best_config['p']
hp[3] = best_config['alpha']
hp[-1] = best_config['batch_size']

hp[0], hp[1], hp[2], hp[3] = within_bounds(hp[0], hp[1], hp[2], hp[3])
print("Best hyperparameters found were: ", hp)
model = Model(hp, seed)
model.set_weights(best_checkpoint['model_state_dict'])
test_acc = model.test_steps()
print('test acc.:', test_acc)

dfs = analysis.trial_dataframes
import pickle
with open('results.pkl', 'wb') as f:
    pickle.dump(dfs, f)

if synch == True:
    acc_all = np.array([])
    loss_all = np.array([])
    i = 0
    for d in dfs.values():
        acc = d.mean_accuracy.values[in_epochs-1::in_epochs]
        loss = d.mean_loss.values[in_epochs-1::in_epochs]
        if i == 0:
            acc_all = acc.copy()
            loss_all = loss.copy()
        else:
            acc_all = np.vstack((acc_all, acc))
            loss_all = np.vstack((loss_all, loss))
        i += 1
    acc_rec = np.max(acc_all, axis=0)
    acc_idx = np.argmax(acc_all, axis=0)
    np.save('acc_record.npy', acc_rec)
    print('valid acc record', acc_rec)
    print('max valid acc: ', np.max(acc_rec))
    np.save('loss_record.npy', loss_rec)
    print('valid loss record', loss_rec)
    acc_all = np.array([])
    i = 0
    for d in dfs.values():
        acc = d.train_acc.values[in_epochs-1::in_epochs]
        if i == 0:
            acc_all = acc.copy()
        else:
            acc_all = np.vstack((acc_all, acc))
        i += 1
    acc_rec = np.max(acc_all, axis=0)
    np.save('train_acc_record.npy', acc_rec)
    print('train acc record', acc_rec)
#   print(d.mean_accuracy)
#[d.mean_accuracy.plot() for d in dfs.values()]
pass

