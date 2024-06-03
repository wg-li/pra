#from mymodel import Model
from fnn import Net
from model_train import train, valid, test
from data_prep import dataset_creator
import torch
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from fix_seed import seed_torch
import time
from pytorchtools import EarlyStopping
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-o", '--oep', type=int, default=4, help='outer epochs')
parser.add_argument("-i", '--iep', type=int, default=2, help='inner epochs')
parser.add_argument("-s", '--seed', type=int, default=2, help='seed')
args = parser.parse_args()

out_epochs = args.oep
in_epochs = args.iep
seed = args.seed
seed_torch(seed)

def hp_conv(hparam):
    p = hparam['p']
    lr = 10 ** (-hparam['lr'])
    mom = hparam['mom']
    alpha = 10 ** (-hparam['alpha'])
    batch_size = int(2 ** hparam['batch_size'])
    return lr, mom, p, alpha, batch_size

def loss_f(hparam):
    lr, mom, p, alpha, batch_size = hp_conv(hparam)
    print('batch_size: ', batch_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    data_dir = "/home/zjlab/liwg/mydata/fashion_mnist/"
    train_loader, valid_loader, test_loader = dataset_creator(use_cuda, data_dir, seed,
                                                              l_downld=False,
                                                              batch_size=batch_size)
    input_size = 784  # 28*28
    hidden_size = 100
    num_classes = 10
    # Initialization
    model = Net(input_size, hidden_size, num_classes, p, alpha).to(device)
#   optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2), eps=eps)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom)

#   torch.save(model.state_dict(),'./tpe_weight.pt')

#   early_stopping = EarlyStopping(15, verbose=False)
    for epoch in range(in_epochs):
        train_res = train(model, device, train_loader, optimizer)
        results = valid(model, device, valid_loader)
#       early_stopping(results['valid loss'], model)
#       if early_stopping.early_stop:
#           print("Early stopping")
#           break
#   model.load_state_dict(torch.load('checkpoint.pt'))
    print('after inner training,')
    print(train_res)
    print(results)
    print('test accuracy: ', test(model, device, test_loader))
    acc = -results['valid accuracy']
    return acc

fspace = {
    'lr': hp.uniform('lr', 1., 4.),
    'mom': hp.uniform('mom', 0., 0.99),
    'p': hp.uniform('p', 0., 1.),
    'alpha': hp.uniform('alpha', 1., 5.),
    'batch_size': hp.choice('batch_size', [3, 4, 5, 6, 7])
    }
trials = Trials()
since = time.time()
best = fmin(fn=loss_f, space=fspace, algo=tpe.suggest, max_evals=out_epochs, rstate=np.random.default_rng(seed), trials=trials, show_progressbar=False)
time_elapsed = time.time() - since
print('-' * 30)
print('训练用时： {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
hp_bst = [None] * 5
hp_bst[0], hp_bst[1], hp_bst[2], hp_bst[3], hp_bst[4] = hp_conv(best)
print('best hp.:', hp_bst)
print('best validation accuracy:', -trials.best_trial['result']['loss'])

acc_rec=[]
print('trials:')
for i in range(len(trials.trials)):
    trial = trials.trials[i]
    x = trial['misc']['vals']
    # x1 = x.copy()
    for k in x:
        x[k] = x[k][0]
    hps = [None] * len(fspace)
    hps[0], hps[1], hps[2], hps[3], hps[4] = hp_conv(x)
    print('hps:', hps)
    acc_rec.append(-trial['result']['loss'])
#   print('valid acc:', -trial['result']['loss'])
print('valid accuracy record:', acc_rec)
np.save('acc_record.npy', acc_rec)
pass
