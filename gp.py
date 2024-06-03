import torch
from fnn import Net
from model_train import train, valid, test
from data_prep import dataset_creator
import numpy as np
from fix_seed import seed_torch
from bayes_opt import BayesianOptimization
import time
from pytorchtools import EarlyStopping
import argparse


# ifile='../init_weights.npz'


parser = argparse.ArgumentParser()
parser.add_argument("-o", '--oep', type=int, default=4, help='outer epochs')
parser.add_argument("-i", '--iep', type=int, default=2, help='inner epochs')
parser.add_argument("-s", '--seed', type=int, default=2, help='seed')
args = parser.parse_args()

out_epochs = args.oep
in_epochs = args.iep
seed = args.seed

seed_torch(seed)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
data_dir = "/home/zjlab/liwg/mydata/fashion_mnist/"

def convlog(a, c, d):
    lr = 10 ** (-a)
#   mom = 10 ** (-b)
    alpha = 10 ** (-c)
    batch_size = int(2 ** round(d))
    return lr, alpha, batch_size

def acc_f(lr, mom, p, alpha, batch_size):
    lr, alpha, batch_size = convlog(lr, alpha, batch_size)
    train_loader, valid_loader, test_loader = dataset_creator(use_cuda, data_dir, seed,
                                                                        l_downld=False,
                                                                        batch_size=batch_size)
    input_size = 784  # 28*28
    hidden_size = 100
    num_classes = 10
    # Initialization
    model = Net(input_size, hidden_size, num_classes, p, alpha).to(device)
#   optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2), mom=mom)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom)

    # model.init_weights(ifile)
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
    acc = results['valid accuracy']
    return acc

fspace = {
    'lr': (1., 4.),
    'mom': (0., 0.99),
    'p': (0., 1.),
    'alpha': (1., 5.),
    'batch_size': (3., 7.)
    }

optimizer = BayesianOptimization(f=acc_f, pbounds=fspace, random_state=seed, verbose=0)

since = time.time()
optimizer.maximize(
#   init_points=8,
    n_iter=out_epochs,
    acq = 'ei'
    # 下面为GP 回归参数
#    alpha=1e-3
#    n_restarts_optimizer=5
)
time_elapsed = time.time() - since
print('-' * 30)
print('训练用时： {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

best_acc = optimizer.max['target']
hp_bst = [None] * len(fspace)
hps = optimizer.max['params']
hp_bst[1] = hps['mom']
hp_bst[2] = hps['p']
hp_bst[0], hp_bst[3], hp_bst[-1] = convlog(hps['lr'], hps['alpha'], hps['batch_size'])
print('best validation accuracy:', best_acc)
print('best hp:', hp_bst)
acc_rec = []
for i in range(len(optimizer.res)):
    acc_rec.append(optimizer.res[i]['target'])
print('valid accuracy record:', acc_rec)
np.save('acc_record.npy', acc_rec)
pass
# print('best:', best)

