import torch
import numpy as np
from fnn import Net
from model_train import train, valid, test
from data_prep import dataset_creator
from skopt import forest_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Integer
from fix_seed import seed_torch
import time
from pytorchtools import EarlyStopping
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-o", '--oep', type=int, default=10, help='outer epochs')
parser.add_argument("-i", '--iep', type=int, default=2, help='inner epochs')
parser.add_argument("-s", '--seed', type=int, default=2, help='seed')
args = parser.parse_args()

out_epochs = args.oep
in_epochs = args.iep
seed = args.seed

if out_epochs < 10:
    out_epochs = 10
seed_torch(seed)

space = [
    Real(10**(-4), 10**(-1), 'log-uniform', name='lr'),
    Real(0, 0.99, 'uniform', name='mom'),
    Real(0., 1., name='p'),
    Real(10**(-5), 10**(-1), 'log-uniform', name='alpha'),
    Integer(3, 7, 'uniform', name='batch_size')
    ]

@use_named_args(space)
def loss_f(lr, mom, p, alpha, batch_size):
    batch_size = 2 ** batch_size
    print('batch_size: ', batch_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    data_dir = "/home/zjlab/liwg/mydata/fashion_mnist/"
    train_loader, valid_loader, test_loader = dataset_creator(use_cuda, data_dir, seed,
                                                              l_downld=False,
                                                              batch_size=int(batch_size))
    input_size = 784  # 28*28
    hidden_size = 100
    num_classes = 10
    # Initialization
    model = Net(input_size, hidden_size, num_classes, p, alpha).to(device)
#   optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2), eps=eps)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom)

#   torch.save(model.state_dict(),'./smac_weight.pt')

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

since = time.time()
best = forest_minimize(loss_f, space, n_calls=out_epochs, verbose=True, 
        random_state=seed, base_estimator="RF", acq_func="EI")
time_elapsed = time.time() - since
print('-' * 30)
print('训练用时： {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print("Best fitness: " + str(-best.fun))
print("Best parameters: {}".format(best.x))

acc_rec=[]
print('trials:')
for i in range(len(best.func_vals)):
    print('hps:', best.x_iters[i])
    acc_rec.append(-best.func_vals[i])
#   print('valid acc:', -trial['result']['loss'])
print('valid accuracy record:', acc_rec)
np.save('acc_record.npy', acc_rec)
pass
