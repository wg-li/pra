import argparse
import torch
import torch.optim as optim
from fnn import Net
from model_train import train_valid, train, valid, test
from data_prep import dataset_creator
from fix_seed import seed_torch
import numpy as np
import time

class Model(object):
    def __init__(self, hp=None, seed=0, data_dir="/home/zjlab/liwg/mydata/fashion_mnist/"):
        # lr,b1,b2,eps,batch_size = self.init_hp(hp)
        if hp is None:
            hp = [0.01, 0.9, 0.5, 0.01, 64]
        lr = hp[0]
        mom = hp[1]
        p = hp[2]
        alpha = hp[3]
        batch_size = int(hp[-1])
#       print('hp:',lr,b1,b2,eps,batch_size) # 并行下，可用于检查hp和loss对应
        use_cuda = torch.cuda.is_available()
        self.device = device = torch.device("cuda" if use_cuda else "cpu")
#       print(self.device)
        self.train_loader, self.valid_loader, self.test_loader = dataset_creator(use_cuda, data_dir, seed,
                                                                                 l_downld=False,
                                                                                 batch_size=batch_size)
        # print(len(self.train_loader.dataset), len(self.test_loader.dataset))

        # Neurons of each layer
        input_size = 784  # 28*28
        hidden_size = 100
#       print('hidden_size: ', hidden_size)
        num_classes = 10
        # Initialization
        self.model = Net(input_size, hidden_size, num_classes, p, alpha).to(device)
        #self.model = Net().to(device)
        '''
        model.parameters()：卷积网络每一层的参数，主要用来传参给优化器
        通过迭代可以查看，迭代方法如下
        '''
        # for param in self.model.parameters():
        #    print(type(param), param.size())
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=mom)
#        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(b1, b2), eps=eps)

    def train_steps(self, epochs, intv_val=1, l_best=0, l_print=0):
        return train_valid(self.model, self.device, self.train_loader, self.valid_loader,
                           self.optimizer, epochs, intv_val, l_best, l_print)

    def train_proc(self):
        return train(self.model, self.device, self.train_loader, self.optimizer)

    def valid_proc(self):
        return valid(self.model, self.device, self.valid_loader)

    def test_steps(self):
        test_dict = test(self.model, self.device, self.test_loader)
        return test_dict

    def init_weights(self, ifile='../init_weights.npz'):
        try:
            wdata = np.load(ifile)
            # print('load init weights file')
            fc1 = wdata['f1']
            fc2 = wdata['f2']
        except IOError:
            # -----------kaiming_normal
            fc1 = torch.normal(0, np.sqrt(2. / 784), (500, 784)).numpy()
            fc2 = torch.normal(0, np.sqrt(2. / 500), (10, 500)).numpy()
            np.savez(ifile, f1=fc1, f2=fc2)

        # t=self.model.state_dict()
        t = self.get_weights()
        for name, param in self.model.named_parameters():
            #           print(name,type(param), param.size())
            if 'fc1.w' in name:
                a = torch.ones(param.shape) * fc1
                # a=torch.from_numpy(f1)
                t[name].copy_(a)
            if 'fc2.w' in name:
                a = torch.ones(param.shape) * fc2
                # a=torch.from_numpy(f2)
                t[name].copy_(a)
            if 'bias' in name:
                a = torch.zeros(param.shape)
                t[name].copy_(a)

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def save(self, outfile="./model.pt"):
        torch.save(self.model.state_dict(), outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--mom", type=float, default=0.9)
    parser.add_argument("--p", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--data_dir",
        type=str,
#       default="/home/liwg/mydata/fashion_mnist/",
        help="Set the path of the dataset."
    )
    args = parser.parse_args()

    np.set_printoptions(precision=8, suppress=True)
    import setproctitle

    setproctitle.setproctitle("liwg-job")
    hps = [args.lr, args.mom, args.p, args.alpha, args.batch_size]

    ifile = '../init_weights.npz'

    seed_torch()
    model = Model(hps)
    # model = Model(data_dir=args.data_dir)
    # model.init_hp(hps)
    # weight_list = [fc1, fc2]
    # model.init_weights(ifile)
    # print('init weights:',model.get_weights())

    since = time.time()
    results = model.train_steps(epochs=10, intv_val=1, l_best=0, l_print=1)
    time_elapsed = time.time() - since
    best_acc = results['valid accuracy']
    print('-' * 30)
    print('训练用时： {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed % 60))
    print('最高准确率: {}'.format(best_acc))

    acc_test = model.test_steps()
    print("test results: {}".format(acc_test))
    pass
