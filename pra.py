import argparse
import ray
import torch
import os
import numpy as np
from mymodel import Model
from fix_seed import seed_torch
import time
from func_pfbo import rand_uni, conv2hp, best_loc, pf_fun
# from collections import OrderedDict
import setproctitle

setproctitle.setproctitle("liwg-job")

np.set_printoptions(precision=8, suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("-n", '--num', type=int, default=2, help='no. of models')
parser.add_argument("-o", '--oep', type=int, default=2, help='outer epochs')
parser.add_argument("-i", '--iep', type=int, default=2, help='inner epochs')
parser.add_argument("-s", '--sam', type=int, default=0, help='sampling method')
parser.add_argument("-d", '--seed', type=int, default=0, help='seed')
# parser.add_argument("-w", '--wet', type=int, default=0, help='weight calc. method')

args = parser.parse_args()

num_models = args.num
out_epochs = args.oep
in_epochs = args.iep
seed = args.seed

seed_torch(seed)

alpha_bs = 0.5
# 开关，方法选择
l_cluster = 0
l_his = 0
# 0: uniform; 1: normal
case_sample = args.sam
if case_sample == 0: intv = 0.1; sigma = 'none'
if case_sample == 1: intv = 0.1; sigma = 0.05
# 0: avgW; 1: best
# case_weight = args.wet
perf_metric = 'valid accuracy'
l_best_in = 0
l_print_in = 0
l_test = 0
intv_test = 2
intv_val = in_epochs
if l_best_in: intv_val = 1

hplower = np.array([1.0, 0., 0., 1., 3])
hpupper = np.array([4.0, 0.99, 1., 5., 7])
hp_len = len(hplower)
ifile = './init_weights.npz'

print("-----------------0.configuration and initialization")
print('case_sample 0: uniform; 1: normal')
# print('case_weight 0: avgWeight; 1: bestWeight')
print('')
print('---switches setting---')
print('l_cluster: ', l_cluster)
print('case sample: {}, sample radius: {}, sigma: {}'.format(case_sample, intv, sigma))
# print('case weight: ', case_weight)
print('performance metric:', perf_metric)
print('l_best_in: ', l_best_in)
print('l_print_in: ', l_print_in)

hp_alp = rand_uni(num_models, hp_len, alpha_bs, 0.5)
ens_hp = conv2hp(hp_alp, hplower, hpupper)
print('initialize ensemble hps:\n', ens_hp)

print("-----------------1.create multi training models")
print("-----------------number of models:", num_models)
# 初始化ray集群连接
ray.shutdown()
runtime_env = {"working_dir": "./"} #,
               # "pip": ["numpy", "os", "scipy", "torch", "torchvision", "filelock", "sklearn"]}

if l_cluster:
    ray.init(address='10.101.104.44:6379', runtime_env=runtime_env)
#   RemoteNetwork = ray.remote(Network)
#   RemoteNetwork = ray.remote(num_cpus=0.8, num_gpus=0.1)(Network)
    # ray.init(address='auto',runtime_env=runtime_env})
    # ray.init(runtime_env=runtime_env)
    # ray.init('ray://10.101.104.45:6379:10001', runtime_env=runtime_env)
else:
    ray.init(num_cpus=num_models, num_gpus=1)#, object_store_memory=1073741824*2)
#   RemoteNetwork = ray.remote(num_cpus=0.8, num_gpus=0.1)(Network)


# 注：不在类上直接加装饰器，而是通过remote来初始化Network神经网络为一个actor角色
RemoteModel = ray.remote(num_cpus=0.8, num_gpus=0.10)(Model)
# Use the below instead of `ray.remote(network)` to leverage the GPU.
# RemoteNetwork = ray.remote(num_gpus=1)(Network)

ModelActors = [RemoteModel.remote(ens_hp[i, :], seed) for i in range(num_models)]
# weights = ray.get([n.get_weights.remote() for n in ModelActors])
# NetworkActors = [RemoteNetwork.remote() for _ in range(num_models)]
if os.path.exists(ifile):
    print('initialize weights from ' + ifile)
    [n.init_weights.remote(ifile) for n in ModelActors]
else:
    print('initial weight fixed by seed')
#--------check weights between differnt models
#--------check weights between differnt methods (bo, pbt, etc.)
#   weights = ray.get([n.get_weights.remote() for n in ModelActors])
#   torch.save(weights[0], 'weight0.pt')
#   torch.save(weights[1], 'weight1.pt')

best_acc_out = 0.
acc_rec = []
loss_rec = []
his_perf = np.array([])
print("-----------------2.start training (epochs)")
print("-----------------total outer epochs", out_epochs)
print("-----------------total inner epochs", in_epochs)
since = time.time()
for out_epoch in range(out_epochs):
    print("-----------------outer epoch:", out_epoch)
    # n个模型并发同时训练
    print('++parallel training')
    print("-----------------inner epochs:")
    results = ray.get([n.train_steps.remote(epochs=in_epochs, intv_val=intv_val,
                                            l_best=l_best_in, l_print=l_print_in)
                       for n in ModelActors])
    # print("{} inner epoch, train results:\n{} ".format(in_epoch, results))
    print("after {} inner epoch, train results: ".format(in_epochs))
    [print('M.{}: {}'.format(i, results[i])) for i in range(num_models)]

    perf, best_id = best_loc(results, metric=perf_metric)
    if l_his:
        his_perf = np.concatenate((his_perf, perf))


    print('best_id', best_id)
    hp_bst = ens_hp[best_id, :]
    print('best hp:', hp_bst)
    # weights = ray.get([n.get_weights.remote() for n in ModelActors])
    new_weight = ray.get(ModelActors[best_id].get_weights.remote())

    if l_test:
        if (out_epoch+1) % intv_test == 0:
            model = Model(hp_bst, seed)
            # model.set_weights(weights[best_id])
            model.set_weights(new_weight)
            print("test results: ", model.test_steps())
            del model

    # output best model
    if results[best_id]['valid accuracy'] > best_acc_out:
        best_acc_out = results[best_id]['valid accuracy']
        best_info = results[best_id]
        # best_wts_out = weights[best_id]
        best_wts_out = new_weight
        best_hp_out = hp_bst

    acc_rec.append(results[best_id]['valid accuracy'])
    loss_rec.append(results[best_id]['valid loss'])

    if out_epoch == out_epochs - 1:
        continue

    if l_his:
        hp_alp = pf_fun(his_alp, his_perf, perf_metric, num_models, case_sample, intv, sigma)
        his_alp = np.vstack((his_alp, hp_alp))
    else:
        hp_alp = pf_fun(hp_alp, perf, perf_metric, num_models, case_sample, intv, sigma, out_epoch)
    # hp_alp = rearrange(pw, num_models, case_sample, intv, sigma)
    # if case_sample == 0: hp_alp = rand_uni(num_models, hp_len, alpha_bs, intv)
    # if case_sample == 1: hp_alp = rand_norm(num_models, hp_len, alpha_bs, intv, sigma)
    ens_hp = conv2hp(hp_alp, hplower, hpupper)
    print('next ensemble hps:\n', ens_hp)

    print("++reset parameters:weight & bias")
    # if case_weight == 0: new_weight = avg_weight(weights, pw)
    # if case_weight == 1: new_weight = weights[best_id]
    # new_weight = weights[best_id]

    # next turn setting
    del ModelActors
    torch.cuda.empty_cache()
#   time.sleep(5)
    ModelActors = [RemoteModel.remote(ens_hp[i, :], seed) for i in range(num_models)]
    # 重置权重
    weight_id = ray.put(new_weight)
    [n.set_weights.remote(weight_id) for n in ModelActors]

time_elapsed = time.time() - since
print('-' * 30)
print('训练用时： {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

ray.shutdown()
print('-----------------3.start testing')
model = Model(best_hp_out, seed)
model.set_weights(best_wts_out)
acc_test = model.test_steps()
print('bets hp:', best_hp_out)
print('bets info:', best_info)
print("test results: {}".format(acc_test))
print('validation accuracy record:\n', acc_rec)
np.save('acc_record.npy', acc_rec)
print('validation accuracy loss:\n', loss_rec)
np.save('loss_record.npy', loss_rec)
pass

