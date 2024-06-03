import numpy as np
from scipy.stats import truncnorm
import sys
from collections import Counter


# -----------the first should be modified---------
def conv2hp(alpha, hplower, hpupper):
    ens_hps = alpha * (hpupper - hplower) + hplower
    hp = ens_hps.copy()
    hp[:, 0] = 10 ** (-ens_hps[:, 0])
    hp[:, 1] = ens_hps[:, 1]
    hp[:, 2] = ens_hps[:, 2]
    hp[:, 3] = 10 ** (-ens_hps[:, 3])
    hp[:, -1] = 2 ** np.round(ens_hps[:, -1])
    return hp
# -----------the first should be modified---------

# def old_rand_uni(num_models, hp_len, alp_base, intv=0.5):
#     samp_vec = np.random.uniform(-intv, intv, (hp_len))
#     for i in range(num_models - 1):
#         rand_arr = np.random.uniform(-intv, intv, (hp_len))
#         samp_vec = np.vstack((samp_vec, rand_arr))
#     alpha = alp_base + samp_vec
#     alpha[alpha > 1] = 1
#     alpha[alpha < 0] = 0
#     return alpha

def rand_uni(num_models, hp_len, alp_base, intv=0.5):
    samp_vec = np.random.uniform(-intv, intv, (num_models))
    for i in range(hp_len - 1):
        rand_arr = np.random.uniform(-intv, intv, (num_models))
        samp_vec = np.vstack((samp_vec, rand_arr))
    alpha = alp_base + samp_vec.T
    alpha[alpha > 1] = 1
    alpha[alpha < 0] = 0
    return alpha

# def old_rand_norm(num_models, hp_len, alp_base, width=0.5, sigma=0.1):
#     # samp_vec=np.random.normal(0,width,(len(hplower)))
#     samp_vec = truncnorm.rvs(-width/sigma, width/sigma, 0, sigma, size=num_models)
#     for i in range(num_models - 1):
#         # rand_arr=np.random.normal(0,width,(len(hplower)))
#         rand_arr = truncnorm.rvs(-width/sigma, width/sigma, 0, sigma, size=num_models)
#         samp_vec = np.vstack((samp_vec, rand_arr))
#     alpha = alp_base + samp_vec
#     alpha[alpha > 1] = 1
#     alpha[alpha < 0] = 0
#     return alpha

def rand_norm(num_models, hp_len, alp_base, width=0.5, sigma=0.1):
    # samp_vec=np.random.normal(0,width,(len(hplower)))
    samp_vec = truncnorm.rvs(-width/sigma, width/sigma, 0, sigma, size=num_models)
    for i in range(hp_len - 1):
        # rand_arr=np.random.normal(0,width,(len(hplower)))
        rand_arr = truncnorm.rvs(-width/sigma, width/sigma, 0, sigma, size=num_models)
        samp_vec = np.vstack((samp_vec, rand_arr))
    alpha = alp_base + samp_vec.T
    alpha[alpha > 1] = 1
    alpha[alpha < 0] = 0
    return alpha

def best_loc(results, metric='valid accuracy'):
    res = []
    for i in range(len(results)):
        res.append(results[i][metric])
    #   print('res:',res)
    res = np.array(res)
    if 'loss' in metric:
        res[np.isnan(res)] = 999
        best_id = np.argmin(res)
        # best_id = res.index(min(res))
    else:
        res[np.isnan(res)] = -999
        best_id = np.argmax(res)
        # best_id = res.index(max(res))
    return res, best_id

def pf_fun(hp_alps, perf, metric, num_models, case_sample, intv, sigma, round_idx):
    # perf = np.array(perf)
    idx = np.where(np.isnan(perf))  # idx -> tuple
    if idx[0].size > 0:
        print('{} NaN exists!!!'.format(idx[0].size))
        perf = perf[~np.isnan(perf)]
        hp_alps = np.delete(hp_alps, idx[0], axis=0)
    if 'loss' in metric:
        pw = np.exp(-0.5 * perf / 0.01)
        # pw = np.exp(-0.5 * perf ** 2 / 0.01)
        # pw = np.exp(-0.5 * perf ** 2 / np.median(perf)**2)
    else:
#       pw = np.exp(50 * (perf - 1))
        pw = np.exp(50 * 1.02**round_idx * (perf - 1))
#       pw = np.exp(-50 * 1.02**(round_idx) / perf)
        # pw = np.exp(-0.5 * (perf - 1) ** 2 / 0.01)
        # pw = np.exp(-0.5 * 0.125 * (perf - 1) ** 2 / (1 - np.median(perf)) ** 2)
    pw = pw / np.sum(pw)
    # alpha_base = np.matmul(hp_alps.T, pw)
    idxes = lv_resample(pw, num_models)
    if len(idxes) != num_models:
        print('error occurs in resampling!!!')
        sys.exit(1)
    idx_dict = Counter(idxes)
    print('index: num', idx_dict)
    cnt = 0
    for i in idx_dict.keys():
        num = idx_dict[i]
        if case_sample == 0: alpha_sub = rand_uni(num, hp_alps.shape[1], hp_alps[i, :], intv)
        if case_sample == 1: alpha_sub = rand_norm(num, hp_alps.shape[1], hp_alps[i, :], intv, sigma)
        if cnt == 0:
            alpha_new = alpha_sub
        else:
            alpha_new = np.vstack((alpha_new, alpha_sub))
        cnt += 1
    alpha_new[alpha_new > 1] = 1
    alpha_new[alpha_new < 0] = 0
    return alpha_new

def rl_resample(pw, num_models):
    print('roulette resampling')
    print('weights: ', pw)
    wcum = np.cumsum(pw)
    wcum = np.hstack((np.array([0.]), wcum))
    idxes = []
    for i in range(num_models):
        r = np.random.rand(1)
        idx = 0
        while idx < wcum.shape[0] - 1:
            if r > wcum[idx] and r < wcum[idx + 1]:
                idxes.append(idx)
            idx += 1
    return idxes

def lv_resample(pw, num_models):
    print('low variance resampling')
    print('weights: ', pw)
    wcum = np.cumsum(pw)
    base = np.cumsum(np.zeros(num_models) + 1. / num_models) - 1. / num_models
    base += np.random.rand(1) / num_models
    idxes = []
    idx = 0
    # 采样NP个粒子 旋转托盘采样 用累加和来选择大权重的粒子
    # 举个例子 目前 基准base 0.1 0.2 0.3 0.4 0.5 0.6 wcum 0.1，0.5 ，0.55,0.6,0.8，那么抽样更多的则是第二个粒子将被选中
    # for ip in range(len(mse)):
    for ip in range(num_models):
        while base[ip] > wcum[idx]:
            #               print(base[ip])
            # 如果该采样累计概率大于权重累计概率 则接着寻找
            idx += 1
            # 如果该采样累计概率小于权重累计概率 则选择该粒子
        idxes.append(idx)
    return idxes
