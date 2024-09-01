import os
import argparse
import torch
from torch.backends import cudnn
from utils.utils import *
import numpy as np
import json
from solver import Solver

# 固定随机种子
import torch
import random
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
seed_everything()


def str2bool(v):
    return v.lower() in ('true')


def save(best_res, config):
    with open(config.filename, "w") as fw:
        json.dump(best_res, fw)

def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    
    res = {"f1":0.0}
    # try:  
    #     with open(config.filename, "r") as fr:
    #         res = json.load(fr)
    # except Exception as e:
    #     pass
    # if res["f1"] > 0.0:
    #     print(res)
    #     return res
    
    solver = Solver(vars(config))
    best_res = {}
    if config.mode == 'train':
        best_res = solver.train(training_type='first_train')
    elif config.mode == 'test':
        solver.test()
    elif config.mode == 'memory_initial':
        solver.get_memory_initial_embedding(training_type='second_train')

    save(best_res, config)
    return solver

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # data parameters
    # parser.add_argument('--dataset', type=str, default='SWAT')
    # parser.add_argument('--dataset', type=str, default='MSL')
    # parser.add_argument('--dataset', type=str, default='SMAP')
    # parser.add_argument('--dataset', type=str, default='PSM')
    parser.add_argument('--dataset', type=str, default='SMD')
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    # parser.add_argument('--data_path', type=str, default='dataset/MSL/')
    parser.add_argument('--data_path', type=str, default='dataset/SMD/')
    # parser.add_argument('--data_path', type=str, default='dataset/SMAP/')
    # parser.add_argument('--data_path', type=str, default='dataset/PSM/')
    # parser.add_argument('--data_path', type=str, default='dataset/SWaT/')
    # parser.add_argument('--times', type=int, default=1)

    
    # train parameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16) # 参数内存敏感
    parser.add_argument('--lambd', type=float, default=1)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'memory_initial'])
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--gpu', type=str, default="1", help='')
    parser.add_argument('--anomaly_ratio', type=float, default=None, help='')
    
    # model parameters
    parser.add_argument('--d_model', type=int, default=128) # 参数内存不敏感
    # parser.add_argument('--d_ff', type=int, default=128) 
    parser.add_argument('--win_size', type=int, default=100, help='input sequence length')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument("--node_vec_size", type=int, default=8, help='') # 参数内存敏感 # 细粒度建模指标间关系 

    # mem parameters
    parser.add_argument('--n_memory', type=int, default=1, help='number of memory items')
    parser.add_argument('--temperature', type=float, default=0.01, choices=[0.001,0.01, 0.05, 0.1, 0.2, 0.5])
    parser.add_argument('--train_use_mem', type=str2bool, default=True, help='') # 没啥卵用，去掉吧，默认使用得了
    parser.add_argument('--test_use_mem', type=str2bool, default=True, help='') # 消融实验
    parser.add_argument('--test_update_mem', type=str2bool, default=True, help='') # 消融实验，本项目下默认使用
    parser.add_argument('--test_loss_type', type=str, default="reconcorr", choices=["recon","corr","reconcorr"], help='') # 消融实验
    
    # async modeling
    parser.add_argument('--async_modeling',type=str2bool,default=True,help='')
    parser.add_argument('--async_type',type=str,default="line", choices=["mean","max","cross_attn","line"] ,help='') # 消融实验
    parser.add_argument('--async_gap',type=int,default=20,help='') # 参数内存敏感

    config = parser.parse_args()
    l = np.zeros(config.win_size)[::config.async_gap]
    config.async_size = len(l) # 如果config.async_gap >= config.win_size, 异步的指标间相关性则降级为特定的异步指标间相关性,  则只有成为K:1,
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    config.d_ff = config.d_model * 4
# train_use_mem 
# test_use_mem
# test_update_mem
# test_loss_type recon corr recon&corr
# async_modeling
# async_type mean max cross_attn line

# async_gap 20 25 30 35 40 
# node_vec_size 1 2 4 6 8 
# d_model 16 32 64 128
    config.filename = f"./res2/{config.dataset}_tum{config.train_use_mem}_tpm{config.test_update_mem}_eum{config.test_use_mem}_tlt{config.test_loss_type}_am{config.async_modeling}_at{config.async_type}_ag{config.async_gap}_nvs{config.node_vec_size}_dm{config.d_model}.txt"
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
