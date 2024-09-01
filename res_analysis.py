import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import math
embed_dim = [ 8, 16, 32,48]
temporal = [0.001, 0.01, 0.05, 0.1, 0.5]
parameters = {
   "train_use_mem":[False,True],
   "test_use_mem":[False,True],
   "test_update_mem":[False,True],
   "test_loss_type":["recon","corr","reconcorr"],
#    "test_loss_type":["recon","relation","recon-relation"],
   "async_modeling":[False,True],
   "async_type":["mean","max","cross_attn","line"],
   "async_gap":[20,25,30,35,40],
   "node_vec_size":[1,2,4,6,8],
   "d_model":[16,32,64,128]
}
parameters2 = {
   "train_use_mem":[False,True],
   "test_use_mem":[False,True],
   "test_update_mem":[False,True],
#    "test_loss_type":["recon","corr","reconcorr"],
   "test_loss_type":["recon error","relation devision","recon & relation"],
   "async_modeling":[False,True],
   "async_type":["mean","max","cross_attn","line"],
   "async_gap":[20,25,30,35,40],
   "node_vec_size":[1,2,4,6,8],
   "d_model":[16,32,64,128]
}

xname = {
    "test_loss_type":"criterion",
    "async_gap":"async_gap p",
    "node_vec_size":"embedding_size m",
    "train_use_mem":"train_use_mem",
    "test_use_mem":"test_use_mem",
    "test_update_mem":"test_update_mem",
    "async_modeling":"async_modeling",
    "d_model":"d_model",
}

def str2bool(v):
    return v.lower() in ('true')

def get_res(config):
    temp = []
    datasets = ["SMD", "SMAP", "MSL", "PSM", "SWAT"]
    for d in datasets:
        config['dataset'] = d
        filename = f"./res2/{config['dataset']}_tum{config['train_use_mem']}_tpm{config['test_update_mem']}_eum{config['test_use_mem']}_tlt{config['test_loss_type']}_am{config['async_modeling']}_at{config['async_type']}_ag{config['async_gap']}_nvs{config['node_vec_size']}_dm{config['d_model']}.txt"
        with open(filename, "r") as f:
            res = json.load(f)
            temp.append(res['f1'])
            temp.append(res['rc'])
            temp.append(res['pc'])
    return np.array(temp).reshape(len(datasets), 3)

def get_all_data_for_diff_config(config, config_name = "", ):
    temp = []
    config_x = parameters[config_name]
    for x in config_x:
        config[f'{config_name}'] = x
        res = get_res(config)
        temp.append(res)
    return np.array(temp).reshape(len(config_x), 5, 3)

def plot_for_diff_config(config, config_name = "train_use_mem"):
    config = vars(config)
    x =  parameters2[config_name]
   
    data = get_all_data_for_diff_config(config, config_name)
    data = data[:, :, 0]
    smd_res = data[:,0]
    smap_res = data[:,1]
    msl_res = data[:,2]
    psm_res = data[:,3]
    swat_res = data[:,4]
    avg_res = data.mean(1)

    plt.cla()
    plt.plot(x, smd_res, "p-c", label="SMD", alpha=0.7)
    # [plt.text(a, b, str(round(b, 4))) for a,b in zip(x,smd_res)]
    plt.plot(x, smap_res, "^:g", label="SMAP", alpha=0.7)
    # [plt.text(a, b, str(round(b, 4))) for a,b in zip(x,smap_res)]
    plt.plot(x, msl_res, "P--b", label="MSL", alpha=0.7)
    # [plt.text(a, b, str(round(b, 4))) for a,b in zip(x,msl_res)]
    plt.plot(x, psm_res, "D-m", label="PSM", alpha=0.7) 
    # [plt.text(a, b, str(round(b, 4))) for a,b in zip(x,psm_res)]
    plt.plot(x, swat_res, "h:y", label="SWaT", alpha=0.7) 
    # [plt.text(a, b, str(round(b, 4))) for a,b in zip(x,swat_res)]
    plt.plot(x, avg_res, "s-.r", label="AVG", alpha=0.7)
    # [plt.text(a, b, str(round(b, 4))) for a,b in zip(x,avg_res)]
    plt.grid()
    plt.xlabel(f"{xname[config_name]}")
    plt.ylabel("F1-Score")
    # plt.ylim(0.85,0.95)
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.07), ncol=4)

    plt.savefig(f"analysis2/{config_name}_pure.pdf")


def grid_search():
    pass

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
    # parser.add_argument('--data_path', type=str, default='../Anomaly-Transformer/dataset/MSL/')
    parser.add_argument('--data_path', type=str, default='../Anomaly-Transformer/dataset/SMD/')
    # parser.add_argument('--data_path', type=str, default='../Anomaly-Transformer/dataset/SMAP/')
    # parser.add_argument('--data_path', type=str, default='../Anomaly-Transformer/dataset/PSM/')
    # parser.add_argument('--data_path', type=str, default='../TSAD/data/SWAT/A1_A2/')
    
    # train parameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
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
    parser.add_argument('--test_use_mem', type=str2bool, default=True, help='') # 消融实验，有点用
    parser.add_argument('--test_update_mem', type=str2bool, default=True, help='') # 消融实验，本项目下默认使用，没用
    parser.add_argument('--test_loss_type', type=str, default="reconcorr", choices=["recon","corr","reconcorr"], help='') # 消融实验，有用
    
    # async modeling
    parser.add_argument('--async_modeling',type=str2bool,default=True,help='') # 有点用 
    parser.add_argument('--async_type',type=str,default="line", choices=["mean","max","cross_attn","line"] ,help='') # 消融实验， 有用
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
    
    # plot_for_diff_config(config,"test_use_mem")
    # plot_for_diff_config(config,"test_update_mem")
    # plot_for_diff_config(config,"test_loss_type")
    # plot_for_diff_config(config,"async_modeling")
    # plot_for_diff_config(config,"async_type")
    plot_for_diff_config(config,"async_gap")
    # plot_for_diff_config(config,"node_vec_size")
    # plot_for_diff_config(config,"d_model")
    
    # display single
    # res = get_res(vars(config))
    # print(f"avg f1: {res[:,0].mean()}")
    # print(f"f1: {res[:,0]}")
    # print(f"rc: {res[:,1]}")
    # print(f"pc: {res[:,2]}")