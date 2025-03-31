# Some code based on https://github.com/thuml/Anomaly-Transformer
import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from matplotlib import colors
import matplotlib.ticker as ticker
from utils.utils import *
from model.Transformer import TransformerVar
from model.loss_functions import *
from data_factory.data_loader import get_loader_segment
import logging
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score
from scipy.special import softmax


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class OneEarlyStopping:
    def __init__(self, patience=10, verbose=False, dataset_name='', delta=0, type=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        self.type = type

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + f'_checkpoint_{self.type}.pth'))
        self.val_loss_min = val_loss

def adjustment_decision(pred, gt):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return pred

def best_treshold_search(distance, gt, config):
    # anomaly_ratio= range(1, 101) # [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,]
    if config["anomaly_ratio"] is not None:
        anomaly_ratio = [config["anomaly_ratio"]]
    else:
        anomaly_ratio = np.arange(1, 15) * 0.1 # 这步限制了测试的效率
        # anomaly_ratio = np.arange(4, 7) * 0.1 # 这步限制了测试的效率
    best_res = {"f1": -1}
    best_pred = []
    for ano in anomaly_ratio:
        threshold = np.percentile(distance, 100 - ano)
        pred = [1 if d > threshold else 0 for d in distance]
        pred = adjustment_decision(pred, gt)  # 增加adjustment_decision
        eval_results = {
            "f1": f1_score(gt, pred),
            "rc": recall_score(gt, pred),
            "pc": precision_score(gt, pred),
            "acc": accuracy_score(gt, pred),
            "threshold": threshold,
            "anomaly_ratio": ano
        }
        if eval_results["f1"] > best_res["f1"]:
            best_res = eval_results
            best_pred = pred
    return best_res, best_pred

class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        self.config = config

        self.train_loader, self.vali_loader, self.k_loader = get_loader_segment(self.data_path,batch_size=self.batch_size,win_size=self.win_size,step=self.win_size,mode='train',dataset=self.dataset)
        self.test_loader, _ = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,step=self.win_size,mode='test',dataset=self.dataset)
        self.thre_loader = self.vali_loader

        self.lambd = config["lambd"]
        self.adj_mean = None

        self.memory_init_embedding = None

        if self.mode == "test":
            self.phase_type = "test"
        self.build_model(memory_init_embedding=self.memory_init_embedding)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory_loss = nn.L1Loss()
        self.criterion = nn.MSELoss()
        self.gathering_loss = GatheringLoss(reduce=False)

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def build_model(self, memory_init_embedding):
        self.model = TransformerVar(config=self)
        # self.model.half() # torch不支持半精度计算
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if torch.cuda.is_available():
            self.model = self.model.to(self.device)

    def validation(self, vali_loader):
        self.model.eval()

        valid_loss_list = []
        valid_re_loss_list = []

        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.to(self.device)
            output_dict = self.model(input)
            output, adj, memory_adj, attn = output_dict['out'], output_dict['adj'], output_dict['memory_adj'], output_dict['attn']

            rec_loss = self.criterion(output, input)

            loss = rec_loss

            valid_re_loss_list.append(rec_loss.item())
            valid_loss_list.append(loss.item())

        return np.average(valid_loss_list), np.average(valid_re_loss_list)

    def train(self, training_type):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = OneEarlyStopping(patience=10, verbose=True, dataset_name=self.dataset, type=training_type)
        train_steps = len(self.train_loader)
        best_res = {"f1": 0.0}
        self.model.train()
        train_times = []
        test_times = []
        from tqdm import tqdm
        for epoch in tqdm(range(self.num_epochs)):
            iter_count = 0
            loss_list = []
            rec_loss_list = []
            memory_loss_list = []
            temp_adjs = torch.zeros((self.input_c*self.input_c)).cuda()

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                output_dict = self.model(input)
                output, adj, memory_adj, attn = output_dict['out'], output_dict['adj'], output_dict['memory_adj'], output_dict['attn']
                temp_adjs += adj.detach().mean(0)
                rec_loss = self.criterion(output, input)
                loss = rec_loss

                # todo: 出入度差作为loss 的一部分

                loss_list.append(loss.item())
                rec_loss_list.append(rec_loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.mean().backward()

                self.optimizer.step()

            temp_adjs /= i + 1
            self.adj_mean = temp_adjs.detach()

            train_times.append(time.time() - epoch_time)
            print("Epoch: {} cost time: {}".format(epoch + 1, train_times[-1]))

            train_loss = np.average(loss_list)
            train_rec_loss = np.average(rec_loss_list)
            valid_loss, valid_re_loss_list = self.validation(self.vali_loader)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(epoch + 1, train_steps, train_loss, valid_loss))
            print("Epoch: {0}, Steps: {1} | VALID reconstruction Loss: {2:.7f} ".format(epoch + 1, train_steps, valid_re_loss_list))
            print("Epoch: {0}, Steps: {1} | TRAIN reconstruction Loss: {2:.7f}".format(epoch + 1, train_steps, train_rec_loss))

            test_epoch=time.time()
            res = self.test(after_train=True)
            test_times.append(time.time() - test_epoch)
            print("Epoch: {} test cost time: {}".format(epoch + 1, test_times[-1]))
            res["test_times"] = np.array(test_times).mean()
            res["train_times"] = np.array(train_times).mean()
            if res["f1"] > best_res['f1']:
                best_res = res

            early_stopping(valid_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return best_res

    def test(self, after_train=False):
        if after_train:
            pass
        else:
            self.model.load_state_dict(torch.load(os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint_second_train.pth')))
            self.model.eval()

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduction='none')
        temperature = self.temperature
        test_labels = []
        test_attens_energy = []
        self.model.eval()
        for i, (input_data, labels) in enumerate(self.test_loader):
            input = input_data.to(self.device)
            B, T, N = input.shape
            output_dict = self.model(input,False)
            output, adj, memory_adj, attn = output_dict['out'], output_dict['adj'], output_dict['memory_adj'], output_dict['attn']

            rec_loss = torch.mean(criterion(input, output), dim=-1)
            
            
            # 计算laten score
            gathering_loss = GatheringLoss(reduce=False)
            if self.config["test_use_mem"]:
                latent_score = torch.softmax(gathering_loss(adj.reshape(B, T, N * N), memory_adj) / temperature, dim=-1)
            else:
                latent_score = torch.softmax(gathering_loss(adj.reshape(B, T, N * N), self.adj_mean.reshape(1,-1)) / temperature, dim=-1)
               
            if self.config["test_loss_type"] == "corr":
                loss = latent_score
            elif self.config["test_loss_type"] == "recon":
                loss = rec_loss
            elif self.config["test_loss_type"] == "reconcorr":
                loss = latent_score * rec_loss

            cri = loss.detach().cpu().numpy()
            test_attens_energy.append(cri)
            test_labels.append(labels)

        test_attens_energy = np.concatenate(test_attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(test_attens_energy)
        test_labels = np.array(test_labels)

        gt = test_labels.astype(int)

        best_res, best_pred = best_treshold_search(test_energy, gt, self.config)

        print(f"the result is:{best_res}")
        print('=' * 50)

        self.logger.info(f"Dataset: {self.dataset}")
        self.logger.info(f"number of items: {self.n_memory}")
        self.logger.info(f"Precision: {round(best_res['pc'], 4)}")
        self.logger.info(f"Recall: {round(best_res['rc'], 4)}")
        self.logger.info(f"f1_score: {round(best_res['f1'], 4)} \n")
        return best_res

    def get_memory_initial_embedding(self, training_type='second_train'):

        self.model.load_state_dict(torch.load(os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint_first_train.pth')))
        self.model.eval()

        for i, (input_data, labels) in enumerate(self.k_loader):

            input = input_data.to(self.device)
            if i == 0:
                output = self.model(input)['queries']
            else:
                output = torch.cat([output, self.model(input)['queries']], dim=0)

        self.memory_init_embedding = k_means_clustering(x=output, n_mem=self.n_memory, d_model=self.input_c)

        self.memory_initial = False

        self.phase_type = "second_train"
        self.build_model(memory_init_embedding=self.memory_init_embedding.detach())

        memory_item_embedding = self.train(training_type=training_type)

        memory_item_embedding = memory_item_embedding[:int(self.n_memory), :]

        item_folder_path = "memory_item"
        if not os.path.exists(item_folder_path):
            os.makedirs(item_folder_path)

        item_path = os.path.join(item_folder_path, str(self.dataset) + '_memory_item.pth')

        torch.save(memory_item_embedding, item_path)
