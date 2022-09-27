from torch.utils.data import Dataset
import torch
import os
import json
import glob
import numpy as np
import math
import random
import time
import cv2
import glob
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from .reducedronin import ReducedRonin


class WifiDataset(Dataset):
    def __init__(self, phase, config):
        super(WifiDataset, self).__init__()
        self.is_train = phase=="train"
        self.data_root = config.data_root
        self.ronin_outdir = config.ronin_outdir
        folder_names = []
        with open(self.ronin_outdir + "folder_list_auto.txt", 'r') as handle:
            for line in handle:
                folder_names.append(line.strip())
        self.folder_names = folder_names
        # read ronin info
        self.all_ronins = []

        for i, folder_name in enumerate(folder_names):
            out_path = self.ronin_outdir+folder_name+"/"
            rro = ReducedRonin(out_path)
            if config.iteration==0:
                init_name = out_path+"c_multi_corres_align_sparse.txt"
            else:
                init_name = config.exp_dir.split('-')[0] + f"-{config.iteration-1}"
                init_name = init_name + f"/results/ckpt-{config.ckpt}/{folder_name}-corres_pos_align.txt"
            rro.read_reduced_ronin(init_name)
            self.all_ronins.append(rro)
            # print(rro.ronin_traj.shape, rro.rssi.shape, rro.haswifi.shape)
        ###### plot
        # fig = plt.figure()
        # ax=fig.add_subplot(111)
        # for rro in self.all_ronins:
        #     ax.scatter(rro.ronin_traj[:,0], rro.ronin_traj[:,1], color=(0.5,0.5,0.5), s=0.01)
        ###### plot

        # calculate ids for each rssi-available position
        all_ori_ids = []
        for i, r in enumerate(self.all_ronins):
            n = r.ronin_traj.shape[0]
            ori_ids = np.arange(n)
            ori_ids = ori_ids[r.haswifi]
            rssi_num = r.rssi.shape[0]
            ori_ids = [np.full(rssi_num,i), ori_ids, np.arange(rssi_num)]
            ori_ids = np.stack(ori_ids, -1)
            all_ori_ids.append(ori_ids)
        self.corres_mats = []
        for i, ref in enumerate(self.all_ronins):
            other_ids = all_ori_ids[:i]+all_ori_ids[i+1:]
            other_pos = []
            for j, src in enumerate(self.all_ronins):
                if i==j:
                    continue
                other_pos.append(src.ronin_traj[src.haswifi,:])
            other_ids = np.concatenate(other_ids)
            other_pos = np.concatenate(other_pos)
        
            knn_num = 10
            nbrs = NearestNeighbors(n_neighbors=knn_num, algorithm='ball_tree', metric='manhattan').fit(other_pos)
            distances, indices = nbrs.kneighbors(ref.ronin_traj[ref.haswifi,:])
            #print(ref.rssi.shape, distances.shape, indices.shape)
            n = indices.shape[0]
            my_ids = all_ori_ids[i]
            for j in range(n):
                if distances[j,0]>20:
                    continue
                line = np.concatenate([my_ids[j,:], other_ids[indices[j,:],:].reshape(-1)])
                self.corres_mats.append(line.reshape(1,-1))
        self.corres_mats = np.concatenate(self.corres_mats)
        ###### plot
        # for i, r in enumerate(self.all_ronins):
        #     flag = self.corres_mats[:,0] ==i
        #     flag = self.corres_mats[flag,1]
        #     ax.scatter(r.ronin_traj[flag,0], r.ronin_traj[flag,1], color=(1,0,0), s=0.1)
        # ax.axis('equal')
        # plt.tight_layout()
        # plt.savefig(f'./experiments/test1.png')
        # plt.close('all')
        ###### plot
    def get_corres_num(self):
        c = self.corres_mats[:,0]
        cn = []
        for i in range(len(self.all_ronins)):
            cn.append(np.sum(c==i))
        return np.array(cn)

    def __len__(self):
        return len(self.all_ronins)

    def __getitem__(self, index):
        # if self.is_train:
        #     t = int(time.time() * 1000000)
        #     random.seed(((t & 0xff000000) >> 24) +
        #                    ((t & 0x00ff0000) >> 8) +
        #                    ((t & 0x0000ff00) << 8) +
        #                    ((t & 0x000000ff) << 24))
        # else:
        #     random.seed(index)

        r = self.all_ronins[index]
        all_locs = r.ronin_traj[r.haswifi,:]

        sample = {
            "all_locs": torch.FloatTensor(all_locs),
            "all_rssi": torch.FloatTensor(r.rssi),
            "folder_name": self.folder_names[index]
        }
        return sample