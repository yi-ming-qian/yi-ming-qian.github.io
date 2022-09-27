import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from networks import get_network
from agent.base import BaseAgent
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio
from util.visualization import fig_to_numpy

class WifiAgent(BaseAgent):
    def __init__(self, config):
        super(WifiAgent, self).__init__(config)

    def build_net(self, config):
        net = get_network('wifi', config).cuda()
        return net

    def set_optimizer(self, config):
        self.optimizer = optim.Adam(self.net.parameters(), config.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.lr_decay)

    def set_loss_function(self):
        self.l2_criterion = nn.MSELoss().cuda()
        self.l1_criterion = nn.L1Loss().cuda()
        #self.bce_criterion = nn.BCEWithLogitsLoss(reduction='none').cuda()

    def update_learning_rate(self):
        """record and update learning rate"""
        self.train_tb.add_scalar('learning_rate', self.optimizer.param_groups[-1]['lr'], self.clock.epoch)
        if self.clock.epoch < 15:#1000
            self.scheduler.step(self.clock.epoch)
    # training and validation
    def forward(self, data):
        ref_locs = data["ref_locs"].cuda()
        src_locs = data["src_locs"].cuda()
        rssi_pairs = data["rssi_pairs"].cuda()
        #print(ref_locs.size(), src_locs.size(), rssi_pairs.size())
        out_locs = self.net(src_locs, rssi_pairs)
        dist_loss = self.l1_criterion(out_locs, ref_locs)

        return None, {"dist": dist_loss}

        
    # testing
    def test_func(self, data):
        with torch.no_grad():
            ref_locs = data["ref_locs"].cuda()
            src_locs = data["src_locs"].cuda()
            rssi_pairs = data["rssi_pairs"].cuda()
            ref_ids = data["ref_ids"]
            #print(ref_locs.size(), src_locs.size(), rssi_pairs.size())
            out_locs = self.net(src_locs, rssi_pairs)
            #out_locs = torch.mean(src_locs, 1, keepdim=True)
        ref_ids = ref_ids.squeeze().numpy()
        out_locs =out_locs.squeeze().cpu().numpy()
        return ref_ids, out_locs
        # print(ref_ids.shape, out_locs.shape)
        # exit()
    def compare_rssi(self, ref_rssi, src_rssi, src_locs):
        error_locs = np.zeros((ref_rssi.shape[0],3))
        for i in range(ref_rssi.shape[0]):
            ref = np.tile(ref_rssi[i:i+1,:], (src_rssi.shape[0],1))
            # rssi_pairs = np.concatenate([ref,src_rssi],1)
            # rssi_pairs = torch.FloatTensor(rssi_pairs).unsqueeze(0).cuda()
            # tmp_locs = torch.zeros(1,rssi_pairs.size(1),2).cuda()
            # with torch.no_grad():
            #     _, dist = self.net(tmp_locs, rssi_pairs)
            # dist = dist.squeeze().cpu().numpy()
            dist = np.mean(np.absolute(ref - src_rssi), axis=1)


            idx = np.argmin(dist)
            ##
            # dist[idx] = 1e10
            # idx = np.argmin(dist)
            ##
            error_locs[i,0] = dist[idx]
            error_locs[i,1:3] = src_locs[idx,:]
        return error_locs
            

    def vis_distance(self, data):
        with torch.no_grad():
            all_locs = data["all_locs"].cuda()
            all_rssi = data["all_rssi"].cuda()
            folder_name = data["folder_name"]
            if folder_name[0] != "20200106014640R_WiFi_SfM":
                return
            vid_l1 = cv2.VideoWriter('./experiments/l1.avi',cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480))
            vid_mlp = cv2.VideoWriter('./experiments/mlp.avi',cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480))
            n = all_rssi.size(1)
            for i in range(n):
                cur_rssi = all_rssi[:,i:i+1,:]
                cur_rssi = cur_rssi.expand_as(all_rssi)
                dist_l1 = torch.mean(torch.abs(cur_rssi - all_rssi),-1)
                cur_rssi = torch.cat([cur_rssi, all_rssi],-1)
                _, dist_mlp = self.net(all_locs, cur_rssi)
                dist_l1 = dist_l1.squeeze().cpu().numpy()
                dist_mlp = dist_mlp.squeeze().cpu().numpy()
                dist_l1 = (dist_l1-np.amin(dist_l1))/(np.amax(dist_l1)-np.amin(dist_l1))
                dist_mlp = (dist_mlp-np.amin(dist_mlp))/(np.amax(dist_mlp)-np.amin(dist_mlp))
                all_locs_cpu = all_locs.squeeze().cpu().numpy()
                cmap = cm.get_cmap('viridis', 256)
                # plot l1
                fig = plt.figure()
                ax=fig.add_subplot(111)
                t=ax.scatter(all_locs_cpu[:,0], all_locs_cpu[:,1], color=cmap(dist_l1), s=8)
                ax.scatter(all_locs_cpu[i,0], all_locs_cpu[i,1], color=(1,0,0), s=10)
                plt.colorbar(t)
                ax.axis('equal')
                plt.tight_layout()
                #img = fig_to_numpy(fig)
                plt.savefig(f"./experiments/gif/{i}-l1.png")
                #img = cv2.imread("./experiments/dist.png")
                #vid_l1.write(img)
                plt.close('all')

                fig = plt.figure()
                ax=fig.add_subplot(111)
                t=ax.scatter(all_locs_cpu[:,0], all_locs_cpu[:,1], color=cmap(dist_mlp), s=8)
                ax.scatter(all_locs_cpu[i,0], all_locs_cpu[i,1], color=(1,0,0), s=10)
                plt.colorbar(t)
                ax.axis('equal')
                plt.tight_layout()
                #img = fig_to_numpy(fig)
                plt.savefig(f"./experiments/gif/{i}-mlp.png")
                #img = cv2.imread("./experiments/dist.png")
                #vid_mlp.write(img)
                plt.close('all')
                
            vid_l1.release()
            vid_mlp.release()

            




