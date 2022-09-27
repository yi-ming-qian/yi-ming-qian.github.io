import os
from collections import OrderedDict
from tqdm import tqdm
from config import get_config
from util.utils import cycle, ensure_dir
from util.visualization import labelcolormap
from dataset import get_dataloader
from agent import get_agent
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataset.reducedronin import ReducedRonin


def train(config):

    # create network and training agent
    tr_agent = get_agent(config)

    # load from checkpoint if provided
    if config.cont:
        tr_agent.load_ckpt(config.ckpt)

    # create dataloader
    train_loader = get_dataloader('train', config)
    #cv2.setNumThreads(0)

    # start training
    clock = tr_agent.clock

    for e in range(clock.epoch, config.nr_epochs):
        # begin iteration
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # train step
            losses = tr_agent.train_func(data)

            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

            clock.tick()

        # update lr by scheduler
        #tr_agent.update_learning_rate()

        clock.tock()
        if clock.epoch % config.save_frequency == 0:
            tr_agent.save_ckpt()
        #tr_agent.save_ckpt('latest')

def test(config):
    config.batch_size = 1
    config.num_worker = 1

    test_agent = get_agent(config)
    test_agent.load_ckpt(config.ckpt)
    test_agent.net.eval()
    test_loader = get_dataloader('test', config)

    save_dir = os.path.join(config.exp_dir, f"results/ckpt-{config.ckpt}/")
    ensure_dir(save_dir)

    folder_names = []
    with open(config.ronin_outdir + "folder_list_auto.txt", 'r') as handle:
        for line in handle:
            folder_names.append(line.strip())
    corres_num = test_loader.dataset.get_corres_num()
    handles = []
    for i, folder_name in enumerate(folder_names):
        handles.append(open(save_dir+f"{folder_name}-corres_pos.txt", 'w'))
        handles[-1].write(str(corres_num[i])+"\n")

    pbar = tqdm(test_loader)
    for i, data in enumerate(pbar):
        ref_ids, out_locs = test_agent.test_func(data)
        handles[ref_ids[0]].write(f"{ref_ids[0]}\t{ref_ids[1]}\t{out_locs[0]}\t{out_locs[1]}\n")
    
    for i in handles:
        i.close()

def plot(config):
    folder_names = []
    with open(config.ronin_outdir + "folder_list_auto.txt", 'r') as handle:
        for line in handle:
            folder_names.append(line.strip())
    all_ronins = []
    for i, folder_name in enumerate(folder_names):
        out_path = config.ronin_outdir+folder_name+"/"
        rro = ReducedRonin(out_path)
        #init_name = out_path+"c_multi_corres_align_sparse.txt"
        init_name = config.exp_dir + f"/results/ckpt-{config.ckpt}/{folder_name}-corres_pos_align.txt"
        rro.read_reduced_ronin(init_name)
        all_ronins.append(rro)
    # plot
    t_colors = labelcolormap(max(40,len(all_ronins)+1))/255.
    fig = plt.figure()
    ax=fig.add_subplot(111)
    for i, rro in enumerate(all_ronins):
        ax.scatter(rro.ronin_traj[:,0], rro.ronin_traj[:,1], color=t_colors[i,:], s=0.01)
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(config.exp_dir + f"/results/ckpt-{config.ckpt}/all_traj.png")
    plt.close('all')
    #exit()
    # localization
    test_agent = get_agent(config)
    test_agent.load_ckpt(config.ckpt)
    test_agent.net.eval()
    sum_pts_all = 0
    num_pts_all = 0
    dist_all = []
    for i in range(len(all_ronins)):
        print("Day "+str(i))
        # i is the reference ronin
        print(all_ronins[i].output_path)
        ref_r = ReducedRonin(all_ronins[i].output_path)
        init_name = config.ronin_outdir+folder_names[i]+"/c_single_corres_align_sparse.txt"
        ref_r.read_reduced_ronin(init_name)
        error_locs = []
        for j in range(len(all_ronins)):
            # j is the source ronin
            if i==j:
                continue
            j_traj = all_ronins[j].ronin_traj[all_ronins[j].haswifi,:]
            error_locs.append(test_agent.compare_rssi(ref_r.rssi, all_ronins[j].rssi, j_traj))
        error_locs = np.stack(error_locs,0)
        error = error_locs[:,:,0]
        idx = np.argmin(error, axis=0)
        #print(error_locs[idx,np.arange(len(idx)),0])
        new_locs = error_locs[idx,np.arange(len(idx)),1:3]
        
        # per-point localization
        fig = plt.figure()
        ax=fig.add_subplot(111)
        for j, rro in enumerate(all_ronins):
            if i==j:
                continue
            ax.scatter(rro.ronin_traj[:,0], rro.ronin_traj[:,1], color=(0.5,0.5,0.5), s=0.1)
        ax.scatter(new_locs[:,0], new_locs[:,1], color=(1,0,0), s=1)
        ax.axis('equal')
        plt.tight_layout()
        plt.savefig(f"./experiments/localization/{folder_names[i]}-localization.png")
        plt.close('all')

        #init
        fig = plt.figure()
        ax=fig.add_subplot(111)
        for j, rro in enumerate(all_ronins):
            if i==j:
                continue
            ax.scatter(rro.ronin_traj[:,0], rro.ronin_traj[:,1], color=(0.5,0.5,0.5), s=0.1)
        
        old_locs = ref_r.ronin_traj[ref_r.haswifi,:]
        ax.scatter(old_locs[:,0], old_locs[:,1], color=(1,0,0), s=1)
        ax.axis('equal')
        plt.tight_layout()
        plt.savefig(f"./experiments/localization/{folder_names[i]}-singlealign.png")
        plt.close('all')

        # optimized
        fig = plt.figure()
        ax=fig.add_subplot(111)
        for j, rro in enumerate(all_ronins):
            if i==j:
                continue
            ax.scatter(rro.ronin_traj[:,0], rro.ronin_traj[:,1], color=(0.5,0.5,0.5), s=0.1)
        
        old_locs = all_ronins[i].ronin_traj[all_ronins[i].haswifi,:]
        # r = ReducedRonin(all_ronins[i].output_path)
        # init_name = config.exp_dir + f"/results/ckpt-{config.ckpt}/{folder_names[i]}-corres_pos_align.txt"
        # r.read_reduced_ronin(init_name)
        # old_locs = r.ronin_traj[r.haswifi,:]

        ax.scatter(old_locs[:,0], old_locs[:,1], color=(1,0,0), s=1)
        ax.axis('equal')
        plt.tight_layout()
        plt.savefig(f"./experiments/localization/{folder_names[i]}-optimized.png")
        plt.close('all')
        dist = np.sqrt(np.sum(np.square(new_locs - old_locs), axis=1))
        sum_pts_all += np.sum(dist)
        num_pts_all += dist.shape[0]
        dist_all.append(np.mean(dist))
    dist_all = np.asarray(dist_all)
    np.save("./experiments/localization/localize-optimize.npy", dist_all)
    print(f'total_average is {np.mean(dist_all)}, median is {np.median(dist_all)}')
        



def visualize(config):
    config.batch_size = 1
    config.num_worker = 1

    test_agent = get_agent(config)
    test_agent.load_ckpt(config.ckpt)
    test_agent.net.eval()
    test_loader = get_dataloader('test', config)

    save_dir = os.path.join(config.exp_dir, f"results/ckpt-{config.ckpt}/")
    ensure_dir(save_dir)

    pbar = tqdm(test_loader)
    for i, data in enumerate(pbar):
        test_agent.vis_distance(data)
        

if __name__ == '__main__':
    config = get_config('wifi')()

    if config.train:
        train(config)
    elif config.test:
        test(config)#visualize(config)
    elif config.plot:
        plot(config)
    else:
        raise NotImplementedError
