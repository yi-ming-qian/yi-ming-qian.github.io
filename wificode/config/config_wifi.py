import os
import shutil
import json
import argparse
from util.utils import ensure_dirs


class WifiConfig(object):
    """Base class of Config, provide necessary hyperparameters.
    """

    def __init__(self):

        # init hyperparameters and parse from command-line
        parser, args = self.parse()

        phase = "train" if args.train else "test"
        self.is_train = phase == "train"

        # set as attributes
        print("----Experiment Configuration-----")
        for k, v in args.__dict__.items():
            print("{0:20}".format(k), v)
            self.__setattr__(k, v)

        # experiment paths
        
        self.exp_dir = os.path.join('experiments', args.proj_dir+"-"+str(args.iteration))
        self.log_dir = os.path.join(self.exp_dir, 'log_{}'.format(args.module))
        self.model_dir = os.path.join(self.exp_dir, 'model_{}'.format(args.module))

        if phase == "train" and args.cont is not True and os.path.exists(self.log_dir):
            response = input('Experiment log/model already exists, overwrite to retrain? (y/n) ')
            if response != 'y':
                exit()
            shutil.rmtree(self.log_dir)
            shutil.rmtree(self.model_dir)

        ensure_dirs([self.log_dir, self.model_dir])

        # GPU usage
        self.parallel = False
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
            if len(str(args.gpu_ids).split(',')) > 1:
                self.parallel = True

        # create soft link to experiment log directory
        # if not os.path.exists('train_log'):
        #     os.symlink(self.exp_dir, 'train_log')

        # save this configuration
        if self.is_train:
            with open(os.path.join(self.log_dir, 'config_{}.txt'.format(args.module)), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    def parse(self):
        """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
        parser = argparse.ArgumentParser()

        # basic configuration
        self._add_basic_config_(parser)

        # dataset configuration
        self._add_dataset_config_(parser)

        # model configuration
        self._add_network_config_(parser)

        # training configuration
        self._add_training_config_(parser)

        #if not self.is_train:
            # testing configuration
        self._add_testing_config_(parser)

        # additional parameters if needed
        pass

        args = parser.parse_args()
        return parser, args

    def _add_basic_config_(self, parser):
        """add general hyperparameters"""
        group = parser.add_argument_group('basic')
        group.add_argument('--proj_dir', type=str, default="proj_log",
                           help="path to project folder where experiment logs/models will be saved")
        group.add_argument('--iteration', type=int, default="0",
                           help="path to project folder where experiment logs/models will be saved")
        group.add_argument('--exp_name', type=str, default=None, help="name of this experiment. "
                           "Automatically generated based on data category if not provided.")
        group.add_argument('-g', '--gpu_ids', type=str, default="0",
                           help="gpu to use, e.g. 0  0,1,2. CPU not supported.")
        group.add_argument('--module', type=str, choices=['wifi'], required=True,
                           help="which network module to set. use 'seq2seq' when testing.")

        mgroup  = parser.add_mutually_exclusive_group(required=True)
        mgroup.add_argument('--train', action='store_true', help="enter training phase for model training")
        mgroup.add_argument('--test', action='store_true', help="enter testing phase for generating fakes")
        mgroup.add_argument('--plot', action='store_true', help="enter testing phase for generating fakes")

    def _add_dataset_config_(self, parser):
        """add hyperparameters for dataset configuration"""
        group = parser.add_argument_group('dataset')
        group.add_argument('--data_root', type=str, default="data", help="file path to data")
        group.add_argument('--ronin_outdir', type=str, default="data", help="file path to data")
        group.add_argument('--num_workers', type=int, default=8, help="number of workers for data loading")

    def _add_network_config_(self, parser):
        """add hyperparameters for network architecture"""
        group1 = parser.add_argument_group('wifi')
        group1.add_argument('--input_channel', type=int, default=7, help='')
        group1.add_argument('--arch', type=str, default='resnet50', help='')
        group1.add_argument('--pretrained', type=bool, default=True, help='')

    def _add_training_config_(self, parser):
        """training configuration"""
        group = parser.add_argument_group('training')
        group.add_argument('--nr_epochs', type=int, default=500, help="total number of epochs to train")
        group.add_argument('--batch_size', type=int, default=40, help="batch size")
        group.add_argument('--lr', type=float, default=5e-4, help="initial learning rate")
        group.add_argument('--lr_decay', type=int, default=0.999, help="factor for exponential learning rate decay")
        group.add_argument('--lr_step_size', type=int, default=300, help="step size for step learning rate decay")
        group.add_argument('--continue', dest='cont', action='store_true', help="continue training from checkpoint")
        group.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
        group.add_argument('--vis', action='store_true', default=False, help="visualize output in tensorboard")
        group.add_argument('--save_frequency', type=int, default=100, help="save models every x epochs")
        group.add_argument('--val_frequency', type=int, default=10, help="run validation every x iterations")
        group.add_argument('--vis_frequency', type=int, default=1000, help="visualize output every x iterations")

        # seq2seq training
        group.add_argument('--teacher_decay', type=float, default=0.999, help="decay factor for teacher forcing ratio")
        group.add_argument('--stop_weight', type=float, default=0.01, help="weight factor for stop(bce) loss")

    def _add_testing_config_(self, parser):

        group = parser.add_argument_group('testing')
        #group.add_argument('--task', type=str, default='train', help="file path to generated fake shape codes")
