__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '9.5'
__status__ = "Research"
__date__ = "30/1/2018"
__license__= "MIT License"

import os
import numpy as np
import torch
from torch.autograd import Variable
import pkg_resources

class HParameters:

    def __init__(self):
        self.use_cuda = True
        self.cuda_device = 0

        self.use_attention = True
        self.last_step_prediction = False
        self.train_split = 'train_1'
        self.val_split = 'val_1'
        self.front_end_cnn = 'ResNet18FC'

        self.epoch_start = 0
        self.epoch_max = 100

        self.l2_req = 0.00001
        self.mem_loc_w = None
        self.seq_steps = 3

        self.lr_epochs = [0]
        self.lr = [0.0001]

        # alpha map cost weight
        # hps.gamma = 0.001
        self.gamma = 0.00001

        # memorability-location cost weight
        self.omega = 0

        self.train_batch_size = 128
        self.test_batch_size = 128

        self.torch_version_major, self.torch_version_minor = [int(v) for v in torch.__version__.split('.')[:2]]
        torchvision_version = pkg_resources.get_distribution("torchvision").version
        self.torchvision_version_major, self.torchvision_version_minor = [int(v) for v in torchvision_version.split('.')[:2]]

        return

    @property
    def seq_steps(self):
        return self._seq_steps

    @seq_steps.setter
    def seq_steps(self, value):
        self._seq_steps = value
        if value <= 0:
            return

        mem_loc_w = (0.1 ** (np.arange(0, self._seq_steps)))
        self.mem_loc_w = Variable(torch.from_numpy(np.array([mem_loc_w]))).float()

    def __str__(self):
        vars = [attr for attr in dir(self) if not callable(getattr(self,attr)) and not (attr.startswith("__") or attr.startswith("_"))]

        info_str = ''
        for i, var in enumerate(vars):
            val = getattr(self, var)
            if isinstance(val, Variable):
                val = val.data.cpu().numpy().tolist()[0]
            info_str += '['+str(i)+'] '+var+': '+str(val)+'\n'

        return info_str


def get_amnet_config(args):

    hps = HParameters()

    hps.dataset_name = args.dataset
    hps.experiment_name = args.experiment
    hps.front_end_cnn = args.cnn
    hps.model_weights = args.model_weights
    hps.dataset_root = args.dataset_root
    hps.images_dir = args.images_dir
    hps.splits_dir = args.splits_dir
    hps.eval_images = args.eval_images
    hps.test_split = args.test_split
    hps.val_split = args.val_split
    hps.train_split = args.train_split
    hps.epoch_max = args.epoch_max
    hps.epoch_start = args.epoch_start
    hps.train_batch_size = args.train_batch_size
    hps.test_batch_size = args.test_batch_size

    # Default configuration
    hps.cuda_device = args.gpu
    hps.seq_steps = args.lstm_steps
    hps.last_step_prediction = args.last_step_prediction
    hps.use_attention = not args.att_off

    hps.use_cuda = hps.cuda_device > -1

    # Create experiment name
    if hps.experiment_name == '':
        hps.experiment_name = hps.dataset_name + '_' + hps.front_end_cnn
        hps.experiment_name += '_lstm' + str(hps.seq_steps)

        if hps.last_step_prediction:
            hps.experiment_name += '_last'

        if not hps.use_attention:
            hps.experiment_name += '_noatt'


#----------------------------------------------------------------------------------
# Dataset specific configurations

    if hps.dataset_name == 'lamem':

        if hps.front_end_cnn == '':
            hps.front_end_cnn = 'ResNet50FC'

        if hps.dataset_root == '':
            hps.dataset_root = 'datasets/lamem/'

        # Set default validation split filename
        if hps.val_split == '':
            if hps.train_split != '':
                hps.val_split = 'val_' + hps.train_split.split('_')[1]

        if hps.epoch_max < 0:
            hps.epoch_max = 55


        if hps.train_batch_size < 0:
            hps.train_batch_size = 222
            #hps.train_batch_size = 128

        if hps.test_batch_size < 0:
            hps.test_batch_size = 370

        hps.l2_req = 0.000001

        hps.target_mean = 0.754
        hps.target_scale = 2.0

        hps.img_mean = [0.485, 0.456, 0.406]
        hps.img_std = [0.229, 0.224, 0.225]



    elif hps.dataset_name == 'sun':
        # SUN memorability dataset

        if hps.val_split == '':
            if hps.train_split != '':
                hps.val_split = 'test_' + hps.train_split.split('_')[1]

        if hps.dataset_root == '':
            hps.dataset_root = 'datasets/SUN_memorability/'

        if hps.epoch_max < 0:
            hps.epoch_max = 50

        if hps.train_batch_size < 0:
            hps.train_batch_size = 222

        if hps.test_batch_size < 0:
            hps.test_batch_size = 370

        hps.l2_req = 0.0001

        # TODO: Should be updated for the SUN dataset!
        hps.target_mean = 0.754
        hps.target_scale = 2.0
        hps.img_mean = [0.485, 0.456, 0.406]
        hps.img_std = [0.229, 0.224, 0.225]



    elif hps.dataset_name == 'ava':
        # AVA image aesthetic dataset

        if hps.val_split == '':
            if hps.train_split != '':
                hps.val_split = 'test_' + hps.train_split.split('_')[1]

        if hps.dataset_root == '':
            hps.dataset_root = 'datasets/ava/'

        if hps.epoch_max < 0:
            hps.epoch_max = 150

        if hps.train_batch_size < 0:
            hps.train_batch_size = 370

        if hps.test_batch_size < 0:
            hps.test_batch_size = 370

        hps.l2_req = 0.000001
        hps.lr = [0.0001]

        hps.target_mean = 0.538388987454
        hps.target_scale = 2.0
        hps.img_mean = [0.485, 0.456, 0.406]
        hps.img_std = [0.229, 0.224, 0.225]

    else:
        print("ERROR Unknown dataset:", hps.dataset_name)

    if hps.front_end_cnn == 'VGG16FC':
        hps.train_batch_size = 138
        hps.test_batch_size = 138

    if hps.train_split != '':
        hps.experiment_name += '_' + hps.train_split

    return hps

