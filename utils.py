__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '2.2'
__status__ = "Research"
__date__ = "28/1/2018"
__license__= "MIT License"

import os
import numpy as np
import glob

import subprocess
import platform
import sys
import pkg_resources
import torch
import PIL as Image

try:
    import cv2
except:
    print("WARNING: Could not load OpenCV python package. Some functionality may not be available.")


def list_files(path, extensions=[], sort=True, max_len=-1):
    if os.path.isdir(path):
        filenames = [os.path.join(path, fn) for fn in os.listdir(path) if
                           any([fn.endswith(ext) for ext in extensions])]
    else:
        print("ERROR. ", path,' is not a directory!')
        return []

    if sort:
        filenames.sort()

    if max_len>-1:
        filenames = filenames[:max_len]

    return filenames


def get_video_list(video_path, max_len=-1):
    return list_files(video_path, extensions=['avi', 'flv', 'mpg', 'mp4'], sort=True, max_len=max_len)

def get_image_list(video_path, max_len=-1):
    return list_files(video_path, extensions=['jpg', 'jpeg', 'png'], sort=True, max_len=max_len)


def get_split_files(dataset_path, splits_path, split_name, absolute_path=False):
    path = os.path.join(dataset_path, splits_path, split_name)
    files = glob.glob(path)
    files.sort()

    if not absolute_path:
        files_out = []
        for file in files:
            _,filename = os.path.split(file)
            files_out.append(filename)
        return files_out

    return files


def get_max_rc_weights(experiment_path):

    log_filename = 'train_log_0.csv'
    try:
        f = open(os.path.join(experiment_path, log_filename), 'rt')
        max_rc = 0
        max_epoch = -1
        max_mse = -1
        for line in f:
            toks = line.split(',')
            if toks[0] == 'val':
                epoch = toks[1]
                try:
                    rc = float(toks[4])
                    if rc > max_rc:
                        max_rc = rc
                        max_epoch = int(epoch)
                        max_mse = float(toks[6])
                except:
                    pass
        f.close()

        chkpt_file = experiment_path + '/' + 'weights_' + str(max_epoch) + '.pkl'
        if not os.path.isfile(chkpt_file):
            print("WARNING: File ",chkpt_file," does not exists!")
            return '', 0, 0, 0

        return chkpt_file, max_rc, max_mse, max_epoch

    except:
        print('WARNING: Could not open  ' + os.path.join(experiment_path, log_filename))

    return '', 0, 0, 0


def get_split_index(split_filename):
    filename, _ = os.path.splitext(split_filename)
    id = int(filename.split('_')[-1])
    return id


def get_weight_files(split_files, experiment_name, max_rc_checkpoints=True):
    data_dir = 'data'
    weight_files = []
    for split_filename in split_files:
        split_name,_ = os.path.splitext(split_filename)

        _, split_id = split_name.split('_')

        weight_files_all = os.path.join(data_dir, experiment_name+'_train_'+split_id+'/*.pkl')
        files = glob.glob(weight_files_all)
        if len(files) == 0:
            # No trained model weights for this split
            weight_files.append('')
            continue
        elif len(files) == 1:
            weight_files.append(files[0])
        else:
            # Multiple weights
            if max_rc_checkpoints:
                weights_dir = os.path.join(data_dir, experiment_name + '_train_' + split_id)
                print("Selecting model weights with the highest RC on validation set in ",weights_dir)
                weight_file, max_rc, max_mse, max_epoch= get_max_rc_weights(weights_dir)

                if weight_file != '':
                    print('Found: ',weight_file, '  RC=', max_rc, '   MSE=', max_rc, '  epoch=', max_epoch)
                    weight_files.append(weight_file)
                    continue

            # Get the weights from the last training epoch
            files.sort(key=lambda x: get_split_index(x), reverse=True)
            weight_file=files[0]
            weight_files.append(weight_file)


    return weight_files


def run_command(command):
    p = subprocess.Popen(command.split(),
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    return '\n'.join([ '\t'+line.decode("utf-8").strip() for line in p.stdout.readlines()])

def ge_pkg_versions():

    dep_versions = {}
    cmd = 'cat /proc/driver/nvidia/version'
    display_driver = run_command(cmd)
    dep_versions['display'] = display_driver

    dep_versions['cuda'] = 'NA'
    cuda_home = '/usr/local/cuda/'
    if 'CUDA_HOME' in os.environ:
        cuda_home = os.environ['CUDA_HOME']

    cmd = cuda_home+'/version.txt'
    if os.path.isfile(cmd):
        cuda_version = run_command('cat '+cmd)

    dep_versions['cuda'] = cuda_version
    dep_versions['cudnn'] = torch.backends.cudnn.version()

    dep_versions['platform'] = platform.platform()
    dep_versions['python'] = sys.version_info[0]
    dep_versions['torch'] = torch.__version__
    dep_versions['numpy'] = np.__version__
    dep_versions['PIL'] = Image.__version__

    dep_versions['OpenCV'] = 'NA'
    if 'cv2' in sys.modules:
        dep_versions['OpenCV'] = cv2.__version__

    dep_versions['torchvision'] = pkg_resources.get_distribution("torchvision").version

    return dep_versions


def print_pkg_versions():
    print("Packages & system versions:")
    print("----------------------------------------------------------------------")
    versions = ge_pkg_versions()
    for key, val in versions.items():
        print(key,": ",val)
    print("")
    return


if __name__ == "__main__":
    print_pkg_versions()

    split_files = get_split_files('datasets/lamem', 'splits', 'test_*.txt')
    print(split_files)

    weight_files = get_weight_files(split_files, experiment_name='lamem_ResNet50FC_lstm3_last', max_rc_checkpoints=True)
    # weight_files = get_weight_files(split_files, experiment_name='lamem_ResNet50FC_lstm3')
    print(weight_files)