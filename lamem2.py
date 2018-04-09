__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '1.1'
__status__ = "Research"
__date__ = "2/1/2018"
__license__= "MIT License"

import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np

import operator
from scipy import stats


img_extensions = ['.jpg', '.png', '.jpeg']
class LaMem2(data.Dataset):

    def __init__(self, img_root='', split_root='', split='train_1', transform=None, target_transform=None):
        self.img_root = os.path.expanduser(img_root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.image_cache = None
        self.valid_labels = False

        self.split_file = os.path.join(split_root, split)
        if not os.path.isdir(self.split_file):
            fname, ext = os.path.splitext(self.split_file)
            if ext == '':
                self.split_file += '.txt'

        self.data = []
        self.labels = []

        if os.path.isdir(self.split_file):
            # Load imaegs from given directory
            image_names = sorted(os.listdir(self.split_file))
            # print(image_names)
            print("Loaded",len(image_names),'images from directory',self.split_file)

            images = []
            for img_name in image_names:
                full_img_path = os.path.join(self.split_file, img_name)
                if not os.path.isfile(full_img_path):
                    continue

                gt_label = 0
                file, ext = os.path.splitext(img_name)
                if ext.lower() in img_extensions:
                    self.data.append(full_img_path)

                    # Set default ground truth memorabiltiy. This Could be extracted form the image filename
                    self.labels.append(float(gt_label))



        else:
            # Load images according to a split file
            with open(self.split_file, 'rt') as f:
                for line in f:
                    parts = line.strip().split(' ')
                    img_filename = parts[0].strip()
                    full_img_path = os.path.join(self.img_root, img_filename)
                    if os.path.isfile(full_img_path):
                        self.data.append(full_img_path)
                        self.labels.append(float(parts[1].strip()))
                        self.valid_labels = True
                    else:
                        print ("WARNING image ", full_img_path," doesn't exist")

        return

    # Loads image from file and returns BGR
    def img_loader(self, path, RGB=False):

        if self.image_cache is not None:
            img = self.image_cache.get_image(path)
            if img is not None:
                return img

        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img_out = img.convert('RGB')
                if self.image_cache is not None:
                    self.image_cache.cache_image(path, img_out)

                return img_out


    def preload_images(self):
        # Preload images
        if self.image_cache is not None:
            for path in self.data:
                self.img_loader(path)


    def __getitem__(self, index):
        sample = self.img_loader(self.data[index])
        target = self.labels[index]

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, self.data[index]


    def __len__(self):
        return len(self.data)


    def getMSE(self, d1, d2):
        mse = 0.0
        for a,b in zip(d1, d2):
            mse += (a-b)**2
        return mse / len(d1)


    def getRankCorrelationWithMSE(self, predicted, gt=None):

        if gt is None:
            gt = self.labels.copy()

        gt = np.array(gt).tolist()
        predicted = np.array(predicted).squeeze().tolist()

        n = min(len(predicted), len(gt))
        if n < 2:
            return 0

        gt = gt[:n]
        predicted = predicted[:n]
        mse = self.getMSE(gt, predicted)

        def get_rank(list_a):
            # get GT rank
            rank_list = np.zeros(len(list_a))
            idxs = np.array(list_a).argsort()

            # Record the GT rank
            for rank, i in enumerate(idxs):
                rank_list[i] = rank

            return rank_list

        gt_rank = get_rank(gt)
        predicted_rank = get_rank(predicted)

        #-------------------------------------------------------
        ssd = 0
        for i in range(len(predicted_rank)):
            ssd += (gt_rank[i] -  predicted_rank[i])**2

        rc = 1-(6*ssd/(n*n*n - n))

        # spearmanr() from scipy package produces the same result
        # rc, _ = stats.spearmanr(a=predicted, b=gt, axis=0)

        return rc, mse

