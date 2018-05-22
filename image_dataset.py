__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '1.1'
__status__ = "Research"
__date__ = "20/4/2018"
__license__= "MIT License"

import torch.utils.data as data
from PIL import Image
import numpy as np



class ImageDataset(data.Dataset):

    def __init__(self, images, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.valid_labels = False

        self.data = images
        n = len(images)
        self.labels = np.zeros((n,), dtype=np.float32)

        return


    def __getitem__(self, index):
        sample = Image.fromarray(self.data[index])
        target = self.labels[index]

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, str(index)


    def __len__(self):
        return len(self.data)
