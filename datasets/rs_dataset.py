import os
import numpy as np
import json
import logging
import torch
import torchvision.transforms as transform

from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger("global")

class RSDatasets(Dataset):
    def __init__(self, name, root, data_file, transform):
        super(RSDatasets, self).__init__()

        self.root = root
        logger.info("loading " + name)
        with open(data_file, 'r') as f:
            meta_data = json.load(f)

        self.images = []
        self.images.extend(meta_data['data'].keys())
        self.labels = meta_data['data']
        np.random.shuffle(self.images)
        self.num = len(self.images)
        self.num_class = len(meta_data['label_mapping'])

        self.transform = transform

    def __getitem__(self, index):
        image_name = self.images[index]
        label = self.labels[image_name] - 1
        image_path = os.path.join(self.root, image_name)
        image = Image.open(image_path)
        image = self.transform(image)

        return image, label

    def __len__(self):
        return self.num


class RSTestDataset(Dataset):
    def __init__(self, root, file, transform):
        super(RSTestDataset, self).__init__()
        self.root = root
        logger.info("loading test dataset" + file)
        with open(file, 'r') as f:
            test_data = json.load(f)

        self.images = []
        self.images.extend(test_data.keys())
        self.labels = test_data
        self.num = len(self.images)
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.images[index]
        label = self.labels[image_name] - 1
        image_path = os.path.join(self.root, image_name)
        image = Image.open(image_path)
        image = self.transform(image)

        return image, label


    def __len__(self):
        return self.num