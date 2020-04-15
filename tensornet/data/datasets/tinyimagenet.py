import os
import csv
import random
import requests
import zipfile
import numpy as np
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset

from tensornet.data.datasets.dataset import BaseDataset


class TinyImageNet(BaseDataset):
    """Load Tiny ImageNet Dataset."""

    # TODO: Calculate mean and std for dataset
    
    @property
    def classes(self):
        """Return list of classes in the dataset."""
        return self.sample_data.classes
    
    def _download(self, train=True, apply_transform=True):
        if not self.path.endswith('tinyimagenet'):
            self.path = os.path.join(self.path, 'tinyimagenet')
        transform = None
        if apply_transform:
            transform = self.train_transform if train else self.val_transform
        return TinyImageNetDataset(
            self.path, train=train, transform=transform
        )
    
    @property
    def image_size(self):
        """Return shape of data i.e. image size."""
        return np.array(self.sample_data.data[0]).shape


class TinyImageNetDataset(Dataset):

    def __init__(self, path, train=True, train_split=0.7, download=True, random_seed=1, transform=None):
        super(TinyImageNetDataset, self).__init__()
        
        self.path = path
        self.train = train
        self.train_split = train_split
        self.transform = transform
        self._validate_params()

        # Download dataset
        if download:
            self.download()

        self._class_ids = self._get_class_map()
        self.data, self.targets = self._load_data()

        self._image_indices = np.arange(len(self.targets))

        np.random.seed(random_seed)
        np.random.shuffle(self._image_indices)

        split_idx = int(len(self._image_indices) * train_split)
        self._image_indices = self._image_indices[:split_idx] if train else self._image_indices[split_idx:]
    
    def __len__(self):
        return len(self._image_indices)
    
    def __getitem__(self, index):
        image_index = self._image_indices[index]
        
        image = self.data[image_index]
        if not self.transform is None:
            image = self.transform(image)
        
        return image, self.targets[image_index]
    
    def __repr__(self):
        head = 'Dataset TinyImageNet'
        body = ['Number of datapoints: {}'.format(self.__len__())]
        if self.path is not None:
            body.append('Root location: {}'.format(self.path))
        body += [f'Split: {"Train" if self.train else "Test"}']
        if hasattr(self, 'transforms') and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [' ' * 4 + line for line in body]
        return '\n'.join(lines)
    
    @property
    def classes(self):
        return tuple(x[1]['name'] for x in sorted(self._class_ids.items(), key=lambda y: y[1]['id']))
    
    def _validate_params(self):
        """Validate input parameters."""
        if self.train_split > 1:
            raise ValueError('train_split must be less than 1')
    
    def _get_class_map(self):
        """Create a mapping from class id to the class name."""
        with open(os.path.join(self.path, 'wnids.txt')) as f:
            class_ids = {x[:-1]: '' for x in f.readlines()}
        
        with open(os.path.join(self.path, 'words.txt')) as f:
            class_id = 0
            for line in csv.reader(f, delimiter='\t'):
                if line[0] in class_ids:
                    # class_ids[line[0]] = line[1].split(',')[0].lower()
                    class_ids[line[0]] = {
                        'name': line[1],
                        'id': class_id
                    }
                    class_id += 1
        
        return class_ids
    
    def _load_image(self, image_path):
        image = Image.open(image_path)

        # Convert grayscale image to RGB
        if image.mode == 'L':
            image = np.array(image)
            image = np.stack((image,) * 3, axis=-1)
            image = Image.fromarray(image.astype('uint8'), 'RGB')
        
        return image

    def _load_data(self):
        """Fetch data from each data directory and store them in a list."""
        data, targets = [], []

        # Fetch train dir images
        train_path = os.path.join(self.path, 'train')
        for class_dir in os.listdir(train_path):
            train_images_path = os.path.join(train_path, class_dir, 'images')
            for image in os.listdir(train_images_path):
                if image.lower().endswith('.jpeg'):
                    data.append(
                        self._load_image(os.path.join(train_images_path, image))
                    )
                    targets.append(self._class_ids[class_dir]['id'])
        
        # Fetch val dir images
        val_path = os.path.join(self.path, 'val')
        val_images_path = os.path.join(val_path, 'images')
        with open(os.path.join(val_path, 'val_annotations.txt')) as f:
            for line in csv.reader(f, delimiter='\t'):
                data.append(
                    self._load_image(os.path.join(val_images_path, line[0]))
                )
                targets.append(self._class_ids[line[1]]['id'])
        
        return data, targets
    
    def download(self):
        if not os.path.exists(self.path):
            print('Downloading dataset...')
            r = requests.get('http://cs231n.stanford.edu/tiny-imagenet-200.zip', stream=True)
            zip_ref = zipfile.ZipFile(BytesIO(r.content))
            zip_ref.extractall(os.path.dirname(self.path))
            zip_ref.close()

            # Move file to appropriate location
            os.rename(
                os.path.join(os.path.dirname(self.path), 'tiny-imagenet-200'),
                self.path
            )
            print('Done.')
        else:
            print('Files already downloaded.')
