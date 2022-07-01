import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split

split = .2
np.random.seed(0)
batch_size = {'train':512, 'val': 1024, 'test': 2048}

augment = transforms.Compose(
    [
        transforms.Resize((160)),
        transforms.RandomCrop(150),
        transforms.ColorJitter(brightness=0.3, contrast=3, hue=0.5, saturation=0.5),
    ]
)

class Mydataset(Dataset):
    def __init__(self, data_type):
        self.images = np.load('train_images.npy')
        # self.labels = np.load('Labels.npy')
        self.labels = (np.load('train_labels.npy')).reshape(-1)
        self.data_type = data_type

        # Normalize
        # self.images = self.images / 255
        self.x, self.images_test, self.y, self.labels_test = train_test_split(self.images/255, self.labels, test_size=split,
                                                                              shuffle=True
                                                                              , random_state=0)
        self.images_train, self.images_val, self.labels_train, self.labels_val = train_test_split(self.x, self.y,
                                                                                                  test_size=0.2,
                                                                                                  shuffle=True,
                                                                                                  random_state=1)


    def __getitem__(self, index):
        if self.data_type == 'train':
            return torch.tensor(self.images_train[index], dtype=torch.float32), torch.tensor(self.labels_train[index],
                                                                                             dtype=torch.int64)
        if self.data_type == 'val':
            return torch.tensor(self.images_val[index], dtype=torch.float32), torch.tensor(self.labels_val[index],
                                                                                           dtype=torch.int64)
        if self.data_type == 'test':
            return torch.tensor(self.images_test[index], dtype=torch.float32), torch.tensor(self.labels_test[index],
                                                                                            dtype=torch.int64)

    def __len__(self):
        if self.data_type == 'train':
            return len(self.labels_train)
        if self.data_type == 'val':
            return len(self.labels_val)
        if self.data_type == 'test':
            return len(self.labels_test)


# data augment
print('preparing data')
train_set = Mydataset('train')
val_set = Mydataset('val')
test_set = Mydataset('test')
print('finish')

# load data
dataloaders_dict = {'train': torch.utils.data.DataLoader(train_set, batch_size=batch_size['train'],
                                                         shuffle=False, num_workers=0, drop_last=False),
                    'val': torch.utils.data.DataLoader(val_set, batch_size=batch_size['val'],
                                                       shuffle=False, num_workers=0, drop_last=False),
                    'test': torch.utils.data.DataLoader(test_set, batch_size=batch_size['test'],
                                                        shuffle=True, num_workers=0)}
if __name__ == '__main__':
    # check number
    n = 0
    for train, label in dataloaders_dict['train']:
        n += len(label)
    print('the number in train set:{}'.format(n))
