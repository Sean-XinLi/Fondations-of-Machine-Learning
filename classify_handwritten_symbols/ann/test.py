import os

from torch.utils.data import Dataset
from data_prepare import *
from Function.Network import *
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from Function.Get_Result import *
import matplotlib.pyplot as plt


class test_dataset(Dataset):
    def __init__(self):
        self.images = np.load('train_images.npy') / 255
        # self.labels = np.load('Labels.npy')
        self.labels = (np.load('train_labels.npy')).reshape(-1)

    def __getitem__(self, index):
        return torch.tensor(self.images[index], dtype=torch.float32), torch.tensor(self.labels[index],
                                                                                   dtype=torch.int64)

    def __len__(self):
        return len(self.labels)


def test_data(directory, dataloader_dict):
    path = os.path.join(directory, 'Results/result/Model.pt')
    save_path = os.path.join(directory, 'Results/test_results')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = torch.load(path)
    running_corrects = 0
    model = model.cpu()
    GT = np.array(0)
    Prediction = np.array(0)
    model.eval()
    n = 0
    with torch.no_grad():
        for test, label in dataloader_dict['test']:
            test = torch.unsqueeze(test, dim=1)

            outputs = model(test)
            _, preds = torch.max(outputs, 1)

            GT = np.concatenate((GT, label.detach().cpu().numpy()), axis=None)

            Prediction = np.concatenate((Prediction, preds.detach().cpu().numpy()), axis=None)
            running_corrects += torch.sum(preds == label.data)

            n += label.shape[0]

        test_acc = running_corrects.double() / n
        print("Accuracy:{:4f}".format(test_acc))
    np.save(os.path.join(save_path, 'truth_label'), GT[1:])
    np.save(os.path.join(save_path, 'prediction_label'), Prediction[1:])

    confusion = confusion_matrix(GT[1:], Prediction[1:])

    classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
               '17', '18', '19', '20', '21', '22', '23', '24', '25']
    title = 'Test Confusion Matrix'

    plot_confusion_matrix(confusion, classes, title=title)
    plt.savefig(os.path.join(save_path, 'cm_acc_' + str(test_acc.item()) + '.png'))
    plt.show()

    return GT[1:], Prediction[1:]


if __name__ == '__main__':
    test_set = test_dataset()
    dataloaders_dict = {'test': torch.utils.data.DataLoader(test_set, batch_size=batch_size['test'],
                                                            shuffle=True, num_workers=0)}
    directory = os.getcwd()
    GT, Prediction = test_data(directory, dataloaders_dict)
