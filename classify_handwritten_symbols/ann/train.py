from __future__ import print_function
from __future__ import division
import os
import torch
import torch.nn as nn
import numpy as np
import time
import copy
from Function.Network import model
from Function.Get_Result import Classifier_results
from data_prepare import dataloaders_dict


def train(model, dataloaders, optimizer, device='cpu', num_epochs=25):
    since = time.time()

    train_error_history = []
    val_error_history = []
    train_acc_history = []
    val_acc_history = []

    CE_loss = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    model = model.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            acc = 0
            n = 0
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            # Iterate over data.
            for idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = torch.unsqueeze(inputs, dim=1)
                inputs = inputs.to(device)
                # input_numpy = inputs.cpu().detach().numpy()
                labels = torch.squeeze(labels)
                labels = labels.to(device)
                n += len(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    # forward pass
                    outputs = model(inputs)
                    loss = CE_loss(outputs, labels)

                    # backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    index = torch.argmax(outputs, dim=1)
                    acc += (torch.argmax(outputs, dim=1) == labels).sum().item()

            epoch_loss = running_loss / len(dataloaders[phase].sampler)

            if phase == 'train':
                train_error_history.append(epoch_loss)
                train_acc_history.append(acc / n)

            # print()
            print('{} Loss: {:.6f}, acc:{:.6f}'.format(phase, epoch_loss, acc / n))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_epoch = epoch + 1
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                val_error_history.append(epoch_loss)
                val_acc_history.append(acc / n)
    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation loss: {:4f} at Epoch {:.0f}'.format(best_loss, best_epoch))

    # load best model weights
    # Fit model on hold out test set
    model.load_state_dict(best_model_wts)

    # Get predictions for val or test set
    GT = {}
    predictions = {}
    for phase2 in ['val', 'test']:
        GT[phase2], predictions[phase2] = predict(dataloaders[phase2], model, device, phase2)

    return (
        model, best_model_wts, train_error_history, val_error_history, train_acc_history, val_acc_history, best_loss,
        time_elapsed, GT, predictions)


def predict(dataloader, model, device, phase):
    # Initialize and accumalate ground truth and predictions
    GT = np.array(0)
    Predictions = np.array(0)
    running_corrects = 0
    model = model.to(device)
    model = nn.Sequential(model, nn.Softmax(dim=1))
    model.eval()
    # Iterate over data.
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            inputs = torch.unsqueeze(inputs, dim=1)
            labels = labels.to(device)

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            GT = np.concatenate((GT, labels.detach().cpu().numpy()), axis=None)

            Predictions = np.concatenate((Predictions, preds.detach().cpu().numpy()), axis=None)
            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / len(dataloader.sampler)
    print(phase + ' Accuracy: {:4f}'.format(test_acc))

    return GT[1:], Predictions[1:]


if __name__ == '__main__':
    lr = 0.1
    epoch = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    channel = [8, 16, 32, 64, 128, 256, 512]
    model = model(channel)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    (model, best_model_wts, train_error_history, val_error_history, train_acc_history, val_acc_history, best_loss,
     time_elapsed, GT, prediction) = train(model, dataloaders_dict, optimizer, device, epoch)

    folder = 'Results'
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, folder)
    # save_results(final_directory, model, best_model_wts, train_error_history, val_error_history, time_elapsed,
    #              best_loss)

    Classifier_results(final_directory, model, best_model_wts, train_error_history, val_error_history,
                       train_acc_history,
                       val_acc_history, GT, prediction)
