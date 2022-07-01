import itertools
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def save_results(directory, model, best_model_wts, train_error_history,
                 val_error_history, train_acc_history, val_acc_history):
    # Create directory to save results
    save_dir = os.path.join(directory, 'result')

    # Create directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate learning curve and save
    fig1 = plt.figure()
    plt.plot(train_error_history)
    plt.plot(val_error_history)
    plt.suptitle('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
    fig1.savefig((save_dir + '/Learning Curve.png'), dpi=fig1.dpi)

    fig2 = plt.figure()
    plt.plot(train_acc_history)
    plt.plot(val_acc_history)
    plt.suptitle('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    fig2.savefig((save_dir + '/Accuracy Curve.png'), dpi=fig2.dpi)
    plt.close()

    # Save model and weights
    torch.save(model, (save_dir + '/Model.pt'))
    torch.save(best_model_wts, (save_dir + '/Best_Weights.pt'))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #        print("Normalized confusion matrix")
    #    else:
    #        print('Confusion matrix, without normalization')
    plt.figure(figsize=(15,12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes,rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.tight_layout()


def Classifier_results(directory,model, best_model_wts, train_error_history,
                       val_error_history, train_acc_history, val_acc_history, GT, predictions,
                       ):
    # Set class names
    class_names = np.array([str(i) for i in range(1,26)])

    # Generate learning curve and save
    save_results(directory, model, best_model_wts, train_error_history, val_error_history, train_acc_history,
                 val_acc_history)

    save_dir = os.path.join(directory, 'result')


    # visualize
    for phase in ['val','test']:
        # Create confusion matrix and compute accuracy
        test_cm = confusion_matrix(GT[phase], predictions[phase])
        test_acc = 100 * sum(np.diagonal(test_cm)) / sum(sum(test_cm))

        cm_title=phase+' confusion matrix'
        np.set_printoptions(precision=2)
        plot_confusion_matrix(test_cm, classes=class_names, title=cm_title)
        plt.savefig(os.path.join(save_dir,phase+' confusion matrix.png'))
        plt.close()

    # # Save model and weights
    # torch.save(model, (save_dir + '/Model.pt'))
    # torch.save(best_model_wts, (save_dir + '/Best_Weights.pt'))

    # Count the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
