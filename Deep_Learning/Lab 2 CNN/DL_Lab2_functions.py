import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, Subset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import random
import warnings
import matplotlib
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import os
import time
import textwrap
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Callable, Optional, List, Any, Tuple, Dict

class Hyper_Params:
    def __init__(self, params_dict: Dict = None):
        if params_dict:
            self.params_dict = params_dict
        else:
            self.params_dict = {'train_size': None, 'batch_size': None, 'epochs': None, 'lr': None,
                                'epoch_loss_train': [], 'epoch_accuracy_train': [], 'train_loss_record': [], 'train_accuracy_record': [], 'test_loss_record': [], 'test_accuracy_record': [],                
                                'fig': None, 'ax1': None, 'ax2': None}
            
    def __getattr__(self, attr: str) -> Any:
        if attr in self.params_dict:
            return self.params_dict[attr]
        else:
            raise AttributeError(f"'Params' object has no attribute '{attr}'")

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr == 'params_dict':
            super().__setattr__(attr, value)
        else:
            self.params_dict[attr] = value

    def __getitem__(self, key: str) -> Any:
        return self.params_dict[key]

def multi_class_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy

def print_performance(epoch: int, i: int, model_params: Hyper_Params,
                      test_loss: float, test_accuracy: float, remove_first_test_loss: int = 0) -> None:
    model_params.train_loss_record.append(np.sum(model_params.epoch_loss_train) / (i + 1))
    model_params.train_accuracy_record.append(np.sum(model_params.epoch_accuracy_train) / (i + 1))
    model_params.test_loss_record.append(test_loss)
    model_params.test_accuracy_record.append(test_accuracy)
    values = [epoch, i+1, round(model_params.train_loss_record[-1], 2), round(model_params.train_accuracy_record[-1], 2), round(test_loss, 2), round(test_accuracy, 2)]

    print_performance_grid(Values=values)
    update_graph(model_params.fig, model_params.ax1, model_params.ax2, np.arange(len(model_params.train_loss_record)),
                 model_params.train_loss_record, model_params.test_loss_record, model_params.train_accuracy_record, model_params.test_accuracy_record,
                 remove_first_test_loss=remove_first_test_loss)

def update_graph(fig: plt.figure, ax1: plt.axes, ax2: plt.axes, x: np.ndarray,
                 train_loss: List[float], test_loss: List[float], train_accuracy: List[float],
                 test_accuracy: List[float], remove_first_test_loss: int = 0) -> None:
    ax1.clear()
    ax1.set_title("Accuracy")
    ax1.plot(x, np.array(train_accuracy)*100, color="green", label='Train')
    ax1.plot(x, np.array(test_accuracy)*100, color="black", linestyle='dashed', label='Test', marker='o')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylim(top=100)
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(10))
    ax1.grid(color='gray', which='both', axis='y', linestyle='--', linewidth=0.5)
    x_values = range(0, int(x[-1])+1)
    for t in x_values:
        ax1.axvline(x=t, color='cyan', linestyle='--', linewidth=0.5)
    ax1.legend(loc='lower right', fontsize='x-large')

    ax2.clear()
    ax2.set_title("Loss")
    ax2.plot(x, train_loss, color="red", label='Train')
    if len(test_loss[remove_first_test_loss:]):
        ax2.plot(x[remove_first_test_loss:], test_loss[remove_first_test_loss:],
                 color="black", linestyle='dashed', label='Test', marker='o')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epochs')
    ax2.legend(loc='upper right', fontsize='x-large')

    for t in x_values:
        ax2.axvline(x=t, color='cyan', linestyle='--', linewidth=0.5)
    fig.tight_layout(pad=2.0)
    plt.draw()
    plt.pause(0.00000001)

def print_performance_grid(Flag: bool = False, Values: Optional[List[Any]] = None) -> None:
    if Flag:
        headers = ['Epoch No.', 'Iter No.', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy']
        print('| {:^9} | {:^8} | {:^10} | {:^14} | {:^9} | {:^13} |'.format(*headers))
        line_width = len('| {:^9} | {:^8} | {:^10} | {:^14} | {:^9} | {:^13} |'.format(*headers))
        print('-' * line_width)
    if Values:
        print('| {:^9} | {:^8} | {:^10} | {:^14} | {:^9} | {:^13} |'.format(*Values))
        line_width = len('| {:^9} | {:^8} | {:^10} | {:^14} | {:^9} | {:^13} |'.format(*Values))
        print('-' * line_width)

def present_confusion_matrix(model: nn.Module, test_loader: DataLoader, class_dict: Dict[int, str], device: torch.device = 'cpu', big: bool = False) -> None:
    y_true = []
    y_pred = []
    n_class = len(class_dict)
    with torch.no_grad():
        model.eval()
        for j, batch in enumerate(test_loader):
            data, target = batch

            outputs = model(data.to(device))
            _, predicted = torch.max(outputs, 1)

            y_pred.extend(list(predicted.cpu().numpy()))
            y_true.extend(list(target.cpu().numpy()))
    class_names = [class_dict[i] for i in range(len(class_dict))]

    conf_matrix = confusion_matrix(y_true, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = class_names)
    cm_display.plot()
    plt.title('Confusion Matrix')
    plt.show()

def print_desserts(train: Dataset, class_names: dict) -> None:
    dataloader = DataLoader(train, batch_size=16, shuffle=True, num_workers=4)
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    # fig.subplots_adjust(hspace=0.6)

    for i, ax in enumerate(axes.flat):
        # Display image
        image = images[i]
        ax.imshow(image)
        ax.axis('off')

        label_index = labels[i].item()
        label_name = class_names[label_index]
        ax.set_title(label_name, fontsize=10, color='blue')

    plt.tight_layout()
    plt.show()

def prepare_dataloaders(train: Dataset, test: Dataset, batch_size, num_workers=16) -> Tuple[DataLoader, DataLoader]:
    if isinstance(batch_size, list):
        train_loader = DataLoader(train, batch_size=batch_size[0], shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test, batch_size=batch_size[1], shuffle=True, num_workers=num_workers)
        return train_loader, test_loader

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    return train_loader, test_loader

def display_mislabeled(model: nn.Module, train_loader: DataLoader, mean: List[float],
                       std: List[float], device: torch.device) -> None:
    imagenet_classes = {}
    with open("imagenet_classes.txt") as f:
        for line in f:
            (key, val) = line.split(sep=": ")
            imagenet_classes[int(key)] = val.strip("',\n ")

    model.eval()

    batch = next(iter(train_loader))

    with torch.no_grad():
        data, _ = batch
        output = model(data.to(device))
        _, labels = torch.max(output, 1)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.6)
    for i, ax in enumerate(axes.flat):
        # Display image
        image = batch[0][i]
        image[0] = image[0]*std[0]+mean[0]
        image[1] = image[1]*std[1]+mean[1]
        image[2] = image[2]*std[2]+mean[2]
        image = image.permute(1, 2, 0)
        ax.imshow(image)
        ax.axis('off')

        label_index = labels[i].item()
        label_name = imagenet_classes[label_index]
        ax.set_title("\n".join(textwrap.wrap(label_name, 20)), color='blue', wrap=True)

    plt.tight_layout()
    plt.show()