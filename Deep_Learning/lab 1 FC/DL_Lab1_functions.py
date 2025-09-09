import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, Subset
from sklearn.metrics import confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
import numpy as np
import random
import warnings
import matplotlib
import matplotlib.pyplot as plt
import torchvision
import os
import time
from typing import Callable, Optional, List, Any, Tuple, Dict
from collections import Counter

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
            
def display_Mnist_Sample(train: Dataset) -> None:
    from mpl_toolkits.axes_grid1 import ImageGrid
    samples = []
    targets = []
    for i in range(16):
        samples.append(train.__getitem__(i)[0])
        targets.append(train.__getitem__(i)[1])

    fig = plt.figure(figsize=(8, 8))
    grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.5)
    for ax, im, lb in zip(grid, samples, targets):
        ax.imshow(im, cmap='Greys_r')
        ax.set_title("Ground Truth: {}".format(lb))

def fashion_mnist_leave_only_clothes(train: torchvision.datasets.FashionMNIST, test: torchvision.datasets.FashionMNIST):
    # Remove undesired classes
    
    # These are the 3 classes we leave in the dataset:
    selected_classes = [0, 1, 2, 3, 4, 6] # Clothes only
    label_mapping = {selected_classes[i]: i for i in range(len(selected_classes))}

    def filter_classes(dataset):
        indices = []
        for idx, (_, label) in enumerate(dataset):
            if label in selected_classes:
                indices.append(idx)
        return Subset(dataset, indices)

    def remap_labels(subset):
        remapped_data = []
        for idx in range(len(subset)):
            sample, label = subset[idx]
            remapped_label = label_mapping[label]
            remapped_data.append((sample, remapped_label))
        return remapped_data    

    # Leave only the selected classes
    train_filtered = filter_classes(train)
    test_filtered = filter_classes(test)

    # Remap labels to be consistent and start from 0
    train_remapped = remap_labels(train_filtered)
    test_remapped = remap_labels(test_filtered) 

    return train_remapped, test_remapped
    
def fashion_mnist_imbalanced(train: torchvision.datasets.FashionMNIST, test: torchvision.datasets.FashionMNIST, seed: int): 
    
    def reduce_samples(dataset):
        class_2_indices = [idx for idx, (_, label) in enumerate(dataset) if label == 2]
        reduced_class_2_indices = random.sample(class_2_indices, len(class_2_indices) * 3 // 4) # remove 3/4 of class '2'
        class_3_indices = [idx for idx, (_, label) in enumerate(dataset) if label == 3]
        reduced_class_3_indices = random.sample(class_3_indices, len(class_3_indices) * 4 // 5) # remove 4/5 of class '3'
        reduced_indices = reduced_class_2_indices + reduced_class_3_indices
        
        remaining_indices = [idx for idx in range(len(dataset)) if idx not in reduced_indices]
        return Subset(dataset, remaining_indices)

    # Remove undesired classes
    train_filtered , test_filtered = fashion_mnist_leave_only_clothes(train, test)
    
    # Reduce the number of classes' samples in the train set
    train_reduced = reduce_samples(train_filtered)

    return train_reduced, test_filtered

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