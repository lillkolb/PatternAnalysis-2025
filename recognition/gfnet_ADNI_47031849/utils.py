"""
From Task Sheet:
"You may create other helper files such as “utils.py” to better organise your project"
"""
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, classification_report, RocCurveDisplay, PrecisionRecallDisplay
import numpy as np
from torchvision import transforms

MEAN = 0.11486841564676334
STD = 0.21826585544938487

# saving_filepath = './drive/MyDrive/Colab_Notebooks/Final_proj_stored'

# https://apxml.com/courses/pytorch-for-tensorflow-developers/chapter-3-pytorch-data-loading-for-tf-users/data-augmentation-pytorch-torchvision
# apply geometric transforms before colour
def get_transforms(train):
    if train:
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=(210, 210), scale=(0.95, 1.02), ratio=(0.95, 1.05)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.98, 1.02)),
            transforms.RandomApply([transforms.ElasticTransform(alpha=10.0, sigma=3.0)], p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.4, scale=(0.01, 0.10), ratio=(0.5, 2.0)), # has to be after tensor
            transforms.Normalize(mean=[MEAN], std=[STD])
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[MEAN], std=[STD])
        ])
    return data_transforms

def plot_data(saving_filepath):
    # load accuracy and loss data
    with open(os.path.join(saving_filepath, 'train_loss_data.pkl'), 'rb') as f:
        tld_load = pickle.load(f)

    with open(os.path.join(saving_filepath, 'train_accuracy_data.pkl'), 'rb') as f:
        tad_load = pickle.load(f)

    with open(os.path.join(saving_filepath, 'valid_loss_data.pkl'), 'rb') as f:
        vld_load = pickle.load(f)

    with open(os.path.join(saving_filepath, 'valid_accuracy_data.pkl'), 'rb') as f:
        vad_load = pickle.load(f)

    total_epochs = len(tld_load)
    ep = np.arange(0, total_epochs, 1)
    
    # plt.subplot(1, 2, 1)
    plt.plot(ep, tld_load, 'g', ep, vld_load, 'r', linewidth=2.0)
    plt.legend(['Train Loss', 'Valid Loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.grid()
    plt.show()

    # plt.subplot(1, 2, 2)
    plt.plot(ep, tad_load, 'g', ep, vad_load, 'r', linewidth=2.0)
    plt.legend(['Train Accuracy', 'Valid Accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.grid()
    plt.show()

def analyse_data(actual_values, predictions_list, saving_filepath):
    predictions_list = np.array(predictions_list)
    actual_values = np.array(actual_values)

    report = classification_report(actual_values, predictions_list, target_names=['AD', 'NC'])
    print(report)

    testing_accuracy = accuracy_score(actual_values, predictions_list)
    print("Testing accuracy: ", testing_accuracy)

    roc_auc = roc_auc_score(actual_values, predictions_list)
    print("ROC AUC: ", roc_auc)

    precision = precision_score(actual_values, predictions_list)
    print("Precision: ", precision)

    recall = recall_score(actual_values, predictions_list)
    print("Recall: ", recall)

    f1 = f1_score(actual_values, predictions_list)
    print("F1 score: ", f1)

    cm = confusion_matrix(actual_values, predictions_list)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['AD', 'NC'])
    disp.plot()
    plt.savefig(os.path.join(saving_filepath, 'confusion_matrix.png'))
    plt.show()

    roc_disp = RocCurveDisplay.from_predictions(actual_values, predictions_list)
    plt.title("ROC Curve")
    plt.savefig(os.path.join(saving_filepath, 'roc_curve.png'))
    plt.show()

    pr_disp = PrecisionRecallDisplay.from_predictions(actual_values, predictions_list)
    plt.title("Precision Recall")
    plt.savefig(os.path.join(saving_filepath, 'precision_recall_curve.png'))
    plt.show()

    plot_data()