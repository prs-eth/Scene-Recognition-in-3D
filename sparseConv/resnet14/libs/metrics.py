import torch, numpy as np
import sys
from sklearn.metrics import confusion_matrix as skconf
EPSILON=sys.float_info.epsilon

def get_IoU(gt,pred,num_labels):
    assert pred.shape == gt.shape
    idxs=gt<num_labels
    confusion_matrix=np.bincount(pred[idxs]*num_labels+gt[idxs],
                                 minlength=num_labels**2).reshape((num_labels,num_labels)).astype(np.ulonglong)+EPSILON
    IoU=np.around(
        confusion_matrix.diagonal()/(confusion_matrix.sum(0)+confusion_matrix.sum(1)-confusion_matrix.diagonal()),decimals=4)
    return IoU,IoU.mean()

def get_Recall(gt,pred,num_labels):
    assert pred.shape == gt.shape
    idxs=gt<num_labels
    confusion_matrix=np.bincount(pred[idxs]*num_labels+gt[idxs],
                                 minlength=num_labels**2).reshape((num_labels,num_labels)).astype(np.ulonglong)+EPSILON
    recall=np.around(
        confusion_matrix.diagonal()/confusion_matrix.sum(0),decimals=4)
    return recall,recall.mean()

def get_Acc(gt,pred,num_labels):
    assert pred.shape == gt.shape
    idxs=gt<num_labels
    
    n_correct=(gt[idxs]==pred[idxs]).sum()
    n_samples=idxs.sum()
    acc=round(n_correct*1.0/n_samples,3)
    return acc

def get_confusion_matrix(gt,pred,num_labels):
    assert pred.shape == gt.shape
    idxs=gt<num_labels
    confusion_matrix=np.bincount(pred[idxs]*num_labels+gt[idxs],
                                 minlength=num_labels**2).reshape((num_labels,num_labels)).astype(np.ulonglong)
    return confusion_matrix


def get_acc_iou_cm(gt,pred,num_labels):
    assert pred.shape == gt.shape
    idxs=gt<num_labels
    
    n_correct=(gt[idxs]==pred[idxs]).sum()
    n_samples=idxs.sum()
    acc=round(n_correct*1.0/n_samples,3)
    
    confusion_matrix=np.bincount(pred[idxs]*num_labels+gt[idxs],
                                 minlength=num_labels**2).reshape((num_labels,num_labels)).astype(np.ulonglong)+EPSILON
    IoU=np.around(
        confusion_matrix.diagonal()/(confusion_matrix.sum(0)+confusion_matrix.sum(1)-confusion_matrix.diagonal()),decimals=4)
    
    mIoU=np.around(IoU.mean(),decimals=4)
    return acc,mIoU,confusion_matrix