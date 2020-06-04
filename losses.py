import torch

###############################################
######## BinaryCrossEntropy Loss ##############
###############################################

def BinaryCrossEntropyLoss(preds, targets):
    bce = torch.mean(preds - preds*targets + torch.log(1. + torch.exp(-preds)))
    
    return bce

###############################################
############## Focal Loss #####################
###############################################

def FocalLoss(preds, targets, eps=1e-8, gamma=2):
    preds = torch.sigmoid(preds)
    
    focal = -torch.mean(torch.pow(1-preds, gamma)*targets*torch.log(preds+eps)+\
                        (1-targets)*torch.log(1-preds+eps)) 
    
    return focal

###############################################
############## Dice Loss ######################
###############################################

def DiceLoss(preds, targets, smooth=1):
    preds = torch.sigmoid(preds) 

    intersection = (preds * targets).sum() + smooth
    union = preds.sum() + targets.sum() + smooth
    dice = torch.div(2*intersection, union)  

    return 1. - dice

###############################################
############## Tversky Loss ###################
###############################################

def TverskyLoss(preds, targets, alpha=0.5, beta=0.5, smooth=1):
    preds = torch.sigmoid(preds, dim=1)
    
    intersection = (preds * targets).sum() + smooth
    fps = (preds * (1 - targets)).sum()
    fns = ((1 - preds) * targets).sum()
    denominator = intersection + alpha * fps + beta * fns + smooth
    tversky = torch.div(intersection, denominator)
    
    return 1. - tversky