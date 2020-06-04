import torch

#######################################################
#### Intersection Over Union (Jaccard Coefficient) ####
#######################################################

def IoUnion(preds, targets):
    
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    IoU = torch.div(intersection, union)
    
    return IoU

######################################################
############### Dice Coefficient #####################
######################################################

def Dice(preds, targets, smooth=1):
    
    intersection = (preds * targets).sum() + smooth
    union = preds.sum() + targets.sum() + smooth
    dice = torch.div(2*intersection, union)  

    return dice

######################################################
############## Tversky Coefficient ###################
######################################################

def Tversky(preds, targets, alpha=0.5, beta=0.5, smooth=1):
    
    intersection = (preds * targets).sum() + smooth
    fps = (preds * (1 - targets)).sum()
    fns = ((1 - preds) * targets).sum()
    denominator = intersection + alpha * fps + beta * fns + smooth
    tversky = torch.div(intersection, denominator)
    
    return tversky