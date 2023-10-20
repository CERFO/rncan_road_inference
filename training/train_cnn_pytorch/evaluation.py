import torch
import torch.nn.functional as F

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc

def dice_coef(SR, GT, const=1e-7):
    # flatten label and prediction tensors
    SR = SR.view(-1)
    GT = GT.view(-1)
    
    # get true pos (TP), false neg (FN), false pos (FP).
    true_pos = torch.sum(GT*SR)
    false_neg = torch.sum(GT*(1-SR))
    false_pos = torch.sum((1-GT)*SR)
    
    # get dice coef
    coef_val = (2 * true_pos) / (2 * true_pos + false_pos + false_neg + const)
    
    return coef_val

def dice_loss(SR, GT):
    return 1 - dice_coef(SR,GT)

def tversky_coef(SR, GT, alpha=0.5, const=1e-7):
    # flatten label and prediction tensors
    inputs = SR.view(-1)
    targets = GT.view(-1)
    
    # get true pos (TP), false neg (FN), false pos (FP).
    true_pos = torch.sum(GT*SR)
    false_neg = torch.sum(GT*(1-SR))
    false_pos = torch.sum((1-GT)*SR)
    
    # TP/(TP + a*FN + b*FP); a+b = 1
    coef_val = (true_pos) / (true_pos + alpha*false_neg + (1-alpha)*false_pos + const)
    
    return coef_val

def focal_tversky_loss(SR,GT, alpha=0.5, gamma=4/3):
    tversky_loss = 1 - tversky_coef(SR,GT, alpha=alpha)
    focal_tversky_loss = tversky_loss ** 1/gamma
    return tversky_loss

