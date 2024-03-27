import torch.nn as nn

def get_loss(loss_name: str):
    
    if loss_name == 'L1Loss':
        return nn.L1Loss()
    
    else:
        raise NotImplementedError
    