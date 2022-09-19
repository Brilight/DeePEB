import numpy as np
import torch 
import torch.nn as nn
import torch.utils.data as data

import os


def save(model, modelPATH, modelname, info=None):
    import time
    f = open(modelPATH+"save_log", "a")
    print("\n\nModel saved at "+time.ctime(),file=f)
    if info is not None:
        print("Critical information", info,file=f)
    print("Model Parameters:", file=f)
    print(model.parameters, '\n\n', file=f)
    if isinstance(model, nn.DataParallel):
        torch.save(model.module, modelPATH+modelname)
    else:
        torch.save(model, modelPATH+modelname) #.state_dict()
    print("!Model saved to ", modelPATH+modelname)
    f.close()
    
    
def load(model, PATH):
    ckpt = torch.load(PATH, map_location='cpu').state_dict()
    model.load_state_dict(ckpt)
    print("!Model load from ", PATH)
    return model