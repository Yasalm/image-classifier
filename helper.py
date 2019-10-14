import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

# Two pre-trained model choices.
arch = {"resnet50":2048,
        "densenet121":1024
         }


def _load_model(model_name):
    model = getattr(models, model_name)(pretrained=True)
    model.name = model_name
    
    # Freezing parameters
    for param in model.parameters():
        param.requires_grad = False
        
    return model

def build_network(modelname='densenet121', hidden_layer1_units = 512 , dropout=0.25):
    arch = {"resnet50":2048,
            "densenet121":1024
            }
    
    if modelname == 'resnet50' or modelname == 'densenet121':
            n_input = arch[modelname]
    else:
        ("Please choose either resnet50 or densenet121")
            
    n_inputs = arch[modelname]
    
    classifier = nn.Sequential(nn.Linear(n_inputs, hidden_layer1_units),
                          nn.ReLU(),
                          nn.Dropout(dropout),
                          nn.Linear(hidden_layer1_units, 102),
                          nn.LogSoftmax(dim=1))
    
    model = _load_model(modelname)
    if hasattr(model, 'fc'):
        model.fc = classifier
    else:
        model.classifier = classifier
    
    return model