import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from workspace_utils import active_session
from PIL import Image
import json
import numpy as np

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

evaluation_transforms = transforms.Compose([transforms.Resize(225),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def get_trainloader(size_batch=64):
    train_set = datasets.ImageFolder(train_dir, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=size_batch, shuffle=True)
    
    return trainloader, train_set

def get_valloader(size_batch=64):
    
    val_set = datasets.ImageFolder(valid_dir, transform=evaluation_transforms)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=size_batch, shuffle=True)
   
    return valloader, val_set

def get_testloader(size_batch=64):
        
    test_set = datasets.ImageFolder(test_dir, transform=evaluation_transforms)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=size_batch, shuffle=True)
    return testloader, test_set


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    
    width, height = pil_image.size

    if width < height: 
        resize_size = [256, 256**600]
    else: 
        resize_size = [256**600, 256]
        
    pil_image.thumbnail(size=resize_size)

    # Crop 
    center = width/4, height/4
    
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1] +(244/2)
    pil_image = pil_image.crop((left, top, right, bottom))

    # Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1
    np_img = np.array(pil_image) / 255

    # Normalize color channel
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    np_img = (np_img - means) / stds
        
    # Set the color channel to the first dim
    np_img = np_img.transpose(2, 0, 1)
    
    return np_img


def load_checkpoint(filepath='checkpoint.pth'):
    checkpoint = torch.load(filepath)
    model_name = checkpoint['architecture']
    model = getattr(models, model_name)(pretrained=True)
    
    for param in model.parameters(): 
        param.requires_grad = False
    
    if hasattr(model, 'fc'):
        model.fc = checkpoint['classifier']
    else:
        model.classifier = checkpoint['classifier']
        
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


# Don't use, unaccuracte.
def get_accuracy(model, inputes, labels, criterion):
    log_ps = model.forward(inputes)
    test_loss = criterion(log_ps, labels)     
    # calculate accuracy
    ps = torch.exp(log_ps)
    _, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor))
    
    return accuracy, test_loss




with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
    
def predict(checkpintpath, image_path, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model = load_checkpoint(checkpintpath)
    model.to(device)
    
    # Set model to evaluate
    model.eval();
   

    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), 
                                                  axis=0)).type(torch.FloatTensor).to(device)

    # Find probabilities (results) by passing through the function (note the log softmax means that its on a log scale)

    log_ps = model.forward(torch_image)

    # Get Probabilities.
    ps = torch.exp(log_ps)

    # Find top classes for value of K's
    top_probs, top_labels = ps.topk(topk, dim=1)
    
    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0] 
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers

def save_checkpoint(model, save_dir, dataset):
    model.class_to_idx = dataset.class_to_idx
    checkpoint = {'architecture': model.name,
                'classifier': model.fc if hasattr(model, 'fc') else model.classifier, 
                'class_to_idx': model.class_to_idx,
                'state_dict': model.state_dict()}
    torch.save(checkpoint, save_dir)


    
    