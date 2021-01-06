import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
import pandas as pd
from skimage import io, transform

from torch.utils.data import Dataset
from PIL import Image

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
#from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import importlib
import os


# Returns model
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    hidden_layer = 256

    return model

# Returns transform for training data
def get_transform(train):
  transforms = []
  transforms.append(T.ToTensor())

  if train:
    transforms.append(T.RandomHorizontalFlip(0.5))

  return T.Compose(transforms)



#def main(): 
  
# Train classifier
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

dataset = OysterDetectionDataset("/content/drive/MyDrive/BayChallenge/Images/Training/",
                          "/content/drive/MyDrive/BayChallenge/Images/Training/DetectionLabels.csv",
                          get_transform(train = True))
  
dataset_test = OysterDetectionDataset("/content/drive/MyDrive/BayChallenge/Images/Training/",
                          "/content/drive/MyDrive/BayChallenge/Images/Training/DetectionLabels.csv",
                                 get_transform(train = False))

indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:100])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])

data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle = False, num_workers=4,
                                            collate_fn = utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle = False, num_workers=4,
                                                  collate_fn=utils.collate_fn)
  
  
# alive, dead, shell, background
num_classes = 4
model = get_model_instance_segmentation(num_classes)

# uncomment after first training
PATH = '/content/drive/MyDrive/BayChallenge/detection_net2.pth'
model.load_state_dict(torch.load(PATH))
  
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.8, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 1

for epoch in range(num_epochs):
  print(epoch)
        
  train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

  lr_scheduler.step()

  evaluate(model, data_loader_test, device=device)

  print("That's it!")

PATH = '/content/drive/MyDrive/BayChallenge/detection_net.pth'
torch.save(model.state_dict(), PATH)
