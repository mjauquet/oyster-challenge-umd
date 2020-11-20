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



# Network
class Net(nn.Module):

    def __init__(self):
      super().__init__()

      # Creation of convolutional layers.
      # Define filters quantity, size, and type.
      self.conv1 = nn.Conv2d(3, 6, 5)
      self.pool = nn.MaxPool2d(2, 2) # Reduces the size of the filter image
      self.conv2 = nn.Conv2d(6, 16, 5)

      # Creation of Linear layer.
      # 
      self.fc1 = nn.Linear(16 * 5 * 5, 120)
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, 10)
      self.fc4 = nn.Linear(10, 2)

    # x is our data (image)
    # Each line is passing our data through our defined layers above
    def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))      # Passing x through first layer
      x = self.pool(F.relu(self.conv2(x)))
      x = x.view(-1, 16 * 5 * 5)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      
      return x

# Dataset
class OysterDataset(Dataset):
    ''' Dataset for detection of oyster images '''

    def __init__(self, root_dir, csv_file, transforms):
        '''
        Initialize variable for data set
        root_dir (string) is the directory with all the images
        csv_file (string) is the path to the csv file with annotations
        transform (callable, optional) optional transform to be applied on sample
        '''

        self.oysters = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transforms = transforms


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # Get path of image which is row[index], col[0]
        img_name = os.path.join(self.root_dir, self.oysters.iloc[index, 0])

        # following PedFundanDataset
        img = Image.open(img_name).convert("RGB")

        # Getting Labels
        num_objs = self.oysters.iloc[index, 4]
  #      print("There are ", num_objs, "objects in ", img_name)
        
        labels = []
        boxes = []

        for i in range(num_objs):
            pos = 8 + i * 5     # get column for object bounding boxes. boxes start at column 8
            xmin  = self.oysters.iloc[index, pos + 1]
            ymin = self.oysters.iloc[index, pos + 2]
            xmax = self.oysters.iloc[index, pos + 3]
            ymax = self.oysters.iloc[index, pos + 4]
            boxes.append([xmin, ymin, xmax, ymax])

            obj_class = self.oysters.iloc[index, pos]   # get class of object 0,1, or 2
            labels.append(1)

            #print("Box: ", boxes[i], "\nClass: ", labels[i])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        #print(boxes)
        
        labels = torch.as_tensor(labels, dtype=torch.int64)
        #print(labels)                

        img_id = torch.tensor([index])    # name of image
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # area of image

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.oysters)


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    #in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    #model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer,
                                                    #    num_classes)

    return model

def get_transform(train):
  transforms = []
  transforms.append(T.ToTensor())

  if train:
    transforms.append(T.RandomHorizontalFlip(0.5))

  return T.Compose(transforms)



def main(): 



    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)

    num_classes = 2

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    backbone = torchvision.models.mobilenet_v2(pretrained = True).features

    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(sizes = ((32,64, 128, 256, 512), ),
                                      aspect_ratios = ((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names = [0], output_size = 7,
                                                  sampling_ratio= 2)

    model = FasterRCNN(backbone, num_classes = 2, rpn_anchor_generator = anchor_generator,
                     box_roi_pool = roi_pooler)

  
  
  # Train classifier
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    dataset = OysterDataset("/Users/darwin/Downloads/Bay project/Training",
                              "/Users/darwin/Downloads/Bay project/Training/TrainingDetectionLabels2.csv",
                          get_transform(train = True))

    dataset_test = OysterDataset("/Users/darwin/Downloads/Bay project/Training",
                              "/Users/darwin/Downloads/Bay project/Training/TrainingDetectionLabels2.csv",
                             get_transform(train = False))

    indices = torch.randperm(len(dataset)).tolist()
    
    dataset = torch.utils.data.Subset(dataset, indices[:100])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle = False, num_workers=4,
                                            collate_fn = utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=2, shuffle = False, num_workers=4,
                                                  collate_fn = utils.collate_fn)

    model = get_model_instance_segmentation(num_classes)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.8, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

        lr_scheduler.step()

        evaluate(model, data_loader_test, device=device)
    

    print("That's it!")



if __name__ == "__main__":
  main()













