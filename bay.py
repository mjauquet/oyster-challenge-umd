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

from torch.utils.data import Dataset
from PIL import Image

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

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
    ''' Dataset for crops of oyster images '''

    def __init__(self, root_dir, csv_file, transform=None):
        '''
        Initialize variable for data set
        root_dir (string) is the directory with all the images
        csv_file (string) is the path to the csv file with annotations
        transform (callable, optional) optional transform to be applied on sample
        '''

        self.oysters = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.oysters)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # Get path of image which is row[index], col[0]
        img_name = os.path.join(self.root_dir, self.oysters.iloc[index, 0])
        
        # Following Face landmark dataset
        # Load image
        #image = io.imread(img_name)

        # following PedFundanDataset
        img = Image.open(img_name).convert("RGB")
        
        img = np.array(img)

        if self.transform is not None:
            img = self.transform(img)

        # Get class for image row[index], col[4]
        target = self.oysters.iloc[index, 4]
        
        return img, target


# main
def run_file(pathName):
  pathName += '/'

  images = []

  for image in os.listdir(pathName):
    if(".jpg" in image): 
      img_loc = plt.imread(pathName + image)

      img_loc = np.array(img_loc)

      img_loc = img_loc.flatten()

      images.append(img_loc)

  classes = ('alive', 'dead')

  network = Net()

  run_network(images, network)


def run_network(images, network):
  print("the")


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained = True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer,
                                                        num_classes)

    return model

def get_transform(train):
  transform = []
  transform.append(T.ToTensor())

  if train:
    transform.append(T.RandomHorizontalFlip(0.5))

  return T.Compose(transform)



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
  device = torch.device('cpu')

  dataset = OysterDataset("/Users/darwin/Downloads/Testing",
                          "/Users/darwin/Downloads/Testing/P1010129.csv",
                          get_transform(train = True))

  dataset_test = OysterDataset("/Users/darwin/Downloads/Testing",
                              "/Users/darwin/Downloads/Testing/P1010129.csv",
                             get_transform(train = False))

  indices = torch.randperm(len(dataset)).tolist()
  dataset = torch.utils.data.Subset(dataset, indices[:-50])
  dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

  data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle = False, num_workers=4,
                                            collate_fn = utils.collate_fn)

  data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=2, shuffle = False, num_workers=4,
                                                  collate_fn=utils.collate_fn)

  model = get_model_instance_segmentation(num_classes)

  model.to(device)

  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.8, weight_decay=0.0005)

  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)


  num_epochs = 10

  for epoch in range(num_epochs):
    print(epoch);
      
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

    print(epoch);

    lr_scheduler.step()

    evaluate(model, data_loader_test, device=device)

    

  print("That's it!")



if __name__ == "__main__":
  main()













