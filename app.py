'''
Authors: Martin Jauquet and Darwin Castillo
Date: 11/20/2020

About: This application will take a file path of an image file 
and pass it through our model to determine the number of alive,
dead, and total oysters in the image

'''
# Imports
import sys
import os

import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from PIL import Image


# Main function
def main(self):
    num_args = len(sys.argv)

    if num_args == 2:
        image_path = sys.argv[1]

        if not os.path.exists(image_path):
            print('Error: File path does not exist')
        else:
            # Call function to run model here
            runModel(image_path)
    else:
        print('Error: Command should have the form:')
        print('app.exe <Input File Path>')


def runModel(filepath):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Open image and convert to PIL image
    img = get_img_as_tensor(filepath)

    # get model 
    num_classes = 4
    model = get_model(num_classes)
    model.eval()

    # Run model
    with torch.no_grad():
      prediction = model([img.to(device)])

    # print output array
    num_live = (prediction[0]['labels'] == 1).sum(dim=0).item()
    num_dead = (prediction[0]['labels'] == 2).sum(dim=0).item()
    total_oysters = num_live + num_dead
    output = [total_oysters, num_live, num_dead]
    print(output)

# Returns Model to be used for detection
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    hidden_layer = 256

    PATH = 'detection_net.pth'
    model.load_state_dict(torch.load(PATH))

    return model

# Return image as transform so it can be run through model
def get_img_as_tensor(filepath):
    # Open image as PIL image
    img = Image.open(filepath).convert("RGB")

    # get transform
    image_transform = get_transform()

    # transform image 
    img = image_transform(img)

    return img

# Return transform for image
def get_transform():
  transforms = []
  transforms.append(T.ToTensor())

  return T.Compose(transforms)

if __name__ == "__main__":
    main(sys.argv)