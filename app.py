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


# Main function
def main(self):
    num_args = len(sys.argv)

    if num_args == 2:
        image_path = sys.argv[1]

        if not os.path.exists(image_path):
            print('File path does not exist')
        else:
            # Call function to run model here
            # 
            print(' ')
    else:
        print('Error: Command should have the form:')
        print('app.exe <Input File Path>')


def runModel(filepath):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # get model 
    # num_classes = 4
    # model = get_model_instance_segmentation(num_classes)
    # model.eval()

    # Run model
    # with torch.no_grad()
    #   prediction = model([img.to(device)])

    # print output array
    # num_live = (prediction[0]['labels'] == 1).sum(dim=0).item()
    # num_dead = (prediction[0]['labels'] == 2).sum(dim=0).item()
    # total_oysters = num_live + num_dead
    # output = [total_oysters, num_live, num_dead]
    # print("Predicted: ", output)