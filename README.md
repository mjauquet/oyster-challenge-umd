# Northrop Gruman and Chesapeake Bay Foundation Oyster Signal Detection Challenge

Martin Jauquet and Darwin Castillo

Fall 2020

## Description
We were tasked to create an application that detects the number of living and dead oysters in a given image of the bay floor. To do this, we decided to use the Pytorch libraries to create a model that detects a possible oyster and then classifies each detected object. We trained our model using 4 classes: live oyster, dead oyster, shell, and background. By using a Faster RCNN model with ResNet 50 and a Feature Pyramid Network used for classification. Using a cuda to train, we were able to get an accuracy of 92% on the images provided to us.

## Using our code
To use our model you must download ```app.py``` and ```detection_net.pth``` found [here](https://drive.google.com/drive/folders/1RBsYEbHL81DHCEdgjWhdQi8iTPFIMCIV?usp=sharing). The ``` detection_net.pth``` has the data used by the model to accurately predict the oysters. 

### Set up:
1. To run this program you must have Python 3.8.6 installed. (make sure it is added to your PATH)
2. Once Python is installed, install pytorch and torchvision with Cuda.
3. Use the command line command from [Pytorch](https://pytorch.org/get-started/locally/) to install
4. We used the pip command to install, but conda also works

### Running our program from the command line:
1. Download the files mentioned above into the same folder.
2. On the command line navigate to the folder with our program code.
3. Use the command:  ```python app.py <Image File Path>```    For example, ```python app.py C:\Users\user\Documents\TestingImage.JPG```
4. The output will be printed in the command line
