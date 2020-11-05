import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
      super().__init__()

      # Creation of convolutional layers.
      # Define filters quantity, size, and type.
      self.conv1 = nn.Conv2d(3, 6, 5)
      self.pool = nn.MaxPool2d(5, 5) # Reduces the size of the filter image
      self.conv2 = nn.Conv2d(6, 50, 5)

      # Creation of Linear layer.
      # 
      self.fc1 = nn.Linear(450, 120)
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, 10)
      self.fc4 = nn.Linear(10, 3)

    # x is our data (image)
    # Each line is passing our data through our defined layers above
    def forward(self, x):
      # x = self.function that detects
        # Returns a list of detected objects
      # For each object in list, classify that object
        # x = self. fucntion that classifies() things below
      x = self.pool(F.relu(self.conv1(x)))      # Passing x through first layer
      #print(x)
      x = self.pool(F.relu(self.conv2(x)))
      #print(x)
      x = x.view(-1, 450)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      
      return x