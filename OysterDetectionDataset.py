import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
import pandas as pd
from skimage import io, transform
from PIL import Image
import numpy as np

class OysterDetectionDataset(Dataset):
    ''' Dataset for detection of oyster images '''

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

        # following PedFundanDataset
        img = Image.open(img_name).convert("RGB")
        img = np.array(img)

        # Getting Labels
        num_objs = self.oysters.iloc[index, 4]
        print("There are ", num_objs, "objects in ", img_name)
        labels = []
        boxes = []
        area = 0

        if num_objs == 0:
            boxes.append([0,0,0,0])
        else:
            for i in range(num_objs):
                pos = 8 + i * 5     # get column for object bounding boxes. boxes start at column 8
                xmin  = self.oysters.iloc[index, pos + 1]
                ymin = self.oysters.iloc[index, pos + 2]
                xmax = self.oysters.iloc[index, pos + 3]
                ymax = self.oysters.iloc[index, pos + 4]
                boxes.append([xmin, ymin, xmax, ymax])

                obj_class = self.oysters.iloc[index, pos]   # get class of object 0,1, or 2
                labels.append([obj_class])

                print("Boxes: ", boxes[i], "\nClass: ", labels[i])

            
 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        print(boxes)
        
        labels = torch.as_tensor(labels, dtype=torch.int64)
        print(labels)                

        img_id = self.oysters.iloc[index, 0]    # name of image
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # area of image
        print(area)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["imgage_id"] = img_id
        target["area"] = area

        if self.transform is not None:
            img = self.transform(img)

        return img, target