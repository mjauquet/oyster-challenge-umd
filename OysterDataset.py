import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
import pandas as pd
from skimage import io, transform
from PIL import Image
import numpy as np

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
