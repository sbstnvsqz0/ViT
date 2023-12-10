from PIL import Image
from os.path import isfile, join
import os
from torch.utils.data import Dataset
from torchvision.transforms import v2

class Yoga(Dataset):
    def __init__(self,images_dir, dictionary, n_classes,transforms=None):
        assert n_classes == 6 or n_classes==20 or n_classes== 82
        self.images_dir = images_dir
        self.dictionary = dictionary
        self.n_classes = n_classes
        
        self.paths = []
        for k in self.dictionary.keys():
            aux = [self.images_dir+"/"+k+"/"+f for f in os.listdir(self.images_dir+"/"+k) if isfile(join(self.images_dir+"/"+k,f))]
            self.paths+=aux
        
        if transforms == None:
            self.transforms = v2.Compose([v2.Resize([224,224]), v2.ToImage(),v2.ToDtype(torch.float32,scale=True)])
        else:
            self.transforms = transforms
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,idx):
        img_path = self.paths[idx]
        position = img_path.split("/")[1]
        labels = self.dictionary[position]
        label = labels[0] if self.n_classes==6 else labels[1] if self.n_classes==20 else labels[2]
        image = Image.open(img_path)
        image = image.convert('RGB')
        image = self.transforms(image)

        return image, label 