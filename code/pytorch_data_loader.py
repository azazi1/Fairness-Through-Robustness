# Inherits from pytorch Dataset class; used to iterate over data

import torch

class PytorchLoader(torch.utils.data.Dataset):

    def __init__(self, ds, portion='train', do_analysis=False):
        """
        This is a custom implementation to load the Adience Dataset
        
        ds: is an instance of Dataloader defined in `data_loader.py`
        portion: train or test
        do_analysis: if True, will save some data analysis to a file (TODO: To be Implemented!)
        """
        self.ds = ds
        self.portion = portion

    def __getitem__(self, index):
        """
        Needs to be defined since we're inheriting from PyTorch's Dataset class
        
        Given an index, fetches the images using ds.get_image
        """
        image = self.ds.get_image(self.portion, index)
        class_label = int(self.ds.get_image_label(self.portion, index))
        if 'utkface' in self.ds.name.lower():
            attr = self.ds.name.lower().split('_')[-1]
            protected_class = int(self.ds.get_image_protected_class(self.portion, index, attr=attr))
        elif 'linear' in self.ds.name.lower():
            protected_class = int(self.ds.get_image_protected_class(self.portion, index))
        else:
            protected_class = -1 # no protected class here, will have to manually define it later

        return (index, image, class_label, protected_class)

    def __len__(self):
        """
        Needs to be defined since we're inheriting from PyTorch's Dataset class
        """
        return self.ds.length(self.portion)

