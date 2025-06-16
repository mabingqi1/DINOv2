import json
import numpy as np
import torch

from monai.data.dataset import Dataset

class YHNpyDataset(Dataset):
    def __init__(self, 
                 data_path, 
                 transform,
                ):
        """Initialize the dataset by loading the JSON file.

        Args:
            json_path (str): Path to the JSON file containing dataset information.
        """
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.data_list = self.data['data_list']
        self.transform =transform

    def __len__(self):
        """Return the total number of samples in the dataset.

        Returns:
            int: Length of the data_list.
        """
        return len(self.data_list)

    def get_image_data(self, index):
        """Retrieve image data as bytes for the given index.

        Args:
            index (int): Index of the sample.

        Returns:
            bytes: Image data in PNG format as bytes.
        """
        img_path = self.data_list[index]['img_path']
        array = np.load(img_path)[None]
        tensor = torch.from_numpy(array).float()

        return tensor
    
    def __getitem__(self, index: int):
        try:
            image_data = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e

        if self.transform is not None:
            image_dict = self.transform(image_data)
            
        return image_dict