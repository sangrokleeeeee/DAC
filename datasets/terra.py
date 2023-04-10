from typing import Optional
import os
from glob import glob
from PIL import Image, ImageFile
from typing import Optional, Callable, Tuple, Any, List
from .imagelist import ImageList
from .utils import download as download_data, check_exits, gdrive_download

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Terra(ImageList):
    
    image_list = {
        "L38": "location_38.txt",
        "L43": "location_43.txt",
        "L46": "location_46.txt",
        "L100": "location_100.txt"
    }
    CLASSES = [
        "bird", "bobcat", "cat", "coyote", "dog", "empty", "opossum", "rabbit",
        "raccoon", "squirrel"
    ]

    def __init__(self, root: str, task: str, split='all', download: Optional[bool] = True, **kwargs):
        assert task in self.image_list
        assert split in ["train", "val", "all", "test"]
        if split == "test":
            split = "all"
        data_list_file = os.path.join(root, self.image_list[task].format(split))

        # if download:
        #     list(map(lambda args: download_data(root, *args), self.download_list))
        # else:
        #     list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(Terra, self).__init__(root, Terra.CLASSES, data_list_file=data_list_file, target_transform=lambda x: x - 1,
                                   **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())

    # def parse_data_file(self, file_name: str) -> List[Tuple[str, int]]:
    #     """Parse file to data list

    #     Args:
    #         file_name (str): The path of data file
    #         return (list): List of (image path, class_index) tuples
    #     """
    #     data_list = []

    #     for file in file_name:
    #         # if VLCS.CLASSES.index(file.split('/')[-2].strip()) == -1:
    #         #     print('*'*50)
    #         #     print(file.split('/')[-2].strip())
    #         data_list.append(
    #             [file, Terra.CLASSES.index(file.split('/')[-2].strip())+1]
    #         )
    #     return data_list

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)