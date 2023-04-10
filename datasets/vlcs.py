from typing import Optional
import os
from glob import glob
from PIL import Image, ImageFile
from typing import Optional, Callable, Tuple, Any, List
from .imagelist import ImageList
from .utils import download as download_data, check_exits, gdrive_download

ImageFile.LOAD_TRUNCATED_IMAGES = True


class VLCS(ImageList):
    download_list = [
        ("VLCS", "VLCS.tar.gz", "https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8")
    ]
    image_list = {
        "V": "VOC2007/*/*",
        "L": "LabelMe/*/*",
        "C": "Caltech101/*/*",
        "S": "SUN09/*/*"
    }
    CLASSES = ['bird', 'car', 'chair', 'dog', 'person']

    def __init__(self, root: str, task: str, download: Optional[bool] = True, **kwargs):
        assert task in self.image_list
        data_list_file = sorted(glob(os.path.join(root, self.image_list[task])))

        if download:
            list(map(lambda args: gdrive_download(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(VLCS, self).__init__(root, VLCS.CLASSES, data_list_file=data_list_file, target_transform=lambda x: x - 1,
                                   **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())

    def parse_data_file(self, file_name: str) -> List[Tuple[str, int]]:
        """Parse file to data list

        Args:
            file_name (str): The path of data file
            return (list): List of (image path, class_index) tuples
        """
        data_list = []

        for file in file_name:
            data_list.append(
                [file, VLCS.CLASSES.index(file.split('/')[-2].strip())+1]
            )
        return data_list

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)
