import json
from pathlib import Path

import torch
import torchvision
from torchvision.transforms import v2
from PIL import Image
from torch.utils.data import Dataset


class Round11SampleDataset(Dataset):
    def __init__(self, root, img_exts=["jpg", "png"], class_info_ext="json", split='test'):
        root = Path(root)
        train_augmentation_transforms = torchvision.transforms.Compose(
            [
                v2.PILToTensor(),
                v2.RandomPhotometricDistort(),
                torchvision.transforms.RandomCrop(size=256, padding=int(10), padding_mode='reflect'),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ConvertImageDtype(torch.float),
            ]
        )

        test_augmentation_transforms = torchvision.transforms.Compose(
            [
                v2.PILToTensor(),
                torchvision.transforms.ConvertImageDtype(torch.float),
            ]
        )
        if split == 'test':
            self.transform = test_augmentation_transforms
        else:
            self.transform = train_augmentation_transforms

        self._img_directory_contents = sorted([path for path in root.glob("*.*") if path.suffix[1:] in img_exts])

        self.data = []
        for img_fname in self._img_directory_contents:
            full_path = root / img_fname
            json_path = root / Path(img_fname.stem + f".{class_info_ext}")
            assert json_path.exists(), f"No {class_info_ext} found for {img_fname}"
            pil_img = Image.open(full_path)
            with open(json_path, 'r') as f:
                label = json.load(f)
            self.data.append((pil_img, label))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]

        img = self.transform(img)
        return img, label