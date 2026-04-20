from __future__ import annotations

from torchvision import transforms


def build_image_transform(image_size: int = 224, train: bool = False) -> transforms.Compose:
    operations = []
    if train:
        operations.extend(
            [
                transforms.Resize((image_size + 16, image_size + 16)),
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
            ]
        )
    else:
        operations.append(transforms.Resize((image_size, image_size)))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(operations)
