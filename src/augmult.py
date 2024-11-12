import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
from typing import Sequence, Tuple, Optional

class AugmultConfig:
    """Configuration for image augmentation with multiplicity.

    Args:
        augmult (int): Number of augmentation multiplicities to use. `augmult=0` 
            for no augmentation, `augmult=1` for one augmented view per batch,
            and `augmult>1` for multiple augmented views of each sample.
        random_crop (bool): Whether to apply random cropping.
        random_flip (bool): Whether to apply random horizontal flip.
        random_color (bool): Whether to apply random color jittering.
        pad (Optional[int]): Padding before cropping. Defaults to 4.
    """
    def __init__(self, augmult: int, random_crop: bool, random_flip: bool, 
                 random_color: bool, pad: Optional[int] = 4):
        self.augmult = augmult
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.random_color = random_color
        self.pad = pad

    def apply(
        self, image: torch.Tensor, label: int, crop_size: Sequence[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return apply_augmult(
            image,
            label,
            augmult=self.augmult,
            random_flip=self.random_flip,
            random_crop=self.random_crop,
            random_color=self.random_color,
            pad=self.pad,
            crop_size=crop_size,
        )


def apply_augmult(
    image: torch.Tensor,
    label: int,
    augmult: int,
    random_flip: bool,
    random_crop: bool,
    random_color: bool,
    crop_size: Sequence[int],
    pad: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies multiple augmentations to a single image.

    Args:
        image: Tensor representing the image to augment.
        label: Label for the image.
        augmult: Number of augmentation multiplicities.
        random_flip: Whether to apply horizontal flips.
        random_crop: Whether to apply random cropping.
        random_color: Whether to apply color jitter.
        pad: Optional padding before cropping.
        crop_size: Crop size after padding.
        
    Returns:
        A tuple of (augmented images, repeated labels).
    """
    if augmult == 0:
        # No augmentations; return original image and label with new dimension.
        images = image.unsqueeze(0)
        labels = torch.tensor([label])
    else:
        augmented_images = []
        base_image = image.clone()

        for _ in range(augmult):
            img = base_image.clone()

            # Apply padding and random crop if required
            if random_crop:
                if pad:
                    img = F.pad(img, (pad, pad, pad, pad), mode='reflect')
                img = T.RandomCrop(crop_size)(img)

            # Apply random horizontal flip
            if random_flip:
                img = T.RandomHorizontalFlip()(img)

            # Apply random color jittering
            if random_color:
                color_jitter = T.ColorJitter(
                    brightness=0.15, contrast=0.3, saturation=0.2, hue=0.1
                )
                img = color_jitter(img)

            augmented_images.append(img)

        images = torch.stack(augmented_images)
        labels = torch.tensor([label] * augmult)

    return images, labels


# Example usage:
# Assuming `dataset` is your Dataset instance and `image`, `label` are retrieved from it.
augmult_config = AugmultConfig(
    augmult=3,
    random_crop=True,
    random_flip=True,
    random_color=True,
    pad=4,
)

# Assume `image` is a torch.Tensor representing a single image and `label` is the corresponding label.
image, label = torch.rand(3, 32, 32), 1  # Example image with shape (C, H, W)
augmented_images, repeated_labels = augmult_config.apply(image, label, crop_size=(32, 32))

print("Augmented images shape:", augmented_images.shape)  # (augmult, C, H, W)
print("Repeated labels shape:", repeated_labels.shape)  # (augmult,)
