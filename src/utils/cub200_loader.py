"""
CUB-200-2011 Dataset Loader

Loads the Caltech-UCSD Birds 200 dataset for multimodal FL experiments.
Supports image + attribute modalities.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CUB200Dataset(Dataset):
    """
    CUB-200-2011 Birds Dataset.
    
    A fine-grained visual categorization dataset with:
    - 11,788 images of 200 bird species
    - 312 binary attributes per image
    - Bounding box annotations
    
    For multimodal FL, we use:
    - Image modality: Bird photos (224x224)
    - Attribute modality: 312-dim binary vector
    
    Download from: https://www.vision.caltech.edu/datasets/cub_200_2011/
    """
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform=None,
        download: bool = False,
        use_attributes: bool = True
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.use_attributes = use_attributes
        
        self.data_dir = os.path.join(root, "CUB_200_2011")
        
        if download:
            self._download()
        
        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. Set download=True or download manually from "
                "https://www.vision.caltech.edu/datasets/cub_200_2011/"
            )
        
        self._load_data()
    
    def _check_exists(self) -> bool:
        """Check if dataset exists."""
        return os.path.exists(os.path.join(self.data_dir, "images"))
    
    def _download(self):
        """Download the dataset."""
        if self._check_exists():
            logger.info("CUB-200 already exists, skipping download")
            return
        
        os.makedirs(self.root, exist_ok=True)
        
        # Note: CUB-200 requires manual download due to size/license
        logger.warning(
            "CUB-200 automatic download not implemented. "
            "Please download manually from: "
            "https://www.vision.caltech.edu/datasets/cub_200_2011/ "
            f"and extract to {self.data_dir}"
        )
    
    def _load_data(self):
        """Load image paths, labels, and attributes."""
        # Load image list
        images_file = os.path.join(self.data_dir, "images.txt")
        self.image_paths = []
        self.image_ids = []
        
        with open(images_file, 'r') as f:
            for line in f:
                img_id, img_path = line.strip().split()
                self.image_ids.append(int(img_id))
                self.image_paths.append(os.path.join(self.data_dir, "images", img_path))
        
        # Load labels
        labels_file = os.path.join(self.data_dir, "image_class_labels.txt")
        self.labels = {}
        with open(labels_file, 'r') as f:
            for line in f:
                img_id, label = line.strip().split()
                self.labels[int(img_id)] = int(label) - 1  # 0-indexed
        
        # Load train/test split
        split_file = os.path.join(self.data_dir, "train_test_split.txt")
        self.is_train = {}
        with open(split_file, 'r') as f:
            for line in f:
                img_id, is_train = line.strip().split()
                self.is_train[int(img_id)] = int(is_train) == 1
        
        # Filter by train/test
        self.indices = [
            i for i, img_id in enumerate(self.image_ids)
            if self.is_train[img_id] == self.train
        ]
        
        # Load attributes if needed
        self.attributes = None
        if self.use_attributes:
            self._load_attributes()
        
        logger.info(
            f"Loaded CUB-200 {'train' if self.train else 'test'}: "
            f"{len(self.indices)} images"
        )
    
    def _load_attributes(self):
        """Load image attributes (312-dim binary vectors)."""
        attr_file = os.path.join(self.data_dir, "attributes", "image_attribute_labels.txt")
        
        if not os.path.exists(attr_file):
            logger.warning("Attributes file not found, disabling attributes")
            self.use_attributes = False
            return
        
        # Initialize attributes matrix
        num_images = len(self.image_paths)
        num_attributes = 312
        self.attributes = np.zeros((num_images, num_attributes), dtype=np.float32)
        
        with open(attr_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                img_id, attr_id, is_present = int(parts[0]), int(parts[1]), int(parts[2])
                # CUB uses certainty levels, we just use presence
                self.attributes[img_id - 1, attr_id - 1] = 1.0 if is_present else 0.0
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        img_id = self.image_ids[real_idx]
        
        # Load image
        img_path = self.image_paths[real_idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[img_id]
        
        # If using attributes, return them in a tuple
        if self.use_attributes and self.attributes is not None:
            attributes = torch.tensor(self.attributes[real_idx])
            # For now, just return image and label (attributes available separately)
            return image, label
        
        return image, label
    
    def get_attributes(self, idx: int) -> Optional[torch.Tensor]:
        """Get attributes for a sample."""
        if not self.use_attributes or self.attributes is None:
            return None
        real_idx = self.indices[idx]
        return torch.tensor(self.attributes[real_idx])


def get_cub200_transforms(train: bool = True):
    """Get transforms for CUB-200 images."""
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def load_cub200(
    data_dir: str = "./data",
    download: bool = True
) -> Tuple[Dataset, Dataset]:
    """
    Load CUB-200 train and test datasets.
    
    Args:
        data_dir: Directory for data storage
        download: Whether to download if not present
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train_transform = get_cub200_transforms(train=True)
    test_transform = get_cub200_transforms(train=False)
    
    train_dataset = CUB200Dataset(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=download
    )
    
    test_dataset = CUB200Dataset(
        root=data_dir,
        train=False,
        transform=test_transform,
        download=download
    )
    
    return train_dataset, test_dataset


def get_cub200_client_data(
    dataset: Dataset,
    client_id: int,
    num_clients: int,
    partition: str = "iid",
    alpha: float = 0.5
) -> Subset:
    """
    Partition CUB-200 data for a specific client.
    
    Args:
        dataset: Full CUB-200 dataset
        client_id: Client identifier
        num_clients: Total number of clients
        partition: 'iid' or 'dirichlet'
        alpha: Dirichlet concentration parameter
        
    Returns:
        Subset for this client
    """
    n = len(dataset)
    
    if partition == "iid":
        # Equal random split
        indices = list(range(n))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        samples_per_client = n // num_clients
        start = client_id * samples_per_client
        end = start + samples_per_client if client_id < num_clients - 1 else n
        
        client_indices = indices[start:end]
    
    elif partition == "dirichlet":
        # Non-IID split using Dirichlet distribution
        np.random.seed(42)
        
        # Get labels for all samples
        labels = np.array([dataset[i][1] for i in range(n)])
        num_classes = len(np.unique(labels))
        
        # Generate Dirichlet distribution
        client_indices = []
        for c in range(num_classes):
            class_indices = np.where(labels == c)[0]
            np.random.shuffle(class_indices)
            
            # Dirichlet allocation
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = (proportions * len(class_indices)).astype(int)
            
            # Fix rounding errors
            proportions[-1] = len(class_indices) - proportions[:-1].sum()
            
            # Assign to this client
            start = sum(proportions[:client_id])
            end = start + proportions[client_id]
            client_indices.extend(class_indices[start:end].tolist())
        
        np.random.shuffle(client_indices)
    
    else:
        raise ValueError(f"Unknown partition: {partition}")
    
    return Subset(dataset, client_indices)


if __name__ == "__main__":
    # Test the loader
    print("Testing CUB-200 loader...")
    
    try:
        train_data, test_data = load_cub200("./data", download=True)
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")
        
        # Test loading a sample
        image, label = train_data[0]
        print(f"Image shape: {image.shape}")
        print(f"Label: {label}")
        
        # Test partitioning
        client_data = get_cub200_client_data(train_data, 0, 5, "iid")
        print(f"Client 0 samples: {len(client_data)}")
        
    except RuntimeError as e:
        print(f"Note: {e}")
        print("This is expected if CUB-200 is not downloaded yet.")
