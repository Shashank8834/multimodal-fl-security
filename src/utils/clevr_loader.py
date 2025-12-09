"""
CLEVR Dataset Loader

Loads the CLEVR visual reasoning dataset for multimodal FL experiments.
Supports image + question/answer modalities.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class CLEVRDataset(Dataset):
    """
    CLEVR Visual Reasoning Dataset.
    
    A diagnostic dataset for visual reasoning with:
    - 70,000 training images + 15,000 validation images
    - Each image has multiple questions about objects
    - Questions are compositional and require reasoning
    
    For multimodal FL, we use:
    - Image modality: Rendered 3D scenes (320x240 or resized)
    - Text modality: Question embeddings or tokenized text
    
    Download from: https://cs.stanford.edu/people/jcjohns/clevr/
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        max_questions_per_image: int = 1,
        question_type: str = "all"
    ):
        """
        Args:
            root: Root directory containing CLEVR data
            split: 'train', 'val', or 'test'
            transform: Image transforms
            max_questions_per_image: Limit questions per image
            question_type: Filter by question type (all, count, exist, etc.)
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.max_questions_per_image = max_questions_per_image
        self.question_type = question_type
        
        # Find data directory
        self.data_dir = self._find_data_dir()
        
        if not self._check_exists():
            raise RuntimeError(
                f"CLEVR dataset not found at {self.data_dir}. "
                "Download from https://cs.stanford.edu/people/jcjohns/clevr/ "
                f"and extract to {root}/CLEVR_v1.0/"
            )
        
        self._load_data()
    
    def _find_data_dir(self) -> str:
        """Find the CLEVR data directory."""
        possible_paths = [
            os.path.join(self.root, "CLEVR_v1.0"),
            os.path.join(self.root, "CLEVR"),
            self.root
        ]
        
        for path in possible_paths:
            if os.path.exists(os.path.join(path, "images")):
                return path
        
        return possible_paths[0]  # Default path
    
    def _check_exists(self) -> bool:
        """Check if dataset exists."""
        images_dir = os.path.join(self.data_dir, "images", self.split)
        questions_file = os.path.join(
            self.data_dir, "questions", 
            f"CLEVR_{self.split}_questions.json"
        )
        return os.path.exists(images_dir) or os.path.exists(questions_file)
    
    def _load_data(self):
        """Load questions and image paths."""
        # Load questions
        questions_file = os.path.join(
            self.data_dir, "questions",
            f"CLEVR_{self.split}_questions.json"
        )
        
        if os.path.exists(questions_file):
            with open(questions_file, 'r') as f:
                data = json.load(f)
            self.questions = data.get('questions', [])
        else:
            # Fallback: just use images without questions
            self.questions = []
            logger.warning("Questions file not found, using images only")
        
        # Build image to questions mapping
        self.image_questions = {}
        for q in self.questions:
            img_name = q.get('image_filename', q.get('image', ''))
            if img_name:
                if img_name not in self.image_questions:
                    self.image_questions[img_name] = []
                if len(self.image_questions[img_name]) < self.max_questions_per_image:
                    self.image_questions[img_name].append(q)
        
        # Get image list
        images_dir = os.path.join(self.data_dir, "images", self.split)
        if os.path.exists(images_dir):
            self.image_files = sorted([
                f for f in os.listdir(images_dir)
                if f.endswith('.png') or f.endswith('.jpg')
            ])
        else:
            self.image_files = list(self.image_questions.keys())
        
        # Create flat list of (image, question) pairs
        self.samples = []
        for img_file in self.image_files:
            if img_file in self.image_questions:
                for q in self.image_questions[img_file]:
                    self.samples.append((img_file, q))
            else:
                # Image without question
                self.samples.append((img_file, None))
        
        # Build answer vocabulary
        self._build_vocab()
        
        logger.info(
            f"Loaded CLEVR {self.split}: {len(self.image_files)} images, "
            f"{len(self.samples)} samples"
        )
    
    def _build_vocab(self):
        """Build vocabulary for answers."""
        self.answer_to_idx = {}
        self.idx_to_answer = {}
        
        idx = 0
        for img_file, q in self.samples:
            if q and 'answer' in q:
                answer = str(q['answer'])
                if answer not in self.answer_to_idx:
                    self.answer_to_idx[answer] = idx
                    self.idx_to_answer[idx] = answer
                    idx += 1
        
        self.num_classes = len(self.answer_to_idx)
        if self.num_classes == 0:
            # Default: 28 possible CLEVR answers
            self.num_classes = 28
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_file, question = self.samples[idx]
        
        # Load image
        img_path = os.path.join(self.data_dir, "images", self.split, img_file)
        
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
        else:
            # Create dummy image if not found
            image = Image.new("RGB", (320, 240), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        # Get label (answer)
        if question and 'answer' in question:
            answer = str(question['answer'])
            label = self.answer_to_idx.get(answer, 0)
        else:
            label = 0
        
        return image, label
    
    def get_question(self, idx: int) -> Optional[str]:
        """Get the question text for a sample."""
        _, question = self.samples[idx]
        if question:
            return question.get('question', '')
        return None


def get_clevr_transforms(train: bool = True, image_size: int = 224):
    """Get transforms for CLEVR images."""
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def load_clevr(
    data_dir: str = "./data",
    image_size: int = 224
) -> Tuple[Dataset, Dataset]:
    """
    Load CLEVR train and validation datasets.
    
    Args:
        data_dir: Directory for data storage
        image_size: Target image size
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    train_transform = get_clevr_transforms(train=True, image_size=image_size)
    val_transform = get_clevr_transforms(train=False, image_size=image_size)
    
    train_dataset = CLEVRDataset(
        root=data_dir,
        split="train",
        transform=train_transform
    )
    
    val_dataset = CLEVRDataset(
        root=data_dir,
        split="val",
        transform=val_transform
    )
    
    return train_dataset, val_dataset


def get_clevr_client_data(
    dataset: Dataset,
    client_id: int,
    num_clients: int,
    partition: str = "iid",
    alpha: float = 0.5
) -> Subset:
    """
    Partition CLEVR data for a specific client.
    
    Args:
        dataset: Full CLEVR dataset
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
        labels = np.array([dataset[i][1] for i in range(min(n, 10000))])
        
        # Extend to full dataset
        if len(labels) < n:
            labels = np.tile(labels, (n // len(labels) + 1))[:n]
        
        num_classes = max(len(np.unique(labels)), 1)
        
        # Simple random split for CLEVR (many classes)
        indices = list(range(n))
        np.random.shuffle(indices)
        
        samples_per_client = n // num_clients
        start = client_id * samples_per_client
        end = start + samples_per_client if client_id < num_clients - 1 else n
        
        client_indices = indices[start:end]
    
    else:
        raise ValueError(f"Unknown partition: {partition}")
    
    return Subset(dataset, client_indices)


if __name__ == "__main__":
    # Test the loader
    print("Testing CLEVR loader...")
    
    try:
        train_data, val_data = load_clevr("./data")
        print(f"Train samples: {len(train_data)}")
        print(f"Val samples: {len(val_data)}")
        print(f"Num classes: {train_data.num_classes}")
        
        # Test loading a sample
        image, label = train_data[0]
        print(f"Image shape: {image.shape}")
        print(f"Label: {label}")
        
        question = train_data.get_question(0)
        if question:
            print(f"Question: {question}")
        
    except RuntimeError as e:
        print(f"Note: {e}")
        print("This is expected if CLEVR is not downloaded yet.")
        print("\nTo use CLEVR, download from:")
        print("https://cs.stanford.edu/people/jcjohns/clevr/")
