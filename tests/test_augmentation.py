import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from src.augmentation import augment_samples


def test_augment_samples():
    state = torch.zeros(7, 3, 3)
    policy = torch.zeros(27)
    value = torch.tensor(1.0)
    samples = augment_samples([(state, policy, value)])
    assert len(samples) == 8
