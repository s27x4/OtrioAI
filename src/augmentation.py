from typing import List, Tuple
import torch


def augment_samples(samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    augmented: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for state, policy, value in samples:
        policy_board = policy.reshape(3, 3, 3)
        for k in range(4):
            s = torch.rot90(state, k, dims=[1, 2])
            p = torch.rot90(policy_board, k, dims=[1, 2]).reshape(27)
            augmented.append((s, p, value))
            sf = torch.flip(s, dims=[2])
            pf = torch.flip(p.reshape(3, 3, 3), dims=[2]).reshape(27)
            augmented.append((sf, pf, value))
    return augmented
