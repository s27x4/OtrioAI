from typing import List, Tuple
import torch


def augment_samples(samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    augmented: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def _rot_flip(st: torch.Tensor, pol: torch.Tensor, val: torch.Tensor):
        board = pol.reshape(3, 3, 3)
        for k in range(4):
            s = torch.rot90(st, k, dims=[1, 2])
            p = torch.rot90(board, k, dims=[1, 2]).reshape(27)
            augmented.append((s, p, val))
            sf = torch.flip(s, dims=[2])
            pf = torch.flip(p.reshape(3, 3, 3), dims=[2]).reshape(27)
            augmented.append((sf, pf, val))

    for state, policy, value in samples:
        _rot_flip(state, policy, value)
        num_players = (state.shape[0] - 1) // 3
        if num_players == 2:
            swapped = state.clone()
            swapped[0:3], swapped[3:6] = state[3:6], state[0:3]
            swapped[-1] = 1.0 - state[-1]
            _rot_flip(swapped, policy, -value)
    return augmented
