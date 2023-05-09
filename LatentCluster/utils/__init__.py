from typing import Iterable, Any, Tuple

from torch.utils.data import Sampler
import numpy as np

class ProgressSampler(Sampler):
    def __init__(self, ds_length, progress, chunk_size):
        self.ds_length = ds_length
        self.progress = progress
        self.chunk_size = chunk_size
    
    def __iter__(self):
        return iter(range(self.progress, self.ds_length, self.chunk_size))
    
    def __len__(self):
        return (self.ds_length // self.chunk_size) - (self.progress // self.chunk_size)

def clear_tailing_zeros(x: np.ndarray) -> np.ndarray:
    n, d = x.shape
    tailing_zeros = 0
    for i in range(n - 1, -1, -1):
        if np.sum(x[i]) == 0:
            tailing_zeros += 1
        else:
            break
    return x[:n - tailing_zeros]

def random_sample(data : Iterable[Iterable[Any]], sample_size: int, seed: int = 42) -> Iterable[Iterable[Any]]:
    """
    Given a list of corresponding data points (i.e. vector, text, label), returns a random sample that maintains
    order between corresponding points. i.e. if data[0][i] ~ data[1][i] ~ ..., then res[0][i] ~ res[1][i] ~ ...
    """
    np.random.seed(seed)
    indices = np.arange(len(data[0]))
    np.random.shuffle(indices)
    sampled_indices = indices[:sample_size]

    res = [[data_i[i] for i in sampled_indices] for data_i in data]

    for i in range(len(res)):
        try:
            new_val = np.stack(res[i])
        except:
            new_val = res[i]
        
        res[i] = new_val
    
    return res