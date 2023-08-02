import numpy as np
import torch


def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch


def from_numpy(data):
    ''' Transform every numpy.ndarray to torch.Tensor
    '''
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data