''' Samples across a batch may have different lane/actor numbers.
    Gather functions handle how to form batches and copy to gpus correctly.
    Largely adapted from LaneGCN.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def gather_actors(actors, device):
    ''' Gather the actors into a batch
        Args:
            actors: list of features of actors, length of batch size
            device: target device, useful for distirbuted training
        Return:
            concatenated actor features and the indices of each sample
    '''
    batch_size = len(actors)

    # each actor has the shape of N * T * C
    actors = [a.transpose(1, 2) for a in actors]
    actor_feats = torch.cat(actors, dim=0).to(device, non_blocking=True)

    # indices
    actor_idcs = list()
    count = 0
    for i in range(batch_size):
        actor_num = actors[i].shape[0]
        sample_idcs = torch.arange(count, count + actor_num).to(device, non_blocking=True)
        actor_idcs.append(sample_idcs)
        count += actor_num
    return actor_feats, actor_idcs


def gather_lanes(graph, device):
    ''' Gather the graph information into a batch
        Args:
            graph: graph information loaded from data loader
            device: target device, useful for distirbuted training
        Return:
            1. concatenated laner features.
            2. calculate the in-batch indices and set them to long
            3. copy everything to gpu
    '''
    result = dict()
    batch_size = len(graph)

    # in-batch indices
    lane_indices, lane_counts = list(), list()
    count = 0
    for i in range(batch_size):
        lane_counts.append(count)
        indices = torch.arange(count, count + graph[i]['num_nodes']).long().to(device, non_blocking=True)
        lane_indices.append(indices)
        count += graph[i]['num_nodes']

    # copy features to gpu
    result['idcs'] = lane_indices
    result['ctrs'] = [g['ctrs'].to(device, non_blocking=True) for g in graph]
    result['segs'] = [g['segs'].to(device, non_blocking=True) for g in graph]

    for key in ['turns', 'controls', 'intersects']:
        result[key] = torch.cat([g[key] for g in graph], dim=0).to(device, non_blocking=True)
    
    # modify the indices in connected nodes
    for key in ['pre', 'suc']:
        result[key] = list()
        for scale in range(len(graph[0][key])):
            result[key].append(dict())
            for uv in ['u', 'v']:
                result[key][scale][uv] = torch.cat(
                    [graph[j][key][scale][uv] + lane_counts[j] for j in range(batch_size)], dim=0
                ).long().to(device, non_blocking=True)
    
    for key in ['left', 'right']:
        result[key] = dict()
        for uv in ['u', 'v']:
            indices = [graph[i][key][uv] + lane_counts[i] for i in range(batch_size)]
            # special case, no connection, use an empty list to represent
            indices = [
                x if x.dim() > 0 else result['pre'][0]['u'].new().resize_(0)
                for x in indices
            ]
            result[key][uv] = torch.cat(indices).long().to(device, non_blocking=True)
    return result