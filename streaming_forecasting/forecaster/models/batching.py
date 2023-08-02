''' For VectorNet, we have to pad according to the maximum number of actors/lanes
'''
import torch
import torch.nn as nn 
import torch.nn.functional as F 


def batch_actors(actors, device):
    ''' Gather the actors into a batch
    Args: 
        actors: list of features of actors
    Return:
        actor features: batch_size * max_actor * obs_len, feat_dim
        actor_num: batch_size
    '''
    batch_size = len(actors)
    feat_len, feat_dim = actors[0].shape[-2:]
    actor_num = [actors[i].shape[0] for i in range(batch_size)]
    
    result = torch.zeros(batch_size, max(actor_num), feat_len, feat_dim).float()
    for i in range(batch_size):
        result[i, :actor_num[i], :, :] = actors[i]
    result = result.to(device)
    
    actor_idcs = list()
    count = 0
    for i in range(batch_size):
        sample_idcs = torch.arange(count, count + actor_num[i]).to(device, non_blocking=True)
        actor_idcs.append(sample_idcs)
        count += actor_num[i]
    return result, actor_num, actor_idcs


def batch_lanes(graph, device):
    ''' Form the lane features and gather them into a batch
        Args:
            graph
        Return:
            lane_features: batch_size * max_lane * seg_num * feat_dim
            lane_num: batch_size
    '''
    batch_size = len(graph)
    seg_num = graph[0]['centerlines'].shape[-2]
    lane_num = [graph[i]['centerlines'].shape[0] for i in range(batch_size)]
    
    result = torch.zeros(batch_size, max(lane_num), seg_num - 1, 7).float()
    lane_ctrs = torch.zeros(batch_size, max(lane_num), 2)
    for i in range(batch_size):
        centerlines = graph[i]['centerlines']
        lane_ctrs[i, :lane_num[i]] = torch.mean(centerlines, dim=1)
        result[i, :lane_num[i], :, :2] = centerlines[:, 1:, :]
        result[i, :lane_num[i], :, 2:4] = centerlines[:, 1:, :] - centerlines[:, :-1, :]
        result[i, :lane_num[i], :, 4] = graph[i]['turns'].unsqueeze(1)
        result[i, :lane_num[i], :, 5] = graph[i]['intersections'].unsqueeze(1)
        result[i, :lane_num[i], :, 6] = graph[i]['controls'].unsqueeze(1)
    result = result.to(device)
    lane_ctrs = lane_ctrs.to(device)
    return result, lane_ctrs, lane_num