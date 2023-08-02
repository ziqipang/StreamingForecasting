''' The subgraph networks are for polylines and history trajectories
    The global gnn is for interaction across all the elements
'''
import torch
import torch.nn as nn


class SubgraphMLP(nn.Module):
    def __init__(self, config, in_dim):
        super(SubgraphMLP, self).__init__()
        self.config = config 
        self.subgraph_depth = config['subgraph_depth']
        self.in_dim = in_dim
        self.hidden_dim = [32 * (2 ** i) for i in range(self.subgraph_depth)]
        
        encoders = list()
        in_channels = self.in_dim
        for i in range(self.subgraph_depth):
            encoders.append(nn.Sequential(
                nn.Linear(in_channels, self.hidden_dim[i]),
                nn.LayerNorm(self.hidden_dim[i]),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim[i], self.hidden_dim[i]),
                nn.LayerNorm(self.hidden_dim[i]),
                nn.ReLU(inplace=True)
            ))
            in_channels = self.hidden_dim[i] * 2
        self.encoders = nn.ModuleList(encoders)
    
    def forward(self, feats):
        ''' Forward the subgraph information.
            The following tensor shapes are both valid.
        Args:
            (1) feats: agent_num * obs_len * in_dim / lane_num * seg_num * in_dim
            (2) feats: batch_size * max_agent_num * obs_len * in_dim / batch_size * max_lane_num * seg_num * in_dim
        Return
            (1) feats: agent_num * hidden_dim / lane_num * hidden_dim
            (2) feats: batch_size * max_agent_num * hidden_dim / batch_szie * max_lane_num * hidden_dim
        '''
        seg_num = feats.shape[-2]
        for i in range(self.subgraph_depth):
            feats = self.encoders[i](feats)
            # max pooling over obs_len or lane segments
            feats_max_pool = torch.max(feats, dim=-2, keepdim=True)[0]
            # concat over feature dimensions
            feats = torch.cat((feats, feats_max_pool.repeat((1, 1, seg_num, 1))), dim=-1)
        feats = torch.max(feats, -2)[0]
        return feats