import torch
import torch.nn as nn 
import torch.nn.functional as F 
from .encoder import (GlobalAttentionLayer, GlobalAttentionGraph, SubgraphMLP)
from .batching import (batch_actors, batch_lanes)
from .head import RegressionDecoder


class VectorNet(nn.Module):
    def __init__(self, config):
        super(VectorNet, self).__init__()
        self.config = config
        self.num_mod = config['num_mods']
        
        # VectorNet parts
        self.actor_subgraphs = SubgraphMLP(config, 6)
        self.lane_subgraphs = SubgraphMLP(config, 7)
        self.globalgraphs = GlobalAttentionGraph(config)
        
        # decoder
        self.multimod = config['multimod']
        if self.multimod:
            self.decoder = RegressDecoderMultiModFeature(config)
        else:
            self.decoder = RegressionDecoder(config)
        return
    
    def forward(self, data, device='cuda:0'):
        # batching the actor and lane information
        actors, actor_nums, actor_idcs = batch_actors(data['actors'], device)
        lanes, lane_ctrs, lane_nums = batch_lanes(data['graph'], device)
        actor_ctrs = [a.to(device, non_blocking=True) for a in data['ctrs']]
        batch_size = len(actor_ctrs)

        # actor and lane subgraph features
        actors = self.actor_subgraphs(actors) # batch_size * max_actor_num * hidden_dim
        lanes = self.lane_subgraphs(lanes) # batch_size * max_lane_num * hidden_dim

        # batch for global interaction
        batch_size = actors.shape[0]
        max_actor_num = actors.shape[1]

        max_node_num = max_actor_num + lanes.shape[1]
        node_feats = torch.cat((actors, lanes), dim=1)
        node_masks = torch.zeros(batch_size, max_node_num).to(node_feats.device)
        for i in range(batch_size):
            node_masks[i, :actor_nums[i]] = 1
            node_masks[i, max_actor_num:max_actor_num + lane_nums[i]] = 1
        
        # global interaction graph
        node_feats = self.globalgraphs(node_feats, node_masks)
        actor_feats = [node_feats[i, :actor_nums[i]] for i in range(batch_size)]
        actor_feats = torch.cat(actor_feats, dim=0)
        if self.multimod:
            N, hidden_dim = actor_feats.shape
            actor_feats = actor_feats.view(N, 1, hidden_dim).repeat(1, self.num_mod, 1).contiguous()
        
        # decode the trajectories
        output = self.decoder(actor_feats, actor_idcs, actor_ctrs)

        # transform
        norm_rot, norm_center = data['norm_rot'], data['norm_center']
        batch_size = len(output['prediction'])
        for i in range(batch_size):
            world_coord_preds = torch.matmul(output['prediction'][i], norm_rot[i].to(device))
            world_coord_preds += norm_center[i].to(device).view(1, 1, 1, -1)
            output['prediction'][i] = world_coord_preds
        
        output['indexes'] = [None for _ in range(batch_size) ]
        for i in range(batch_size):
            output['indexes'][i] = data['indexes'][i]
        return output