''' Global interaction graph in the style of VectorNet
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAttentionLayer(nn.Module):
    def __init__(self, config):
        super(GlobalAttentionLayer, self).__init__()
        self.config = config 
        self.hidden_dim = 256
        
        self.q_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.norm = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True))
    
    def forward(self, feats, masks):
        ''' Compute the self attention across the actors and lanes
        Args:
            feats: batch_size * max_node * hidden_dim, features of each node
            masks: batch_size * max_node, 0 or 1, whether a node contains valid information
        Return:
            feats: batch_size * max_node * hidden_dim
        '''
        # ========== Compute the cross attention ========== #
        q, k, v = self.q_layer(feats), self.k_layer(feats), self.v_layer(feats) # batch_size * node_num * hidden_dim
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        
        # ========== Filter out the invalid pairs (those padded) ========== #
        max_node = feats.shape[1]
        attention_masks = masks.unsqueeze(2) * masks.unsqueeze(1)
        # learn this trick from DenseTNT, thanks! The scores after softmax will be extremely small
        attention_masks = attention_masks - (1e5 * (1- attention_masks)) 
        scores = scores + attention_masks
        scores = F.softmax(scores, -1) # batch_size * max_node * max_node
        
        # ========== Weighted sum information from values ========== #
        cross_attention_feats = torch.matmul(scores, v) #
        cross_attention_feats = self.norm(cross_attention_feats)
        return cross_attention_feats


class GlobalAttentionGraph(nn.Module):
    def __init__(self, config):
        super(GlobalAttentionGraph, self).__init__()
        self.config = config
        self.hidden_dim = 256
        self.global_graph_depth = self.config['global_graph_depth']
        
        self.global_attention_layers = nn.ModuleList([
            GlobalAttentionLayer(config) for _ in range(self.global_graph_depth)])
        self.linear_cross_attn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim)
            ) for _ in range(self.global_graph_depth)])
        self.linears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim)
            ) for _ in range(self.global_graph_depth)])
    
    def forward(self, feats, masks):
        for i in range(self.global_graph_depth):
            cross_attention_feats = self.global_attention_layers[i](feats, masks)
            cross_attention_feats = self.linear_cross_attn_layers[i](cross_attention_feats)
            feats = self.linears[i](feats) + cross_attention_feats
            if i < self.global_graph_depth - 1:
                feats = F.relu(feats)
        return feats