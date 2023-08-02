''' Compute the loss jointly from all the actors
    Largely adapted from LaneGCN
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class JointLoss(nn.Module):
    def __init__(self, config) -> None:
        super(JointLoss, self).__init__()
        self.config = config
        self.num_modalities = self.config['num_mods']
        self.num_preds = self.config['num_preds']

        self.pred_loss = nn.SmoothL1Loss(reduction='sum')
    
    def forward(self, out, gt_futures, gt_future_masks, device='cuda:0'):
        ''' Compute the loss for the output
            Args:
                out: output for all the agents
                gt_futures: tensors of ground truth futures, length of agents
                gt_future_masks: tensors of binary masks, indicating the availability of ground truth on each frame
            Return:
                conf_loss, pred_loss, and combined loss
        '''
        # ========== Preprare the data ========== #
        confs, preds = out['confidence'], out['prediction']
        confs = torch.cat(confs, dim=0)  # N * K
        preds = torch.cat(preds, dim=0)  # N * K * T * 2
        # N * T * 2
        gt_futures = torch.cat([x for x in gt_futures], dim=0).to(device, non_blocking=True)
        # N * T
        gt_future_masks = torch.cat([x for x in gt_future_masks], dim=0).to(device, non_blocking=True)

        # ========== Find the branch with min FDE to compute loss ========== #
        # 1. find the actors with ground truth; 2. find the last position of available ground truth
        last = gt_future_masks.float() + \
            0.1 * torch.arange(self.num_preds).float().to(device) / float(self.num_preds)
        max_last_vals, max_last_idcs = last.max(1) # N, N
        mask = (max_last_vals > 1.0) # N, actors that have future ground truth
        confs, preds, gt_futures, gt_future_masks = confs[mask], preds[mask], gt_futures[mask], gt_future_masks[mask]
        last_idcs = max_last_idcs[mask]

        # 2. select the branches that minimize FDE values
        actor_num = last_idcs.shape[0]
        actor_idcs = torch.arange(actor_num).long().to(device, non_blocking=True)
        pred_final = preds[actor_idcs, :, last_idcs] # N * K * 2
        gt_final = gt_futures[actor_idcs, last_idcs] # N * 2
        dist = pred_final - gt_final.unsqueeze(1)    # N * K * 2
        dist = (dist ** 2).sum(2)                    # N * K
        min_dist, branch_idx = dist.min(1)       # N, and N

        # ========== Confidence Loss ========== #
        conf_loss = F.cross_entropy(confs, branch_idx.long(), reduction='sum')
        # mgn = confs[actor_idcs, branch_idx].unsqueeze(1) - confs
        # mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        # mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        # mgn = mgn[mask0 * mask1]
        # mask = mgn < self.config["mgn"]
        # coef = self.config["cls_coef"]
        # conf_loss = coef * (
        #     self.config["mgn"] * mask.sum() - mgn[mask].sum()
        # )
        num_conf = mask.sum().item()

        # ========== Prediction Loss ========== #
        pred_branches = preds[actor_idcs, branch_idx]
        pred_loss = self.pred_loss(pred_branches[gt_future_masks], gt_futures[gt_future_masks])

        # ========== Wrap up ========== #
        num_conf = num_pred = actor_num
        conf_loss /= (num_conf + 1e-10)
        pred_loss /= (num_pred + 1e-10)
        loss = {
            'conf': conf_loss,
            'pred': pred_loss,
            'loss': conf_loss + pred_loss
        }
        return loss