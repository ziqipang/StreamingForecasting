# Differentiable Filter
import math
import torch
import torch.nn as nn


class DiffKF(nn.Module):
    def __init__(self, K, state_dim) -> None:
        super(DiffKF, self).__init__()
        self.K = K
        self.state_dim = state_dim
        self.process_init = 1.0
        self.observation_init = 1.0
    
    def initialize_beliefs(self, instances):
        max_batch_size, max_agt_num, _, fut_len, _ = instances.fut_trajs.shape
        device = instances.fut_trajs.device

        # init
        instances.initialized = torch.full((max_batch_size, max_agt_num), 0, dtype=torch.bool, device=device)

        # means
        instances.belief_mean = torch.full((max_batch_size, max_agt_num, self.K * self.state_dim), 0, dtype=torch.float32, device=device)

        # covariance, assume it is a constant number
        instances.belief_covariance = torch.full((max_batch_size, max_agt_num, self.K * self.state_dim, self.K * self.state_dim), 0.0, dtype=torch.float32, device=device)

        # dynamic matrix
        self.dynamics_A = torch.zeros(self.K * self.state_dim, self.K * self.state_dim, device=device)
        single_modal_A = torch.zeros(self.state_dim, self.state_dim, device=device)
        for i in range(fut_len):
            if i != fut_len - 1:
                single_modal_A[2 * i, 2 * i + 2] = 1.0
                single_modal_A[2 * i + 1, 2 * i + 3] = 1.0
            else:
                single_modal_A[2 * fut_len - 2, 2 * fut_len - 2] = 1.0
                single_modal_A[2 * fut_len - 1, 2 * fut_len - 1] = 1.0
        for i in range(self.K):
            self.dynamics_A[i * self.state_dim: (i + 1) * self.state_dim, i * self.state_dim: (i + 1) * self.state_dim] = single_modal_A.clone()
        return instances
    
    def forward(self, agt_instances, observations, indexes, rs):
        batch_size = len(indexes)
        device = agt_instances.fut_trajs.device

        for batch_idx in range(batch_size):
            if len(indexes[batch_idx]) == 0:
                continue

            # init the states of the filters with first observation
            init_mask = agt_instances.initialized[batch_idx, indexes[batch_idx]]
            idxes = indexes[batch_idx][init_mask.cpu()]
            init_idxes = indexes[batch_idx][~init_mask.cpu()]
            agt_instances.belief_mean[batch_idx][init_idxes] = observations[batch_idx][~init_mask]
            agt_instances.belief_covariance[batch_idx][init_idxes] += \
                (self.observation_init + rs[batch_idx][~init_mask].view(-1, 1, 1)) * torch.eye(self.K * self.state_dim)[None, :, :].to(device)
            agt_instances.initialized[batch_idx][init_idxes] = True

            agt_num = len(idxes)
            if agt_num == 0:
                continue
            # prediction step
            belief_mean = agt_instances.belief_mean[batch_idx, idxes]
            belief_covariance = agt_instances.belief_covariance[batch_idx, idxes]
            dynamics_covariance = (self.process_init * torch.eye(self.K * self.state_dim)[None, :, :] + \
                torch.zeros(agt_num, self.K * self.state_dim, self.K * self.state_dim)).to(device)

            belief_mean, belief_covariance = self.predict_step(belief_mean.clone(), belief_covariance.clone(), dynamics_covariance)

            # update step
            sample_observations = observations[batch_idx][init_mask]
            # observation_covariance = (self.observation_init * torch.ones(agt_num)[:, None, None].to(device)) * torch.eye(self.K * self.state_dim)[None, :, :].to(device)
            observation_covariance = (rs[batch_idx][init_mask].view(-1, 1, 1) + self.observation_init) * torch.eye(self.K * self.state_dim)[None, :, :].to(device)
            belief_mean, belief_covariance = self.update_step(belief_mean.clone(), belief_covariance.clone(), sample_observations, observation_covariance)

            # write back to the instances
            agt_instances.belief_mean[batch_idx, idxes] = belief_mean
            agt_instances.belief_covariance[batch_idx, idxes] = belief_covariance

        return agt_instances
    
    def predict_step(self, belief_mean, belief_covariance, dynamics_covariance):
        # We assume that the dynamics is a identity matrix
        device = belief_mean.device
        dynamics_A = self.dynamics_A
        # dynamics_A = torch.eye(self.K * self.state_dim).to(device)
        belief_mean = (dynamics_A @ belief_mean.transpose(-1, -2)).transpose(-1, -2).contiguous()
        belief_covariance = (dynamics_A @ belief_covariance @ dynamics_A.transpose(-1, -2) + dynamics_covariance).contiguous()

        return belief_mean, belief_covariance
    
    def update_step(self, belief_mean, belief_covariance, observations, observation_covariance):
        innovation = observations - belief_mean
        innovation_covariance = belief_covariance + observation_covariance
        kalman_gain = belief_covariance @ torch.inverse(innovation_covariance)

        corrected_belief_mean = belief_mean + (kalman_gain @ innovation[:, :, None]).squeeze(-1)
        
        identity = torch.eye(self.K * self.state_dim, device=kalman_gain.device)
        corrected_belief_covariance = (identity - kalman_gain) @ belief_covariance
        return corrected_belief_mean, corrected_belief_covariance


class MultiDiffKF(DiffKF):
    def __init__(self, K, state_dim) -> None:
        super(MultiDiffKF, self).__init__(K, state_dim)

        # initialize the configurations
        self.K = K
        self.state_dim = state_dim
        self.process_init = 5.0
        self.observation_init = 1.0
        self.embed_dim = 256
        return
    
    def initialize_beliefs(self, instances):
        max_batch_size, max_agt_num, _, fut_len, _ = instances.fut_trajs.shape
        device = instances.fut_trajs.device

        # init
        instances.mm_initialized = torch.full((max_batch_size, max_agt_num), 0, dtype=torch.bool, device=device)
        instances.mm_branch_idx = torch.full((max_batch_size, max_agt_num), -1, dtype=torch.long, device=device)

        # agent features
        instances.mm_agt_feats = torch.full((max_batch_size, max_agt_num, self.embed_dim), 0, dtype=torch.float32, device=device)
        
        # means for trajectories
        instances.mm_belief_mean = torch.full((max_batch_size, max_agt_num, self.K, self.state_dim), 0, dtype=torch.float32, device=device)
        instances.mm_belief_conf = torch.full((max_batch_size, max_agt_num, self.K), 0, dtype=torch.float32, device=device)

        # covariance
        instances.mm_belief_cov = torch.full((max_batch_size, max_agt_num, self.K, self.state_dim, self.state_dim), 0, dtype=torch.float32, device=device)
        
        # dynamic matrix
        self.dynamics_A = torch.zeros(self.state_dim, self.state_dim, device=device)
        single_modal_A = torch.zeros(self.state_dim, self.state_dim, device=device)
        for i in range(fut_len):
            if i != fut_len - 1:
                single_modal_A[2 * i, 2 * i + 2] = 1.0
                single_modal_A[2 * i + 1, 2 * i + 3] = 1.0
            else:
                single_modal_A[2 * fut_len - 2, 2 * fut_len - 2] = 1.0
                single_modal_A[2 * fut_len - 1, 2 * fut_len - 1] = 1.0
        self.dynamics_A = single_modal_A.clone()
        return instances
    
    def forward(self, agt_instances, observations, observation_confs, indexes, r_cov=None):
        batch_size = len(indexes)
        device = agt_instances.fut_trajs.device

        for batch_idx in range(batch_size):
            if len(indexes[batch_idx]) == 0:
                continue
            
            # init the states of the filters with first observation
            init_mask = agt_instances.mm_initialized[batch_idx, indexes[batch_idx]]
            idxes = torch.tensor(indexes[batch_idx])[init_mask.cpu()]
            init_idxes = torch.tensor(indexes[batch_idx])[~init_mask.cpu()]

            agt_instances.mm_agt_feats[batch_idx][init_idxes] = agt_instances.agt_feats[batch_idx][init_idxes].clone()
            agt_instances.mm_belief_mean[batch_idx][init_idxes] = observations[batch_idx][~init_mask]
            agt_instances.mm_belief_conf[batch_idx][init_idxes] = observation_confs[batch_idx][~init_mask]
            agt_instances.mm_belief_cov[batch_idx][init_idxes] += (self.observation_init + r_cov[batch_idx][~init_mask])[:, :, None, None] * torch.eye(self.state_dim)[None, None, :, :].to(device)
            agt_instances.mm_initialized[batch_idx][init_idxes] = True

            agt_num = len(idxes)
            if agt_num == 0:
                continue

            sample_observations = observations[batch_idx][init_mask] # N * K * (T * 2)
            sample_confs = observation_confs[batch_idx][init_mask] # N * K

            mm_means = agt_instances.mm_belief_mean[batch_idx][idxes] # N * K * (T * 2)
            mm_covs = agt_instances.mm_belief_cov[batch_idx][idxes]
            mm_confs = agt_instances.mm_belief_conf[batch_idx][idxes] # N * K

            dynamics_covariance = torch.zeros_like(mm_covs) + self.process_init * torch.eye(self.state_dim)[None, None, :, :].to(device)
            mm_means, mm_covs = self.predict_step(mm_means.clone(), mm_covs.clone(), dynamics_covariance)

            # observation_covariance = torch.zeros_like(mm_covs) + (self.observation_init) * torch.eye(self.state_dim)[None, None, :, :].to(device)
            observation_covariance = torch.zeros_like(mm_covs) + (self.observation_init+ r_cov[batch_idx][init_mask])[:, :, None, None] * torch.eye(self.state_dim)[None, None, :, :].to(device)
            mm_means, mm_confs, mm_covs = self.update_step(
                mm_means, mm_confs, mm_covs,
                sample_observations, sample_confs, observation_covariance)

            agt_instances.mm_belief_mean[batch_idx][idxes] = mm_means.clone()
            agt_instances.mm_belief_conf[batch_idx][idxes] = mm_confs.detach().clone()
            agt_instances.mm_belief_cov[batch_idx][idxes] = mm_covs.clone()
        return agt_instances
    
    def predict_step(self, belief_mean, belief_covariance=None, dynamics_covariance=None):
        device = belief_mean.device
        dynamics_A = self.dynamics_A
        # dynamics_A = torch.eye(self.K * self.state_dim).to(device)
        belief_mean = (dynamics_A @ belief_mean.transpose(-1, -2)).transpose(-1, -2).contiguous()
        belief_covariance = (dynamics_A @ belief_covariance @ dynamics_A.transpose(-1, -2) + dynamics_covariance).contiguous()
        return belief_mean, belief_covariance
    
    def update_step(self, belief_mean, belief_confs, belief_covariance, observations, observation_confs, observation_covariance):
        innovation = observations - belief_mean
        innovation_confs = observation_confs - belief_confs
        innovation_covariance = belief_covariance + observation_covariance
        kalman_gain = belief_covariance @ torch.inverse(innovation_covariance)

        corrected_belief_mean = belief_mean + (kalman_gain @ innovation[:, :, :, None]).squeeze(-1)
        corrected_belief_conf = belief_confs + (kalman_gain[:, :, 0, 0] * innovation_confs)

        identity = torch.eye(self.state_dim, device=kalman_gain.device)[None, None, :, :]
        corrected_belief_covariance = (identity - kalman_gain) @ belief_covariance
        return corrected_belief_mean, corrected_belief_conf, corrected_belief_covariance