""" Streaming Vectornet
    Follow how AVProphet implements VectorNet
"""
import numpy as np, copy, os, math
from typing import List, Dict
from ...streaming.core import Track, Position, Instances

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...forecaster.models.encoder import GlobalAttentionGraph, SubgraphMLP
from ...forecaster.models.batching import (batch_actors, batch_lanes)
from ...forecaster.models.head import RegressionDecoder, LinearResidualBlock
from ...forecaster.dataset import ref_copy, collate_fn

from .df import DiffKF, MultiDiffKF

from .loss import JointLoss


class StreamingVectorNet(nn.Module):
    def __init__(self, config):
        super(StreamingVectorNet, self).__init__()
        self.config = config
        predictor_config = config['predictor']
        model_config = config['predictor']['model']
        self.model_config = model_config
        self.num_mod = model_config['num_mods']
        self.hist_len = model_config['num_hist']
        self.fut_len = model_config['num_preds']
        self.occ_steps = self.model_config['num_preds']
        self.embed_dim = 256

        # streaming parts
        self.max_agt_num = model_config['streaming']['max_agt_num']
        self.max_batch_size = model_config['streaming']['max_batch_size']
        self.use_prev_traj = model_config['streaming']['use_prev_traj']
        
        # VectorNet parts
        self.pred_range = model_config['pred_range']
        self.actor_subgraphs = SubgraphMLP(model_config, 6)
        self.lane_subgraphs = SubgraphMLP(model_config, 7)
        self.globalgraphs = GlobalAttentionGraph(model_config)
        
        # decoder
        self.decoder = RegressionDecoder(model_config)
        
        # loss function
        self.loss_func = JointLoss(model_config)
        
        # module for converting trajectories into initial features
        # self.data_api = build_data_api('VectorNet', predictor_config['data'])

        # streaming inference logics
        self.instances = None
        self.cur_seq = ''
        self.cur_frame_index = 0

        # Use differentiable filters
        self.use_df = model_config['streaming']['use_df']
        if self.use_df:
            # DF for single-model trajectories
            self.df = DiffKF(1, self.fut_len * 2)
            self.df_r_net = nn.Sequential(
                LinearResidualBlock(256, 256),
                nn.Linear(256, 1))
            self.df_r_net.apply(init_weights)
            # DF for multi-modal trajectories
            self.multi_df = MultiDiffKF(self.num_mod, self.fut_len * 2)
            self.multi_df_r_net = nn.Sequential(
                LinearResidualBlock(256, 256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1))
            self.multi_df_r_net.apply(init_weights)
        return

    def generate_empty_instances(self, device, inference=False):
        instances = Instances((1, 1))
        if inference:
            max_batch_size = 1
        else:
            max_batch_size = self.max_batch_size
        instances.agt_feats = torch.full((max_batch_size, self.max_agt_num, self.embed_dim), 0, dtype=torch.float32, device=device)
        instances.hist_trajs = torch.full((max_batch_size, self.max_agt_num, self.hist_len, 2), 0, dtype=torch.float32, device=device)
        instances.fut_trajs = torch.full((max_batch_size, self.max_agt_num, 6, self.fut_len, 2), 0, dtype=torch.float32, device=device)
        instances.fut_scores = torch.full((max_batch_size, self.max_agt_num, 6), 0, dtype=torch.float32, device=device)
        instances.occ_trajs = torch.full((max_batch_size, self.max_agt_num, self.occ_steps, 2), 0, dtype=torch.float32, device=device)
        instances.occ_len = torch.full((max_batch_size, self.max_agt_num), self.occ_steps, dtype=torch.long, device=device)

        if self.use_df:
            instances = self.generate_df_instances(instances)
        
        # store the information of agents of the instances
        self.track_ids = list(list() for _ in range(self.max_batch_size))
        return instances
    
    def generate_df_instances(self, instances):
        instances = self.df.initialize_beliefs(instances)
        instances = self.multi_df.initialize_beliefs(instances)
        return instances
    
    def forward(self, data, device='cuda:0', compute_loss=True):
        if compute_loss:
           all_preds, loss_dict = self.forward_train(data, device, compute_loss)
           return all_preds, loss_dict
        else:
            all_preds = self.forward_inference(data, device)
            return all_preds, None
    
    def forward_train(self, data, device='cuda:0', compute_loss=True):
        """ Sequentially run the model over every frame in a batch-first manner.
        Args: 
            data: Data is a nested dictionary, with the order of batch_size --> frame_num --> keys
        """
        frame_num = len(data[0])
        batch_size = len(data)

        all_preds = list()
        loss_dict = dict()

        agt_instances = self.generate_empty_instances(device, inference=False)
        for frame_index_in_sample in range(frame_num):
            frame_data = [data[i][frame_index_in_sample] for i in range(batch_size)]
            # ========== Collate the fields of hdmaps, trajectories ========== #
            frame_data = collate_fn(frame_data)
            # ========== Load the trajectori data and forward on every frame ========== #
            agt_instances = self.load_traj_into_instances(agt_instances, frame_data, observation_only=(frame_index_in_sample != 0) and self.use_prev_traj)
            preds, loss_dict = self.forward_single_frame(agt_instances, frame_data, loss_dict, 
                                                         frame_index_in_sample, 
                                                         device, compute_loss)
            all_preds.append(preds)
        if compute_loss:
            loss_dict = self.loss(loss_dict, frame_num)
        return all_preds, loss_dict
    
    def forward_inference(self, data, device='cuda:0'):
        """ Run on a single frame
        """
        frame_num = len(data[0])
        batch_size = len(data)

        all_preds = list()
        loss_dict = dict()

        seq_name = data[0][0]['seq_name']
        if seq_name != self.cur_seq:
            self.cur_seq = seq_name
            self.cur_frame_index = 0
            self.instances = self.generate_empty_instances(device, inference=True)

        for frame_index_in_sample in range(frame_num):
            frame_data = [data[i][frame_index_in_sample] for i in range(batch_size)]
            # ========== Collate the fields of hdmaps, trajectories ========== #
            frame_data = collate_fn(frame_data)
            # ========== Load the trajectori data and forward on every frame ========== #
            self.instances = self.load_traj_into_instances(self.instances, frame_data, observation_only=(self.cur_frame_index > 0) and self.use_prev_traj)
            preds, loss_dict = self.forward_single_frame(self.instances, frame_data, loss_dict, 
                                                         frame_index_in_sample, 
                                                         device, compute_loss=False)
            all_preds.append(preds)
            self.cur_frame_index += 1
        return all_preds
    
    def encoding(self, agt_instances: Instances, data, loss_dict, frame_index, 
                device='cuda:0', compute_loss=True):
        # ========== Generate the agent features ========== #
        # enumerate over the batch
        # agent_feats: a collated dictionary {key0: [batch0, batch1, ...], key1: ...}
        # query_keys: [keys0, keys1, ...]
        # prior_fut_trajs: [trajs0, trajs1, ...] the future trajectories generated from some simple priors, such as Kalman filter
        agent_feats, query_keys, prior_fut_trajs = self.extract_traj_feats(agt_instances, data)

        # load into the data
        for key in agent_feats.keys():
            data[key] = ref_copy(agent_feats[key])
        
        # ========== Batching the actor and lane information ========== #
        actors, actor_nums, actor_idcs = batch_actors(data['actors'], device)
        lanes, lane_ctrs, lane_nums = batch_lanes(data['graph'], device)
        actor_ctrs = [a.to(device, non_blocking=True) for a in data['ctrs']]
        batch_size = len(actor_ctrs)

        # ========== Actor and lane subgraph features ========== #
        actors = self.actor_subgraphs(actors) # batch_size * max_actor_num * hidden_dim
        lanes = self.lane_subgraphs(lanes) # batch_size * max_lane_num * hidden_dim

        # ========== Global interaction ========== #
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
        return actor_feats, actor_idcs, actor_ctrs, query_keys, prior_fut_trajs
    
    def decoding(self, actor_feats, actor_idcs, actor_ctrs, data, device='cuda:0'):
        # ========== Decode the trajectories ========== #
        output = self.decoder(actor_feats, actor_idcs, actor_ctrs)

        # ========== Transform to the global coordinate ========== #
        norm_rot, norm_center = data['norm_rot'], data['norm_center']
        batch_size = len(output['prediction'])
        for i in range(batch_size):
            world_coord_preds = torch.matmul(output['prediction'][i], norm_rot[i].to(device))
            world_coord_preds += norm_center[i].to(device).view(1, 1, 1, -1)
            output['prediction'][i] = world_coord_preds

            single_preds = torch.matmul(output['single_prediction'][i], norm_rot[i].to(device))
            single_preds += norm_center[i].to(device).view(1, 1, -1)
            output['single_prediction'][i] = single_preds
        
        output['indexes'] = [None for _ in range(batch_size) ]
        for i in range(batch_size):
            output['indexes'][i] = data['indexes'][i]
        return output
    
    def forward_single_frame(self, agt_instances: Instances, data, loss_dict, frame_index, 
                             device='cuda:0', compute_loss=True):
        """ Run the model on a single frame.
        """
        # ========== Extract agent features ========== #
        actor_feats, actor_idcs, actor_ctrs, query_keys, prior_fut_trajs = \
            self.encoding(agt_instances, data, loss_dict, frame_index, device, compute_loss)

        # ========== Decode trajectories in the world coordinate ========== #
        output = self.decoding(actor_feats, actor_idcs, actor_ctrs, data, device)
        indexes = [x.detach().cpu().numpy() for x in output['indexes']]

        # ========== Process with differentiable filtering ========== #
        if self.use_df:
            batch_size = len(output['prediction'])
            agt_instances_indexes = [torch.tensor([self.track_ids[b].index(k) for k in data['query_keys'][b]]) for b in range(batch_size)]
    
            # save the agent features
            for i in range(batch_size):
                idcs = actor_idcs[i]
                feats = actor_feats[idcs]
                idxes = agt_instances_indexes[i]
                agt_instances.agt_feats[i, idxes] = feats.detach().clone()

        if self.use_df:
            r = self.df_r_net(actor_feats) ** 2
            r = [r[idcs] for idcs in actor_idcs]

            multi_r = self.multi_df_r_net(actor_feats) ** 2
            multi_r = [multi_r[idcs] for idcs in actor_idcs]
        else:
            r = None
            multi_r = None
        
        # ========== Save the trajectories back to the instances ========== #
        agt_instances = self.load_trajs(output['prediction'], output['confidence'],
                                        agt_instances, data['query_keys'],
                                        data['update_keys'], r_cov=multi_r)
        trajs = self.restore_prediction_trajs(agt_instances, agt_instances_indexes)
        output['prediction'] = trajs

        occ_predictions = list()
        for batch_idx in range(batch_size):
            sample_predictions = output['prediction'][batch_idx]
            sample_confs = output['confidence'][batch_idx]
            cls_scores, cls_idcs = sample_confs.sort(dim=1, descending=True)
            cls_max_idcs = cls_idcs[:, 0]
            row_idcs = torch.arange(len(cls_max_idcs)).long().to(cls_max_idcs.device)
            occ_predictions.append(sample_predictions[row_idcs, cls_max_idcs])
        
        agt_instances = self.load_occ_trajs(occ_predictions, agt_instances, data['query_keys'], data['update_keys'])
        occ_trajs = self.restore_occ_prediction_trajs(agt_instances, agt_instances_indexes)

        # ========== Aggregate the results ========== #
        preds = [x for x in output['prediction']]
        preds = torch.cat(preds, dim=0)
        confidences = [x for x in output['confidence']]
        confidences = torch.cat(confidences, dim=0)
        indexes = [x.detach().cpu().numpy() for x in output['indexes']]

        # ========== Compute the loss =========== #
        if compute_loss:
            loss_dict = self.loss_single_frame(loss_dict, data, query_keys,
                                               output, indexes, frame_index)

        # ========== Return the results ========== #
        output_result = dict()
        cnt = 0
        for b in range(len(indexes)):
            for i, track_id in enumerate(query_keys[b]):
                if i not in indexes[b]:
                    output_result[track_id] = {
                        'trajs': prior_fut_trajs[b][i],
                        'confidences': np.ones(6),
                    }
                    continue
                output_result[track_id] = {
                    'trajs': preds[cnt].detach().cpu().numpy(),
                    'confidences': confidences[cnt].detach().cpu().numpy()
                }
                cnt += 1
        return output_result, loss_dict

    def load_traj_into_instances(self, agt_instances, data, observation_only=True):
        """ Load trajectory data into the agent instances
            observation_only: except for the first frame, all the subsequent frames only load the observed states 
        """
        field_keys = list(data.keys())
        batch_size = len(data[field_keys[0]])
        # shift the future trajectories into the history trajectories
        # shift on the time axis (dim=-2)
        agt_instances.hist_trajs = torch.cat((agt_instances.hist_trajs[:, :, 1:, :], 
                                              agt_instances.fut_trajs[:, :, 0, :1, :] + agt_instances.hist_trajs[:, :, -1:, :]), 
                                              dim=-2)
        agt_instances.fut_trajs = torch.cat((agt_instances.fut_trajs[:, :, :, 1:, :], 
                                             agt_instances.fut_trajs[:, :, :, -1:, :]),
                                             dim=-2) 
        
        # load new data
        data['update_keys'] = list()
        for batch_idx in range(batch_size):
            # query keys
            observation_keys = data['observation_keys'][batch_idx]
            query_keys = data['query_keys'][batch_idx]
            ego_key = [k for k in query_keys if 'ego' in k][0]
            query_keys.remove(ego_key)
            query_keys.insert(0, ego_key)

            # load the trajectory data
            instance_indexes = list()
            device = data['hist_kf_trajs'][batch_idx][ego_key].device
            related_keys = list()
            # some objects may return to view
            for track_id in query_keys:
                if observation_only:
                    if (track_id not in observation_keys) and (self.key_to_index(batch_idx, track_id, query_only=True) < 0):
                        related_keys.append(track_id)
                    elif track_id in observation_keys:
                        related_keys.append(track_id)
                    else:
                        # print(batch_idx, track_id)
                        pass
                else:
                    related_keys.append(track_id)
            data['update_keys'].append(related_keys)

            hist_trajs = torch.full((len(related_keys), self.hist_len, 2), 0, dtype=torch.float32, device=device)
            for i, track_id in enumerate(related_keys):
                instance_indexes.append(self.key_to_index(batch_idx, track_id, query_only=False))
                trajs = data['hist_kf_trajs'][batch_idx][track_id].float()
                steps = data['hist_kf_steps'][batch_idx][track_id].long()
                hist_trajs[i][steps] = trajs
            agt_instances.hist_trajs[batch_idx, instance_indexes] = hist_trajs.clone().to(agt_instances.hist_trajs.device)
        return agt_instances
    
    def key_to_index(self, batch_idx, track_id, query_only=True):
        """ return the index of the instances
        """
        if track_id not in self.track_ids[batch_idx] and (not query_only):
            self.track_ids[batch_idx].append(track_id)
        if track_id in self.track_ids[batch_idx]:
            return self.track_ids[batch_idx].index(track_id)
        else:
            return -1
    
    def extract_traj_feats(self, agt_instances, data):
        field_keys = list(data.keys())
        batch_size = len(data[field_keys[0]])
        agt_feats, query_keys, prior_fut_trajs = list(), list(), list()
        for i in range(batch_size):
            feat, keys, fut_trajs = self.extract_traj_feats_from_sample(agt_instances, data, i)
            agt_feats.append(feat)
            query_keys.append(keys)
            prior_fut_trajs.append(fut_trajs)
        agt_feats = collate_fn(agt_feats)
        return agt_feats, query_keys, prior_fut_trajs
    
    def extract_traj_feats_from_sample(self, agt_instances, data, batch_idx):
        all_timestamps = data['hist_ts'][batch_idx]

        # put ego as the first agent, for normalization convenience
        query_keys = data['query_keys'][batch_idx]
        ego_key = [k for k in query_keys if 'ego' in k][0]
        query_keys.remove(ego_key)
        query_keys.insert(0, ego_key)

        # prepare the trajectory data for inference
        # hallucinate the missing observations using a Kalman filter
        all_trajs, all_steps, all_ts, all_prior_fut_trajs = list(), list(), list(), list()
        frame_index, full_trajs = data['frame_index'][batch_idx], data['trajs'][batch_idx]
        for track_id in query_keys:
            index = self.key_to_index(batch_idx, track_id)
            steps = data['hist_kf_steps'][batch_idx][track_id]
            trajs = agt_instances.hist_trajs[batch_idx][index][steps.long()]
            # trajs = data['hist_kf_trajs'][batch_idx][track_id]
            prior_fut_trajs = data['fut_kf_trajs'][batch_idx][track_id]
            ts = all_timestamps[steps.long()]

            all_trajs.append(trajs)
            all_steps.append(steps)
            all_ts.append(ts)
            all_prior_fut_trajs.append(prior_fut_trajs)
        
        # extract the features of the trajectories
        raw_traj_data = {
            'idx': -1, 'city': data['city_name'][batch_idx], 'trajs': all_trajs,
            'steps': all_steps, 'timestamps': all_ts,
            'frame_num': all_trajs[0].shape[0]
        }
        agt_feats = self.get_obj_feats(raw_traj_data)
        agt_feats['query_keys'] = query_keys
        return agt_feats, query_keys, all_prior_fut_trajs
    
    def loss(self, loss_dict, frame_num):
        for field in ['loss_pred', 'loss_conf', 'loss_single', 'loss']:
            sum_loss = 0
            for i in range(frame_num):
                sum_loss += loss_dict[f'f{i}.{field}']
            sum_loss /= frame_num
            loss_dict[field] = sum_loss
        return loss_dict

    def loss_single_frame(self, loss_dict, data, query_keys, output, indexes, frame_index):
         # ========== Acquire the ground truth ========== #
        raw_fut_trajs, raw_fut_masks = data['fut_trajs'], data['fut_masks']
        batch_size = len(raw_fut_trajs)
        fut_trajs, fut_masks = list(), list()
        device = output['prediction'][0].device
        for b in range(batch_size):
            keys = [query_keys[b][i] for i in indexes[b]]
            
            sample_fut_trajs = [raw_fut_trajs[b][k][None, ...] for k in keys]
            sample_fut_trajs = torch.cat(sample_fut_trajs, dim=0)
            fut_trajs.append(sample_fut_trajs.float()[..., :2])

            sample_fut_masks = [raw_fut_masks[b][k][None, ...] for k in keys]
            sample_fut_masks = torch.cat(sample_fut_masks, dim=0)
            fut_masks.append(sample_fut_masks.bool())

        # ========== Compute the loss ========== #
        losses = self.loss_func(
            out=output,
            gt_futures=fut_trajs,
            gt_future_masks=fut_masks,
            device=device)
        
        # ========== Modify the loss dictionary ========== #
        loss_dict[f'f{frame_index}.loss_pred'] = losses['pred']
        loss_dict[f'f{frame_index}.loss_conf'] = losses['conf']
        loss_dict[f'f{frame_index}.loss_single'] = losses['single_pred']
        loss_dict[f'f{frame_index}.loss'] = losses['loss']
        return loss_dict
    
    def compute_prediction_movements(self, predictions, agt_instances, indexes):
        movements = list()
        for i, idxes in enumerate(indexes):
            preds = predictions[i].detach().clone() # agt_num * K * T * 2
            hist_pos = agt_instances.hist_trajs[i, idxes, -1] # agt_num * 2
            normalized_predictions = preds - hist_pos[:, None, None, :]
            normalized_predictions[:, :, 1:] = normalized_predictions[:, :, 1:] - normalized_predictions[:, :, :-1]
            movements.append(normalized_predictions)
        return movements
    
    def compute_occ_prediction_movements(self, occ_predictions, agt_instances, indexes):
        movements = list()
        for i, idxes in enumerate(indexes):
            preds = occ_predictions[i].detach().clone() # agt_num * T * 2
            hist_pos = agt_instances.hist_trajs[i, idxes, -1] # agt_num * 2
            normalized_predictions = preds - hist_pos[:, None, :]
            normalized_predictions[:, 1:] = normalized_predictions[:, 1:] - normalized_predictions[:, :-1]
            movements.append(normalized_predictions)
        return movements
    
    def restore_prediction_trajs(self, agt_instances, indexes):
        trajs = list()
        for i, idxes in enumerate(indexes):
            preds = agt_instances.fut_trajs[i, idxes] # N * K * T * 2
            hist_pos = agt_instances.hist_trajs[i, idxes, -1] # N * 2
            normalized_trajs = torch.cumsum(preds, dim=-2)
            restored_trajs = normalized_trajs + hist_pos[:, None, None, :]
            trajs.append(restored_trajs)
        return trajs
    
    def restore_occ_prediction_trajs(self, agt_instances, indexes):
        trajs = list()
        for i, idxes in enumerate(indexes):
            preds = agt_instances.occ_trajs[i, idxes] # agt_num * T * 2
            hist_pos = agt_instances.hist_trajs[i, idxes, -1] # agt_num * 2
            normalized_trajs = torch.cumsum(preds, dim=-2)
            occ_trajs = normalized_trajs + hist_pos[:, None, :]
            trajs.append(occ_trajs)
        return trajs
    
    def load_trajs(self, predictions, scores, agt_instances, query_keys, update_keys, r_cov):
        # use r_cov only when df is used
        batch_size = len(query_keys)
        indexes = [[self.track_ids[b].index(k) for k in query_keys[b]] for b in range(batch_size)]
        movements = self.compute_prediction_movements(predictions, agt_instances, indexes)

        if not self.use_df:
            for i in range(batch_size):
                agt_instances.fut_trajs[i, indexes[i]] = movements[i]
        else:
            observations = [movements[i].detach().clone().reshape(-1, self.num_mod, self.fut_len * 2) for i in range(batch_size)]
            observation_confs = [scores[i].detach().clone() for i in range(batch_size)]
            agt_instances = self.multi_df(agt_instances, observations, observation_confs, indexes, r_cov=r_cov)
            for i in range(batch_size):
                agt_instances.fut_trajs[i, indexes[i]] = agt_instances.mm_belief_mean[i, indexes[i]].clone().reshape(-1, self.num_mod, self.fut_len, 2)
                agt_instances.fut_scores[i, indexes[i]] = agt_instances.mm_belief_conf[i, indexes[i]].clone()
        return agt_instances
    
    def load_occ_trajs(self, occ_predictions, agt_instances, query_keys, update_keys):
        batch_size = len(query_keys)
        indexes = [torch.tensor([self.track_ids[b].index(k) for k in query_keys[b]]) for b in range(batch_size)]
        occluded = [torch.tensor([k not in update_keys[b] for k in query_keys[b]]) for b in range(batch_size)]
        device = agt_instances.occ_trajs.device
        occ_movements = self.compute_occ_prediction_movements(occ_predictions, agt_instances, indexes)
        
        for i in range(batch_size):
            agt_instances.occ_trajs[i, indexes[i]] = occ_movements[i]
        return agt_instances
    
    def load_pretrain(self, pretrain_dict):
        state_dict = self.state_dict()
        for key in pretrain_dict.keys():
            if key in state_dict and (pretrain_dict[key].size() == state_dict[key].size()):
                value = pretrain_dict[key]
                if not isinstance(value, torch.Tensor):
                    value = value.data
                state_dict[key] = value
        self.load_state_dict(state_dict)
        return
    
    def get_obj_feats(self, data, hist_len=20, fut_len=30, gt=False):
        pred_range = self.pred_range
        frame_num = data['frame_num']
        future_num = fut_len
        orig = data['trajs'][0][frame_num - future_num - 1].clone().float()
        pre = data['trajs'][0][frame_num - future_num - 2].float() - orig
        device = orig.device
    
        theta = math.pi - np.arctan2(pre[1].detach().cpu().numpy(), pre[0].detach().cpu().numpy())
        rot = torch.Tensor([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]]).to(device)
    
        feats, ctrs, gt_preds, has_preds, valid_agt_idx = [], [], [], [], []
        gt_preds = torch.full((len(data['trajs']), fut_len, 2), 0, dtype=torch.float32, device=device)
        gt_preds = torch.full((len(data['trajs']), fut_len), 0, dtype=torch.bool, device=device)
        feats = torch.full((len(data['trajs']), hist_len, 6), 0, dtype=torch.float32, device=device)
        ctrs = torch.full((len(data['trajs']), 2), 0, dtype=torch.float32, device=device)
        for index, (traj, step) in enumerate(zip(data['trajs'], data['steps'])):
            if gt:
                future_mask = torch.logical_and(step >= 20, step < 50)
                post_step = step[future_mask] - 20
                post_traj = traj[future_mask]
                gt_preds[index][post_step] = post_traj
                has_preds[index][post_step] = 1
            
            obs_mask = step < hist_len
            step = step[obs_mask].float()
            traj = traj[obs_mask].float()
            ts = data['timestamps'][index][obs_mask].to(device)
            idcs = step.argsort()
    
            step = step[idcs]
            traj = traj[idcs]
            ts = ts[idcs]
    
            for i in range(len(step)):
                if step[i] == hist_len - len(step) + i:
                    break
    
            step = step[i:].long()
            traj = traj[i:]
            ts = ts[i:].float()
            
            feats[index, step, :2] = torch.matmul(rot, (traj - orig[None, :]).T).T # centers
            feats[index, 1:, 2:4] = feats[index, 1:, :2] - feats[index, :-1, :2] # motions
            feats[index, :, 4] = 1.0 # mask
            feats[index, step, 5] = ts # timstamp
            ctrs[index, :] = feats[index, -1, :2].clone()

            # out of the prediction range
            # x_min, x_max, y_min, y_max = pred_range
            # if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
            #     continue
            valid_agt_idx.append(index)
    
        data['indexes'] = torch.tensor(valid_agt_idx).long().to(device)
        data['actors'] = feats[:, 1:, :]
        data['ctrs'] = ctrs
        data['norm_center'] = orig
        data['theta'] = theta
        data['norm_rot'] = rot
        data['gt_futures'] = gt_preds
        data['gt_future_masks'] = has_preds
        return data


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.uniform_(m.bias)