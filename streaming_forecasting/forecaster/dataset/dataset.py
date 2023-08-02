import torch
import numpy as np
from torch.utils.data import Dataset
import os
import copy
import pickle
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from .map import VecMapReader


def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data


class ArgoDataset(Dataset):
    def __init__(self, data_dir, config, map_preprocess, ratio=1.0, gt=True):
        '''
            Args:
                data_dir: path to argoverse data
                config: config content
                ratio: how much data to use, for example, 100\% ot 50\%
                gt: query ground truth information?
                map_preprocess: path to preprocessed map. if None, process map during data loading.
        '''
        super(ArgoDataset, self).__init__()
        self.dataset_type = 'Vec'
        self.data_dir = data_dir
        self.config = config
        self.ratio = ratio
        self.gt = gt
        self.map_preprocess = map_preprocess

        self.avl = ArgoverseForecastingLoader(data_dir)
        self.am = ArgoverseMap()
        self.map_reader = VecMapReader(self.am, self.config)
        self.avl.seq_list = sorted(self.avl.seq_list)
        self.num_samples = int(self.ratio * len(self.avl.seq_list))
        # print(f'Argo Dataset Prepared. Use {self.ratio:.2f} Data and {self.num_samples} samples.')
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        # load actors information
        data = self.read_argo_data(index)
        data = self.get_obj_feats(data, self.config['pred_range'], gt=self.gt)

        # load graph connection
        if self.map_preprocess is None:
            # process graph
            graph = self.map_reader.read_map_from_raw(data, data['city'])
        else:
            # load preprocessed graph
            file_path = os.path.join(self.map_preprocess, f'{index}.pkl')
            preprocessed_graph = pickle.load(open(file_path, 'rb'))
            graph = self.map_reader.read_map_from_preprocessed(preprocessed_graph, data, data['city'])
        data['graph'] = graph

        # copy the needed information
        result = dict()
        for key in ['idx', 'city', 'norm_center', 'gt_futures', 'gt_future_masks', 'theta', 'norm_rot', \
            'actors', 'ctrs', 'graph', 'indexes']:
            if key in data:
                result[key] = ref_copy(data[key])
        return result
    
    def read_argo_data(self, idx):
        city = copy.deepcopy(self.avl[idx].city)
        df = copy.deepcopy(self.avl[idx].seq_df)
        all_ts = np.array(df['TIMESTAMP'].values)
        agt_ts = np.sort(np.unique(df['TIMESTAMP'].values))

        mapping = dict()
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i

        trajs = np.concatenate((
            df.X.to_numpy().reshape(-1, 1),
            df.Y.to_numpy().reshape(-1, 1)), 1)
        
        steps = [mapping[x] for x in df['TIMESTAMP'].values]
        steps = np.asarray(steps, np.int64)

        objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
        keys = list(objs.keys())
        obj_type = [x[1] for x in keys]

        agt_idx = obj_type.index('AGENT')
        idcs = objs[keys[agt_idx]]
       
        agt_traj = trajs[idcs]
        agt_step = steps[idcs]

        frame_num = agt_traj.shape[0]

        ctx_trajs, ctx_steps, ctx_ts = [], [], []
        del keys[agt_idx]

        for key in keys:
            idcs = objs[key]
            ctx_trajs.append(trajs[idcs])
            ctx_steps.append(steps[idcs])
            ctx_ts.append(all_ts[idcs] - agt_ts[0])
        agt_ts = agt_ts - agt_ts[0]

        data = dict()
        data['idx'] = idx
        data['city'] = city
        data['trajs'] = [agt_traj] + ctx_trajs
        data['steps'] = [agt_step] + ctx_steps
        data['timestamps'] = [agt_ts] + ctx_ts
        data['frame_num'] = frame_num
        return data
    
    def get_obj_feats(self, data, pred_range, hist_len=20, fut_len=30, gt=False):
        """ Transform trajectories into features
        """
        frame_num = data['frame_num']
        future_num = fut_len
        orig = data['trajs'][0][frame_num - future_num - 1].copy().astype(np.float32)
        pre = data['trajs'][0][frame_num - future_num - 2] - orig
    
        theta = np.pi - np.arctan2(pre[1], pre[0])
        rot = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32)
    
        feats, ctrs, gt_preds, has_preds, valid_agt_idx = [], [], [], [], []
        for index, (traj, step) in enumerate(zip(data['trajs'], data['steps'])):
            if (hist_len - 1) not in step:
                continue
            if gt:
                gt_pred = np.zeros((30, 2), np.float32)
                has_pred = np.zeros(30, np.bool)
                future_mask = np.logical_and(step >= 20, step < 50)
                post_step = step[future_mask] - 20
                post_traj = traj[future_mask]
                gt_pred[post_step] = post_traj
                has_pred[post_step] = 1
            
            obs_mask = step < hist_len
            step = step[obs_mask]
            traj = traj[obs_mask]
            ts = data['timestamps'][index][obs_mask]
            idcs = step.argsort()
    
            step = step[idcs]
            traj = traj[idcs]
            ts = ts[idcs]
    
            for i in range(len(step)):
                if step[i] == hist_len - len(step) + i:
                    break
    
            step = step[i:]
            traj = traj[i:]
            ts = ts[i:]
            feat = np.zeros((hist_len, 4), np.float32)
            feat[step, :2] = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T
            feat[step, 2] = 1.0   
            feat[step, 3] = ts
            # if self.gt:
            #     gt_pred[post_step, :2] = np.matmul(rot, (gt_pred[post_step, :2] - orig.reshape(-1, 2)).T).T
            x_min, x_max, y_min, y_max = pred_range
            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue
            ctrs.append(feat[-1, :2].copy())
            
            result_feat = np.zeros((19, 6))
            result_feat[:, :2] = feat[1:, :2] # centers
            result_feat[:, 2:4] = feat[1:, :2] - feat[:-1, :2] # motions
            result_feat[:, 4] = feat[1:, 2] # mask
            result_feat[:, 5] = feat[1:, 3] # timstamp
            feats.append(result_feat)
            if gt:
                gt_preds.append(gt_pred)
                has_preds.append(has_pred)
            valid_agt_idx.append(index)
    
        feats = np.asarray(feats, np.float32)
        ctrs = np.asarray(ctrs, np.float32)
        if gt:
            gt_preds = np.asarray(gt_preds, np.float32)
            has_preds = np.asarray(has_preds, np.bool)
    
        data['indexes'] = np.array(valid_agt_idx).astype(np.int)
        data['actors'] = feats
        data['ctrs'] = ctrs
        data['norm_center'] = orig
        data['theta'] = theta
        data['norm_rot'] = rot
        data['gt_futures'] = gt_preds
        data['gt_future_masks'] = has_preds
        return data