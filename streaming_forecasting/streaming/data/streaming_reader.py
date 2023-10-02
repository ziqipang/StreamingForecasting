""" Loading and preprocessing the data per-frame
"""
import os, pickle, numpy as np, copy
import torch
from ..core import Track, TrackBank, Position
from torch.utils.data import Dataset
from ..utils.kf_prior import KFPrior
from ...forecaster.api import api_builder
from ...forecaster.dataset import collate_fn, from_numpy


class StreamingReader(Dataset):
    def __init__(self, configs,
                 data_dir, benchmark_file, infos_file, hdmap_dir=None,
                 frames_per_sample=1,
                 hist_len=20, fut_len=30, trainval=True, interp=False, **kwargs) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.benchmark_file = pickle.load(open(benchmark_file, 'rb'))
        self.infos_file = pickle.load(open(infos_file, 'rb'))
        self.frames_per_sample = frames_per_sample
        self.hdmap_dir = hdmap_dir
        
        self.timestamps = self.infos_file['timestamps']
        self.trajs = self.infos_file['trajectories']
        self.samples = self.infos_file['samples']

        self.hist_len = hist_len
        self.fut_len = fut_len
        self.trainval = trainval
        self.interp = False

        # use Kalman filter as a default policy for unobserved states
        self.prior_module = KFPrior()

        # build up forecaster API for data loading
        self.forecaster_api = api_builder(configs['forecasting']['model_name'], 
                                          configs['predictor']['data'], configs['predictor']['model'], 
                                          configs['forecasting']['model_weight_path'], 
                                          'cuda:0',
                                          load_weight=False)

        # fields for loading data
        self.cur_seq_name = ''

        # remove the invalid indexes for multiple frames per sample
        self._valid_frame_indexes()

    def _generate_streaming_data_indexes(self, index):
        """ Generate the frame indexes for every sample
        """
        index_list = [i for i in range(index - self.frames_per_sample + 1, index + 1)]
        seq_names = [self.samples[i]['seq_name'] for i in index_list]
        tgt_seq_name = seq_names[-1]
        for i in range(self.frames_per_sample)[::-1]:
            if seq_names[i] != tgt_seq_name:
                return None
        return index_list
    
    def _valid_frame_indexes(self):
        """ Return the indexes valid for multi-frame samples
        """
        self.valid_indexes = list()
        for i, _ in enumerate(self.samples):
            if self._generate_streaming_data_indexes(i) is not None:
                self.valid_indexes.append(i)
        return self.valid_indexes
    
    def __len__(self):
        return len(self.valid_indexes)
    
    def __getitem__(self, dataset_index):
        """ Load the samples during data loading
        """
        index = self.valid_indexes[dataset_index]
        frame_indexes = self._generate_streaming_data_indexes(index)

        data_queue = list()
        for frame_index in frame_indexes:
            frame_data = self.prepare_single_frame(frame_index)
            data_queue.append(frame_data)
        return data_queue
    
    def prepare_single_frame(self, index):
        """ Prepare the data for a single frame
        """
        sample = self.samples[index]
        query_keys = sample['query_keys']
        
        # save the information
        result = copy.deepcopy(sample)

        # read the trajectory information
        seq_name = sample['seq_name']
        frame_index = sample['frame_index']
        
        result['hist_ts'] = np.array(self.timestamps[seq_name][frame_index - self.hist_len + 1: frame_index + 1]) / 1e9
        result['hist_ts'] -= result['hist_ts'][0]
        result['fut_ts'] = np.array(self.timestamps[seq_name][frame_index + 1: frame_index + 1 + self.fut_len]) / 1e9
        result['fut_ts'] -= result['hist_ts'][0]

        hist_trajs, fut_trajs, hist_masks, fut_masks = dict(), dict(), dict(), dict()
        hist_kf_trajs, hist_kf_steps, fut_kf_trajs = dict(), dict(), dict()
        for key in query_keys:
            traj, mask = self.trajs[key]['traj'], self.trajs[key]['mask']
            hist_trajs[key] = traj[frame_index - self.hist_len + 1: frame_index + 1]
            fut_trajs[key] = traj[frame_index + 1: frame_index + 1 + self.fut_len]
            hist_masks[key] = mask[frame_index - self.hist_len + 1: frame_index + 1]
            fut_masks[key] = mask[frame_index + 1: frame_index + 1 + self.fut_len]
            
            # by default, use kalman filters to hallucinate the unobserved positions
            track = Track(key, self.trajs[key]['obj_type'], 
                          Position(np.zeros(3)), 0)
            track.traj = self.trajs[key]['traj']
            track.mask = self.trajs[key]['mask']
            track.frames = self.trajs[key]['frames']
            trajs, steps, prior_fut_trajs = self.prior_module.estimate(
                track,
                frame_index - self.hist_len + 1,
                hist_len=self.hist_len,
                fut_len=self.fut_len)

            hist_kf_trajs[key] = trajs
            hist_kf_steps[key] = steps
            fut_kf_trajs[key] = prior_fut_trajs

        full_trajs = dict()
        for key in query_keys:
            full_trajs[key] = self.trajs[key]
        
        # connect with forecaster data loaders
        # load HD-Map features in the dataloader
        ego_key = [k for k in query_keys if 'ego' in k][0]
        ego_traj = [self.trajs[ego_key]['traj'][frame_index - self.hist_len + 1: frame_index + 1, :2]]
        ego_steps = [np.arange(self.hist_len)]
        ego_ts = [result['hist_ts'][ego_steps[0].astype(np.int32)]]
        
        # use forecaster api to convert the data formats that are suitable for direct inference
        hdmap_data = {
            'idx': -1, 'city': result['city_name'], 'trajs': ego_traj,
            'steps': ego_steps, 'timestamps': ego_ts,
            'frame_num': ego_traj[0].shape[0]
        }

        # load the hdmap features with forecaster API
        preprocessed_map_path = None
        if self.hdmap_dir is not None:
            preprocessed_map_path = os.path.join(self.hdmap_dir, f'{index}.pkl')
        network_data = self.forecaster_api.load_data(hdmap_data, preprocessed_map_path=preprocessed_map_path)
        network_data = network_data
        result['graph'] = network_data['graph']

        # save trajectory information
        result['seq_name'] = seq_name
        result['frame_index'] = frame_index
        result['hist_trajs'] = hist_trajs
        result['fut_trajs'] = fut_trajs
        result['hist_masks'] = hist_masks
        result['fut_masks'] = fut_masks

        result['hist_kf_trajs'] = hist_kf_trajs
        result['hist_kf_steps'] = hist_kf_steps
        result['fut_kf_trajs'] = fut_kf_trajs

        # save all trajectories in case of need
        result['trajs'] = full_trajs
        return result
    
    def preprocess_map(self, index, hdmap_dir):
        sample = self.samples[index]
        query_keys = sample['query_keys']
        
        # save the information
        result = copy.deepcopy(sample)

        # read the trajectory information
        seq_name = sample['seq_name']
        frame_index = sample['frame_index']
        
        result['hist_ts'] = np.array(self.timestamps[seq_name][frame_index - self.hist_len + 1: frame_index + 1]) / 1e9
        result['hist_ts'] -= result['hist_ts'][0]
        result['fut_ts'] = np.array(self.timestamps[seq_name][frame_index + 1: frame_index + 1 + self.fut_len]) / 1e9
        result['fut_ts'] -= result['hist_ts'][0]

        hist_trajs, fut_trajs, hist_masks, fut_masks = dict(), dict(), dict(), dict()
        for key in query_keys:
            traj, mask = self.trajs[key]['traj'], self.trajs[key]['mask']
            hist_trajs[key] = traj[frame_index - self.hist_len + 1: frame_index + 1]
            fut_trajs[key] = traj[frame_index + 1: frame_index + 1 + self.fut_len]
            hist_masks[key] = mask[frame_index - self.hist_len + 1: frame_index + 1]
            fut_masks[key] = mask[frame_index + 1: frame_index + 1 + self.fut_len]
        
        full_trajs = dict()
        for key in query_keys:
            full_trajs[key] = self.trajs[key]
        
        # connect with forecaster data loaders
        # load HD-Map features in the dataloader
        ego_key = [k for k in query_keys if 'ego' in k][0]
        ego_traj = [self.trajs[ego_key]['traj'][frame_index - self.hist_len + 1: frame_index + 1, :2]]
        ego_steps = [np.arange(self.hist_len)]
        ego_ts = [result['hist_ts'][ego_steps[0].astype(np.int32)]]
        
        # use forecaster api to convert the data formats that are suitable for direct inference
        hdmap_data = {
            'idx': -1, 'city': result['city_name'], 'trajs': ego_traj,
            'steps': ego_steps, 'timestamps': ego_ts,
            'frame_num': ego_traj[0].shape[0]
        }

        self.forecaster_api.preprocess_map(hdmap_data, os.path.join(hdmap_dir, f'{index}.pkl'))
        return


def streaming_collate_fn(batch_data):
    return batch_data