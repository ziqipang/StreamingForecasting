"""Load a frame every time"""
import os
import pickle
import numpy as np
import torch
import copy

from ..core import Track, TrackBank, Position
from torch.utils.data import Dataset


class FrameReader(Dataset):
    def __init__(self, data_dir, benchmark_file, infos_file,
                 hist_len=20, fut_len=30, trainval=True, interp=False, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.benchmark_file = pickle.load(open(benchmark_file, 'rb'))
        self.infos_file = pickle.load(open(infos_file, 'rb'))
        
        self.timestamps = self.infos_file['timestamps']
        self.trajs = self.infos_file['trajectories']
        self.samples = self.infos_file['samples']

        self.hist_len = hist_len
        self.fut_len = fut_len
        self.trainval = trainval
        self.interp = False

        self.cur_seq_name = ''
        self.tracks = dict()
        return
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        # read the trajectory data, etc.
        # everything in the world coordinate
        # load the sample information
        sample = self.samples[index]
        query_keys, observ_keys = sample['query_keys'], sample['observation_keys']
        
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
        occ_keys = list()
        for key in query_keys:
            traj, mask = self.trajs[key]['traj'], self.trajs[key]['mask']
            hist_trajs[key] = traj[frame_index - self.hist_len + 1: frame_index + 1]
            fut_trajs[key] = traj[frame_index + 1: frame_index + 1 + self.fut_len]
            hist_masks[key] = mask[frame_index - self.hist_len + 1: frame_index + 1]
            fut_masks[key] = mask[frame_index + 1: frame_index + 1 + self.fut_len]

            if key not in observ_keys:
                occ_keys.append(key)
        
        # full tracks, maybe useful for supervision
        full_tracks = dict()
        for key in query_keys:
            full_tracks[key] = Track(key, self.trajs[key]['obj_type'], 
                                     Position(np.zeros(3)), 0)
            full_tracks[key].traj = self.trajs[key]['traj']
            full_tracks[key].mask = self.trajs[key]['mask']
            full_tracks[key].frame = self.trajs[key]['frames']
        
        # if a new sequence, reload the tracks
        seq_name = sample['seq_name']
        if seq_name != self.cur_seq_name:
            self.tracks = dict()
            for key in self.trajs.keys():
                self.tracks[key] = Track(key, self.trajs[key]['obj_type'], 
                                     Position(np.zeros(3)), 0)
                self.tracks[key].traj = self.trajs[key]['traj']
                self.tracks[key].mask = self.trajs[key]['mask']
                self.tracks[key].frames = self.trajs[key]['frames']
            result['all_seq_tracks'] = copy.deepcopy(self.tracks)

        # return the results
        result['seq_name'] = seq_name
        result['frame_index'] = frame_index
        result['hist_trajs'] = hist_trajs
        result['fut_trajs'] = fut_trajs
        result['hist_masks'] = hist_masks
        result['fut_masks'] = fut_masks
        result['tracks'] = full_tracks # only include the tracks related to this frame
        return result


def interpolate(traj, mask):
    interp_positions = np.zeros((traj.shape[0], traj.shape[1]))
    interp_steps = np.arange(traj.shape[0])
    interp_mask = mask
    has_pos = interp_mask.nonzero()[0]
    orig_traj = traj[has_pos]

    for i in range(traj.shape[1]):
        interp_positions[:, i] = np.interp(interp_steps, has_pos, orig_traj[:, i])
    return interp_positions