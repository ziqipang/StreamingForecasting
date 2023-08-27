""" Using KF as a prior for trajectories
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from ..core import Track


class KFPrior:
    def __init__(self):
        return
    
    def estimate(self, track: Track, frame_index, hist_len, fut_len, num_mod=6):
        """ Estimate the prior information of a track
            Infer for (frame_idx: frame_idx + expect_len)
        """
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        kf.P[2:, 2:] *= 1000.  # state uncertainty
        kf.P *= 10.
    
        # use the observed history to fit the KF
        # make prediction jointly
        result_traj, result_steps = np.zeros((hist_len, 2)), np.zeros(hist_len).astype(np.int32)
        fut_traj = np.zeros((fut_len, 2))
        first_step = track.frames[0]
        kf.x[:2] = track.traj[first_step, :2].reshape((2, 1))

        # starting frame earlier than the history
        if first_step < frame_index:
            for i in range(first_step, frame_index):
                kf.predict()
                if track.mask[i]:
                    kf.update(track.traj[i, :2])
        
            # predict
            for i in range(0, hist_len):
                kf.predict()
                if track.mask[i + frame_index]:
                    kf.update(track.traj[i + frame_index, :2])
                    states = track.traj[frame_index + i, :2]
                else:
                    states = kf.x[:2].reshape(-1)
                result_traj[i, :2] = states
            result_steps = np.arange(hist_len)
        else:
            for i in range(first_step, frame_index + hist_len):
                kf.predict()
                if track.mask[i]:
                    kf.update(track.traj[i, :2])
                    states = track.traj[i, :2]
                else:
                    states = kf.x[:2].reshape(-1)
                result_traj[i - frame_index, :2] = states
            result_steps = np.arange(first_step - frame_index, hist_len)
            result_traj = result_traj[first_step - frame_index:]
        
        # predicting a prior future trajectory
        for i in range(fut_len):
            kf.predict()
            states = kf.x[:2].reshape(-1)
            fut_traj[i, :2] = states
        fut_traj = np.repeat(fut_traj[np.newaxis, ...], num_mod, axis=0)

        return result_traj, result_steps, fut_traj