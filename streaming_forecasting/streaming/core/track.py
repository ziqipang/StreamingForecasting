import numpy as np
from copy import deepcopy
from .position import Position


class Track:
    def __init__(self, id, obj_type, start_pos: Position, start_frame, max_len=320) -> None:
        self.id = id
        self.obj_type = obj_type

        # trajectory data
        self.traj = np.zeros((max_len, 3))
        # mask of having data or not
        self.mask = np.zeros(max_len).astype(np.int32)
        self.frames = list()
        
        self.traj[start_frame] = start_pos.xyz
        self.mask[start_frame] = 1
        self.frames.append(start_frame)
        return
    
    def load_new_pos(self, pos: Position, frame_idx):
        self.traj[frame_idx] = pos.xyz
        self.mask[frame_idx] = 1
        self.frames.append(frame_idx)
        return
    
    def set_pos(self, pos: Position, frame_idx):
        self.traj[frame_idx] = pos.xyz
        self.mask[frame_idx] = 1
        if frame_idx not in self.frames:
            self.frames.append(frame_idx)
            self.frames = sorted(self.frames)
        return
    
    def traj_by_frame(self, first, last):
        """ Return the trajectories between the first and the last frames.
        """
        return deepcopy(self.traj[first:last]), deepcopy(self.mask[first:last])

    def interpolate(self, first, last):
        assert first >= self.frames[0]
        assert last > self.frames[-1]
        
        interp_positions = np.zeros((last - first, 3))
        interp_steps = np.arange(last - first)
        interp_mask = self.mask[first:last]
        has_pos = interp_mask.nonzero()[0]
        orig_traj = self.traj[first:last][has_pos]

        for i in range(3):
            interp_positions[:, i] = np.interp(interp_steps, has_pos, orig_traj[:, i])
        
        return interp_positions, interp_steps + first
    
    @classmethod
    def init_from_infos(self, id, obj_type, traj, mask, frames):
        start_frame = frames[0]
        start_pos = Position(traj[start_frame])
        track = Track(id, obj_type, start_pos, start_frame)
        track.traj = deepcopy(traj)
        track.mask = deepcopy(mask)
        track.frames = deepcopy(frames)
        return track