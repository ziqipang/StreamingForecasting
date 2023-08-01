import enum
from ..core import Position, Track, TrackBank
import numpy as np


def seq2tracklets(positions, ids, classes, class_filter):
    bank = TrackBank()
    for _, (frame_pos, frame_ids, frame_cls) in enumerate(zip(positions, ids, classes)):
        if class_filter is not None:
            sel_idx = [i for i, c in enumerate(frame_cls) if c in class_filter]
            frame_pos = [frame_pos[i] for i in sel_idx]
            frame_ids = [frame_ids[i] for i in sel_idx]
            frame_cls = [frame_cls[i] for i in sel_idx]
        bank.load_new_pos(frame_pos, frame_ids, frame_cls)
        bank.frame_idx += 1
    bank.frame_idx -= 1
    return bank.get_tracks()


def positions2world(positions, ego_poses):
    frame_num = len(ego_poses)
    for i in range(frame_num):
        ego = ego_poses[i]
        for j in range(len(positions[i])):
            positions[i][j].pos2world(ego, inplace=True)
    return positions