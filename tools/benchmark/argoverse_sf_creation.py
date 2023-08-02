""" For every tracklet, assign a category for evaluation:
    1. If the object is of invalid object type, >100m, far from road, we exclude it from the evaluation.
    2. For the valid objects, we classify them according to move/static and visible/occluded.
"""
import numpy as np
import os, argparse, sys
import pickle
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from tqdm import tqdm

import streaming_forecasting as SF
from streaming_forecasting.streaming.core import Position, Track


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/argoverse_tracking/val/')
    parser.add_argument('--output_dir', type=str, default='./data/streaming_forecasting/')
    parser.add_argument('--save_prefix', type=str, default='eval_cat_val')
    parser.add_argument('--hist_length', type=int, default=20, help='default history length')
    parser.add_argument('--fut_length', type=int, default=30, help='default future length')
    parser.add_argument('--mini', action='store_true', default=False, help='use 2 sequences')
    args = parser.parse_args()
    return args


def load_seq_gt_infos(argo_loader: ArgoverseTrackingLoader, seq_index):
    """ Load the positions, ids, classes of all the objects
        Load the ego poses, and city names
    """
    argo_data = argo_loader[seq_index]
    city_name = argo_data.city_name
    frame_num = len(argo_data.lidar_list)
    positions, egos, ids, classes = list(), list(), list(), list()
    for frame_index in range(frame_num):
        frame_argo_labels = argo_data.get_label_object(frame_index)
        frame_ids = [l.track_id for l in frame_argo_labels]
        frame_classes = [l.label_class for l in frame_argo_labels]
        frame_positions = [Position(l.translation) for l in frame_argo_labels]
        frame_ego_poses = argo_data.get_pose(frame_index).transform_matrix

        ids += [frame_ids]
        classes += [frame_classes]
        positions += [frame_positions]
        egos += [frame_ego_poses]
    
    positions = SF.streaming.utils.data_utils.positions2world(positions, egos)
    return ids, classes, positions, egos, city_name


def evaluation_category_filter(tracks, egos, hdmap: ArgoverseMap, hist_len, fut_len, city_name):
    """ -1: invalid, 0: visible, 1: frames of occlusion
    """
    frame_num = len(egos)
    track_keys = list(tracks.keys())
    eval_category = dict()

    """ Enumerate over tracks, classify (seq, frame, id) into evaluation categories """
    # interpolate for missing observations
    interp_tracks = dict()
    for k in track_keys:
        trk: Track
        trk = tracks[k]
        interp_tracks[k] = trk.interpolate(trk.frames[0], trk.frames[-1] + 1)[0]
    
    for frame_idx in range(hist_len - 1, frame_num - fut_len):
        ego_xyz = egos[frame_idx][:3, 3]
        eval_category[frame_idx] = dict()

        for k in track_keys:
            trk: Track
            trk = tracks[k]

            # Filter 1: class type filter
            if trk.obj_type not in SF.streaming.core.macros.ARGO_VEHICLE_CLASSES:
                eval_category[frame_idx][k] = 'invalid'
                continue

            # Filter 2: temporal range filter
            if trk.frames[0] > frame_idx or trk.frames[-1] <= frame_idx:
                eval_category[frame_idx][k] = 'invalid'
                continue

            # get the trajectory data, filter according to the ranges
            traj = interp_tracks[k]

            # Filter 3: distance to ego vehicle
            xyz = traj[frame_idx - trk.frames[0]]
            if np.linalg.norm(xyz[:2] - ego_xyz[:2]) >= 100:
                eval_category[frame_idx][k] = 'invalid'
                continue

            # Filter 4: near the road, justified with ROI in Argoverse
            raster_layer = hdmap.get_raster_layer_points_boolean(xyz.reshape((1, 3)), city_name, 'roi')
            if np.sum(raster_layer) == 0: 
                eval_category[frame_idx][k] = 'invalid'
                continue

            # Generate description strings for categorization
            # Category 1: At least it is  valid trajectory
            # Category 2: Move or not
            start_pos, end_pos = trk.traj[trk.frames[0]], trk.traj[trk.frames[-1]]
            dist = np.linalg.norm((end_pos - start_pos)[:2])
            if dist < 3.0:
                motion_cat = 'stay'
            else:
                motion_cat = 'move'
            
            # Category 3: Visible or not
            if trk.mask[frame_idx]:
                vis_cat = 'vis'
            else:
                vis_cat = 'occ'
            
            # Category 4: Occlusion length
            if vis_cat == 'vis':
                occ_len = 0
            else:
                prev_observed_frame = np.searchsorted(np.array(trk.frames), frame_idx)
                prev_observed_frame -= 1
                occ_len = frame_idx - trk.frames[prev_observed_frame]
            
            cat_string = f'{vis_cat}_{motion_cat}_{occ_len}'
            eval_category[frame_idx][k] = cat_string
    return eval_category


def main(args):
    argo_loader = ArgoverseTrackingLoader(args.data_dir)
    seq_num = len(argo_loader.log_list)
    if args.mini:
        seq_num = 2
    hdmap = ArgoverseMap()

    pbar = tqdm(total=seq_num)

    eval_cat = dict()
    for seq_index in range(seq_num):
        seq_name = argo_loader.log_list[seq_index]        
        ids, classes, positions, egos, city_name = load_seq_gt_infos(argo_loader, seq_index)
        tracks = SF.streaming.utils.data_utils.seq2tracklets(positions, ids, classes, SF.streaming.core.macros.ARGO_VEHICLE_CLASSES)
        seq_eval_cat = evaluation_category_filter(tracks, egos, hdmap, 
                                                  args.hist_length, args.fut_length, city_name)
        eval_cat[seq_name] = seq_eval_cat

        pbar.update(1)
    pbar.close()

    save_traj_path = os.path.join(args.output_dir, f'{args.save_prefix}.pkl')
    f = open(save_traj_path, 'wb')
    pickle.dump(eval_cat, f)
    f.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)