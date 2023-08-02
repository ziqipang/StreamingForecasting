""" Acquire the trajectories and the query/observation keys on every frame
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
    parser.add_argument('--benchmark_file', type=str, default='eval_cat_val.pkl')
    parser.add_argument('--output_dir', type=str, default='./data/streaming_forecasting/')
    parser.add_argument('--save_prefix', type=str, default='infos_val')
    parser.add_argument('--hist_length', type=int, default=20, help='default history length')
    parser.add_argument('--fut_length', type=int, default=30, help='default future length')
    parser.add_argument('--mini', action='store_true', default=False)
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

        # add ego as a vehicle
        frame_ids += [f'ego_{seq_index}']
        frame_classes += ['VEHICLE']
        frame_positions += [Position(np.zeros(3))]

        # add agent information to the sequence
        ids += [frame_ids]
        classes += [frame_classes]
        positions += [frame_positions]
        egos += [frame_ego_poses]
    
    positions = SF.streaming.utils.data_utils.positions2world(positions, egos)
    return ids, classes, positions, egos, city_name


def sample_infos(argo_loader: ArgoverseTrackingLoader, hdmap: ArgoverseMap, seq_index, hist_len, fut_len):
    # prepare the track information
    seq_name = argo_loader.log_list[seq_index]
    ids, classes, positions, egos, city_name = load_seq_gt_infos(argo_loader, seq_index)
    tracks = SF.streaming.utils.data_utils.seq2tracklets(positions, ids, classes, SF.streaming.core.macros.ARGO_VEHICLE_CLASSES)

    frame_num = len(egos)
    track_keys = list(tracks.keys())
    samples = list()

    # interpolate for missing observations
    interp_tracks = dict()
    for k in track_keys:
        trk: Track
        trk = tracks[k]
        interp_tracks[k] = trk.interpolate(trk.frames[0], trk.frames[-1] + 1)[0]

    # extract the sample information
    for frame_idx in range(hist_len - 1, frame_num - fut_len):
        sample = dict()

        ego = egos[frame_idx]
        ego_xyz = ego[:3, 3]
        timestamp = argo_loader[seq_index].lidar_timestamp_list[frame_idx]

        sample['seq_name'] = seq_name
        sample['ego'] = ego
        sample['timestamp'] = timestamp
        sample['frame_index'] = frame_idx
        sample['city_name'] = city_name

        observation_keys, query_keys = list(), list()
        occ_keys = list()
        for k in track_keys:
            trk: Track
            trk = tracks[k]

            # Filter 1: class type filter
            if trk.obj_type not in SF.streaming.core.macros.ARGO_VEHICLE_CLASSES:
                continue

            # Filter 2: temporal range filter
            if trk.frames[0] > frame_idx or trk.frames[-1] <= frame_idx:
                continue

            # get the trajectory data, filter according to the ranges
            traj = interp_tracks[k]

            # Filter 3: distance to ego vehicle
            xyz = traj[frame_idx - trk.frames[0]]
            if np.linalg.norm(xyz[:2] - ego_xyz[:2]) >= 100:
                continue

            # Filter 4: near the road, justified with ROI in Argoverse
            raster_layer = hdmap.get_raster_layer_points_boolean(xyz.reshape((1, 3)), city_name, 'roi')
            if np.sum(raster_layer) == 0:
                continue

            # Save the object information, indicate whether it is visible or not
            # ego vehicle is not part of prediction
            query_keys.append(k)
            if trk.mask[frame_idx]:
                observation_keys.append(k)
            else:
                occ_keys.append(k)
        sample['query_keys'] = query_keys
        sample['observation_keys'] = observation_keys
        sample['occ_keys'] = occ_keys
        samples.append(sample)
            
    return samples


def main(args):
    argo_loader = ArgoverseTrackingLoader(args.data_dir)
    seq_num = len(argo_loader.log_list)
    if args.mini:
        seq_num = 2
    hdmap = ArgoverseMap()

    pbar = tqdm(total=seq_num)
    trajectories, samples = dict(), list()
    timestamps = dict()
    for seq_index in range(seq_num): 
        ids, classes, positions, egos, city_name = load_seq_gt_infos(argo_loader, seq_index)
        tracks = SF.streaming.utils.data_utils.seq2tracklets(positions, ids, classes, SF.streaming.core.macros.ARGO_VEHICLE_CLASSES)

        # save the trajectory data
        # everything in the world coordinate
        track_keys = list(tracks.keys())
        for key in track_keys:
            trajectories[key] = {
                'traj': tracks[key].traj,
                'mask': tracks[key].mask,
                'frames': tracks[key].frames,
                'id': tracks[key].id,
                'obj_type': tracks[key].obj_type
            }
        
        timestamps[argo_loader.log_list[seq_index]] = argo_loader[seq_index].lidar_timestamp_list
        
        # save the sample information
        seq_samples = sample_infos(argo_loader, hdmap, seq_index, args.hist_length, args.fut_length)
        samples += seq_samples
        pbar.update(1)
    pbar.close()

    infos = {
        'timestamps': timestamps,
        'trajectories': trajectories,
        'samples': samples
    }

    save_traj_path = os.path.join(args.output_dir, f'{args.save_prefix}.pkl')
    f = open(save_traj_path, 'wb')
    pickle.dump(infos, f)
    f.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)