import os, numpy as np, argparse, yaml, pickle, json
from tqdm import tqdm
import streaming_forecasting as SF


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/streamer/config.yaml')
    parser.add_argument('--result_file', type=str)
    parser.add_argument('--metric_file_prefix', type=str, default='')
    args = parser.parse_args()
    return args


def main(args):
    config_path = args.config
    configs = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    benchmark_file = pickle.load(open(configs['data']['val']['benchmark_file'], 'rb'))
    infos_file = pickle.load(open(configs['data']['val']['infos_file'], 'rb'))
    trajectories = infos_file['trajectories']
    hist_len, fut_len = configs['general']['hist_len'], configs['general']['fut_len']
    results = pickle.load(open(args.result_file, 'rb'))

    eval_groups = {
        'move': {'vis': {}, 'occ': {}},
        'stay': {'vis': {}, 'occ': {}}
    }
    all_eval_track_infos = dict()

    # load gt and predictions
    print('Loading Predictions and Ground-truths')
    pbar = tqdm(total=len(results))
    for sample_idx, sample in enumerate(results):
        # locate the benchmark record
        seq_name, frame_index = sample['seq_name'], sample['frame_index']
        eval_cats = benchmark_file[seq_name][frame_index]

        # enumerate over all the tracks
        for track_id in eval_cats.keys():
            if eval_cats[track_id] == 'invalid':
                continue
            
            vis_cat, motion_cat, occ_len = eval_cats[track_id].split('_')
            if track_id not in eval_groups[motion_cat][vis_cat].keys():
                eval_groups[motion_cat][vis_cat][track_id] = {
                    'pd': dict(), 'confidences': dict(), 'frame_indexes': list(),
                    'gt': dict(), 'gt_masks': dict(), 'city_names': dict(),}
            if track_id not in all_eval_track_infos.keys():
                all_eval_track_infos[track_id] =  {
                    'pd': dict(), 'confidences': dict(), 'frame_indexes': list(),
                    'gt': dict(), 'gt_masks': dict(), 'city_names': dict(),}
            
            # record prediction information
            track_eval_infos = eval_groups[motion_cat][vis_cat][track_id]
            track_eval_infos['frame_indexes'].append(frame_index)
            track_eval_infos['pd'][frame_index] = sample['results'][track_id]['trajs']
            track_eval_infos['confidences'][frame_index] = sample['results'][track_id]['confidences']
            track_eval_infos['city_names'][frame_index] = sample['city_name']

            all_eval_track_infos[track_id]['frame_indexes'].append(frame_index)
            all_eval_track_infos[track_id]['pd'][frame_index] = sample['results'][track_id]['trajs']
            all_eval_track_infos[track_id]['confidences'][frame_index] = sample['results'][track_id]['confidences']
            all_eval_track_infos[track_id]['city_names'][frame_index] = sample['city_name']
            
            # record gt information
            track_eval_infos['gt'][frame_index] = trajectories[track_id]['traj'][frame_index + 1: frame_index + 1 + fut_len]
            track_eval_infos['gt_masks'][frame_index] = trajectories[track_id]['mask'][frame_index + 1: frame_index + 1 + fut_len]
            all_eval_track_infos[track_id]['gt'][frame_index] = trajectories[track_id]['traj'][frame_index + 1: frame_index + 1 + fut_len]
            all_eval_track_infos[track_id]['gt_masks'][frame_index] = trajectories[track_id]['mask'][frame_index + 1: frame_index + 1 + fut_len]
        pbar.update(1)
    pbar.close()

    # compute the ade and fde per track id
    metric_vals = {
        'move': {'vis': {}, 'occ': {}},
        'stay': {'vis': {}, 'occ': {}}
    }
    for motion in ['move', 'stay']:
        for vis in ['vis', 'occ']:
            print(f'Compute for Group {motion}-{vis}')
            track_keys = list(eval_groups[motion][vis].keys())
            pbar = tqdm(total=len(track_keys))
            for _, k in enumerate(track_keys):
                obj_info = eval_groups[motion][vis][k] # the information about a single id
                frame_indexes = obj_info['frame_indexes']
                pds = np.concatenate([np.array(obj_info['pd'][i])[np.newaxis, ...] for i in frame_indexes], axis=0)
                confs = np.concatenate([np.array(obj_info['confidences'][i])[np.newaxis, ...] for i in frame_indexes], axis=0)
                gts = np.concatenate([np.array(obj_info['gt'][i])[np.newaxis, ...] for i in frame_indexes], axis=0)
                gt_masks = np.concatenate([np.array(obj_info['gt_masks'][i])[np.newaxis, ...] for i in frame_indexes], axis=0)

                vals = SF.streaming.metrics.compute_forecasting_metrics(pds, confs, gts, gt_masks)
                metric_vals[motion][vis][k] = vals
                pbar.update(1)
            pbar.close()
    
    # summarize the numbers
    results = dict()
    for motion in ['move', 'stay']:
        results[motion] = dict()
        for vis in ['vis', 'occ']:
            track_keys = list(eval_groups[motion][vis].keys())
            results[motion][vis] = dict()
            for m in SF.streaming.metrics.METRIC_NAMES:
                vals = np.array([metric_vals[motion][vis][k][m] for k in track_keys])
                # NOTE: we use <0 values to denote invalid values
                vals = vals[vals >= 0]
                results[motion][vis][m] = np.average(vals)

    for m in SF.streaming.metrics.METRIC_NAMES:
        val = 0
        for motion in ['move', 'stay']:
            for vis in ['vis', 'occ']:
                val += results[motion][vis][m]
        results[f'ave-{m}'] = val / 4
    print(results)

    if args.metric_file_prefix != '':
        save_path = os.path.join(args.metric_file_prefix, 'metrics.json')
        f = open(save_path, 'w')
        json.dump(results, f)
        f.close()
    return


if __name__ == '__main__':
    args = parse_args()
    main(args)