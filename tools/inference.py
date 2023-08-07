""" Inference the online forecasting
"""
import argparse, sys, os, numpy as np, yaml, copy, pickle, random

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data.distributed as DataDDP

from tqdm import tqdm
import streaming_forecasting as SF


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/streamer/config.yaml')
    parser.add_argument('--weight_path', type=str, default='')
    parser.add_argument('--save_prefix', type=str, default='./')
    args = parser.parse_args()
    return args


def worker_init_fn(pid):
    np_seed = int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)
    return


def main(args):
    config_path = args.config
    configs = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    # build a streaming dataset
    data_reader_configs = copy.deepcopy(configs['data']['val'])
    data_reader_configs.update(configs['general'])
    model_configs = yaml.load(open(
        configs['forecasting']['model_config_path'], 'r'), 
        Loader=yaml.FullLoader)
    configs['predictor'] = model_configs
    
    streaming_reader = SF.streaming.data.StreamingReader(
        configs=configs, data_dir=data_reader_configs['data_dir'], 
        benchmark_file=data_reader_configs['benchmark_file'], infos_file=data_reader_configs['infos_file'],
        hdmap_dir=data_reader_configs['hdmap_dir'],
        frames_per_sample=data_reader_configs['frames_per_sample'],
        hist_len=data_reader_configs['hist_len'], fut_len=data_reader_configs['fut_len'])
    
    sampler = DataDDP.DistributedSampler(
        streaming_reader, num_replicas=1, shuffle=False, rank=0)
    streaming_dataloader = DataLoader(
        streaming_reader,
        batch_size=1,
        num_workers=1,
        collate_fn=SF.streaming.data.streaming_collate_fn,
        sampler=sampler,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        drop_last=True)

    device = 'cuda:0'
    model_name = configs['forecasting']['streaming_model_name']
    model =SF.streamer.build_model(model_name, configs).to(device)
    ckpt = torch.load(args.weight_path, map_location=lambda storage, loc: storage)
    model.load_pretrain(ckpt['state_dict'])

    pbar = tqdm(total=len(streaming_reader))
    all_results = list()
    for sample_idx, sample in enumerate(streaming_dataloader):
        # sample = streaming_reader[i]
        
        # inference
        sample_result = dict()
        sample_result['results'] = model(sample, device, compute_loss=False)[0][0]
        sample_result['seq_name'] = sample[0][0]['seq_name']
        sample_result['city_name'] = sample[0][0]['city_name']
        sample_result['frame_index'] = sample[0][0]['frame_index']
        all_results.append(sample_result)
        pbar.update(1)
    pbar.close()

    os.makedirs(args.save_prefix, exist_ok=True)
    save_path = os.path.join(args.save_prefix, 'streaming_inference.pkl')
    pickle.dump(all_results, open(save_path, 'wb'))


if __name__ == '__main__':
    args = parse_args()
    main(args)