""" Preprocess the hdmaps for faster training and inference
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
    parser.add_argument('--config', type=str, default='./configs/streamer/config_single_frame.yaml')
    parser.add_argument('--hdmap_dir', type=str, default='./data/argoverse_tracking/')
    args = parser.parse_args()
    return args


def main(args):
    config_path = args.config
    configs = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    # build a streaming dataset
    for split in ['train', 'val']:
        print(f'Preprocessing for {split}')
        data_reader_configs = copy.deepcopy(configs['data'][split])
        data_reader_configs.update(configs['general'])
        model_configs = yaml.load(open(
            configs['forecasting']['model_config_path'], 'r'), 
            Loader=yaml.FullLoader)
        configs['predictor'] = model_configs
        
        streaming_reader = SF.streaming.data.StreamingReader(
            configs=configs, data_dir=data_reader_configs['data_dir'], 
            benchmark_file=data_reader_configs['benchmark_file'], infos_file=data_reader_configs['infos_file'],
            frames_per_sample=data_reader_configs['frames_per_sample'],
            hist_len=data_reader_configs['hist_len'], fut_len=data_reader_configs['fut_len'])
        
        save_dir = os.path.join(args.hdmap_dir, f'{split}_hdmaps')
        os.makedirs(save_dir, exist_ok=True)

        pbar = tqdm(total=len(streaming_reader))
        for i in range(len(streaming_reader)):
            streaming_reader.preprocess_map(
                i, save_dir)
            pbar.update(1)
        pbar.close()
    return


if __name__ == '__main__':
    args = parse_args()
    main(args)