""" Train online forecasting models
"""
import argparse, sys, os, numpy as np, yaml
import time, copy, pickle, random

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data.distributed as DataDDP

from tqdm import tqdm
import wandb
import streaming_forecasting as SF


WANDB_LOGGING = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/streamer/config.yaml')
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--weight_path', type=str, default='')
    parser.add_argument('--save_prefix', type=str, default='./results/')
    parser.add_argument('--wandb', action='store_true', default=False)
    args = parser.parse_args()
    return args


def worker_init_fn(pid):
    np_seed = int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)
    return


def save_checkpoint(model, optimizer, save_dir, epoch):
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    save_name = f'{epoch}.ckpt'
    torch.save(
        {'epoch': epoch, 'state_dict': state_dict, 'optimizer': optimizer.state_dict()},
        os.path.join(save_dir, save_name),
    )
    return


def train(epoch, configs, dataloader, model, optimizer, device='cuda:0'):
    dataloader.sampler.set_epoch(int(epoch))
    num_batches = len(dataloader)
    display_iteration = 1
    
    start_time = time.time()
    model.train()

    for i, data in enumerate(dataloader):
        output, loss_dict = model(data, device, compute_loss=True)

        optimizer.zero_grad()
        loss_dict['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), configs['optim']['grad_clip'])
        optimizer.step()

        if i % display_iteration == 0:
            loss_keys = loss_dict.keys()
            message = ''
            for k in loss_keys:
                message += '%s: %.4f, ' % (k, loss_dict[k].item())
            print('E %d TIME %.2f ITER %d/%d LR %2.6f: %s' % (
                epoch, time.time() - start_time, i + 1, num_batches,
                optimizer.param_groups[0]['lr'], message))
            
            global WANDB_LOGGING
            if WANDB_LOGGING:
                infos = {
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr'],
                    'time': time.time() - start_time}
                for k in loss_keys:
                    infos[f'loss/{k}'] = loss_dict[k].item()
                wandb.log(infos)
    return


def main(args):
    config_path = args.config
    configs = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    # ========== Misc ========== #
    work_dir = os.path.join(args.save_prefix, args.name)
    os.makedirs(work_dir, exist_ok=True)
    save_dir = os.path.join(args.save_prefix, args.name, 'ckpts')
    os.makedirs(save_dir, exist_ok=True)
    if args.wandb:
        wandb.init(project='OnlineForecasting', entity='ziqipang', name=args.name, 
            dir=work_dir)
        global WANDB_LOGGING
        WANDB_LOGGING = True

    # ========== Streaming dataset ========== #
    data_reader_configs = copy.deepcopy(configs['data']['train'])
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
        streaming_reader, num_replicas=1, shuffle=True, rank=0)
    streaming_dataloader = DataLoader(
        streaming_reader,
        batch_size=data_reader_configs['batch_size'],
        num_workers=4,
        collate_fn=SF.streaming.data.streaming_collate_fn,
        sampler=sampler,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        drop_last=True)
    
    # ========== Models ========== #
    device = 'cuda:0'
    model_name = configs['forecasting']['streaming_model_name']
    model = SF.streamer.build_model(model_name, configs).to(device)
    ckpt = torch.load(args.weight_path, map_location=lambda storage, loc: storage)
    model.load_pretrain(ckpt['state_dict'])

    # ========== Training related modules ========== #
    optimizer = torch.optim.AdamW(model.parameters(), lr=configs['optim']['lr'], 
                                  weight_decay=configs['optim']['weight_decay'])
    
    # ========== Training ========== #
    start_epoch = 0
    for epoch in range(start_epoch, configs['optim']['num_epochs']):
        if epoch in configs['optim']['lr_decay_epoch']:
            for g in optimizer.param_groups:
                g['lr'] *= configs['optim']['lr_decay_rate']
        train(epoch, configs, streaming_dataloader, model, optimizer, device)
        save_checkpoint(model, optimizer, save_dir, epoch)
    return


if __name__ == '__main__':
    args = parse_args()
    main(args)