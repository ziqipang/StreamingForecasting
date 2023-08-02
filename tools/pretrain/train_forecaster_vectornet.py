import os, argparse, random, time, numpy as np, sys, subprocess, yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data.distributed as DataDDP
from torch.utils.tensorboard import SummaryWriter
import wandb
# ========== Builder ========== #
from streaming_forecasting.forecaster.builder import builder
# ========== Utilities ========== #
from streaming_forecasting.forecaster.utils import Formatter


def parse_args():
    # ==================== Paths ==================== #
    parser = argparse.ArgumentParser(description='Train models')
    parser.add_argument('--model_name', type=str, default='VectorNet')
    parser.add_argument('--config_path', type=str, default='./configs/forecaster/VectorNet.yaml')
    parser.add_argument('--exp_name', type=str, default='debug')
    parser.add_argument('--model_save_dir', type=str, default='./results/')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--wandb', action='store_true', default=False)
    args = parser.parse_args()
    return args


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.uniform_(m.bias)


def worker_init_fn(pid):
    np_seed = int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)
    return


def read_configs(config_path):
    configs= dict()
    main_config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    for key in main_config.keys():
        configs[key] = yaml.load(open(main_config[key], 'r'), Loader=yaml.FullLoader)
    return configs


def main(args, config_path, save_dir, checkpoint):
    if args.wandb:
        wandb.init(project='VectorNet', entity='ziqipang', name=f'{args.model_name}:{args.exp_name}', 
            dir=os.path.join('./wandb', args.model_name, args.exp_name))

    configs = read_configs(config_path)
    dataset, collate_fn, model, loss = builder(args.model_name, configs['data'], configs['model'])

    # ========== Init Dataset ========== #
    train_data_dir = configs['data'].get('train_dir')
    train_map_dir = configs['data'].get('train_map_dir')
    # train_map_dir = None
    train_dataset = dataset(train_data_dir, configs['data'], map_preprocess=train_map_dir, ratio=configs['data']['ratio'])

    train_sampler = DataDDP.DistributedSampler(
        train_dataset, num_replicas=1, shuffle=True, rank=0)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=configs['data']['batch_size'],
        num_workers=configs['data']['workers'],
        collate_fn=collate_fn,
        sampler=train_sampler,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        drop_last=True
    )

    # ========== Init Model ========== #
    device = torch.device(0)
    model = model(configs['model']).to(device)
    model.apply(init_weights)
    loss = loss(configs['model']).to(device)

    # ========== Init Optimization ========== #
    optimizer = torch.optim.AdamW(model.parameters(), lr=configs['optim']['lr'], weight_decay=configs['optim']['weight_decay'])

    # # ========== Training ========== #
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, configs['optim']['num_epochs'],
    #                                                        0.1 * configs['optim']['lr'], -1)
    # ========== Warmup ========== #
    start_epoch = 0
    if start_epoch == 0:
        warmup(configs, train_dataloader, model, loss, optimizer, 
            configs['optim']['warmup_iters'], configs['optim']['lr'], configs['optim']['lr'] * configs['optim']['warmup_ratio'],
            device)
    
    # ========== Training ========== #
    for epoch in range(start_epoch, configs['optim']['num_epochs']):
        if epoch in configs['optim']['lr_decay_epoch']:
            lr = optimizer.param_groups[0]['lr']
            for g in optimizer.param_groups:
                g['lr'] *= configs['optim']['lr_decay_rate']
        train(epoch, configs, train_dataloader, model, loss, optimizer, device)
        save_checkpoint(model, optimizer, save_dir, epoch)
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


def warmup(configs, data_loader, model, loss, optimizer, warmup_iter=4000, lr_max=1e-3, lr_min=1e-6, device='cuda:0', local_rank=0):
    display_iteration = 10
    start_time = time.time()
    model.train()
    formatter = Formatter()
    for i, data in enumerate(data_loader):
        # set learning rate
        for g in optimizer.param_groups:
            g['lr'] = lr_min + (lr_max - lr_min) * i / warmup_iter
        
        # train
        data = dict(data)
        output = model(data, device)
        loss_out = loss(output, data['gt_futures'], data['gt_future_masks'], device)

        optimizer.zero_grad()
        loss_out['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), configs['optim']['grad_clip'])
        optimizer.step()

        formatter.append(output, data, loss_out)
        if i % display_iteration == 0 and local_rank == 0:
            iter_num = i
            message, infos = formatter.display(iter_num)
            print('WARMUP TIME %.2f ITER %d/%d LR %2.4f: %s' % (
                time.time() - start_time, i + 1, warmup_iter, 
                optimizer.param_groups[0]['lr'], message))
            infos['lr'] = optimizer.param_groups[0]['lr']
        if args.wandb:
            wandb.log(infos)
        
        if i >= warmup_iter:
            break
    return


def train(epoch, configs, data_loader, model, loss, optimizer, device='cuda:0'):
    data_loader.sampler.set_epoch(int(epoch))
    num_batches = len(data_loader)
    display_iteration = 10
    
    start_time = time.time()
    model.train()

    formatter = Formatter()
    for i, data in enumerate(data_loader):
        data = dict(data)
        output = model(data, device)
        loss_out = loss(output, data['gt_futures'], data['gt_future_masks'], device)

        optimizer.zero_grad()
        loss_out['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), configs['optim']['grad_clip'])
        optimizer.step()

        formatter.append(output, data, loss_out)
        if i % display_iteration == 0:
            iter_num = num_batches * epoch + i
            message, infos = formatter.display(iter_num)
            print('EPOCH %d TIME %.2f ITER %d/%d LR %2.5f: %s' % (
                epoch, time.time() - start_time, i + 1, num_batches, 
                optimizer.param_groups[0]['lr'], message))
            infos['lr'] = optimizer.param_groups[0]['lr']
            if args.wandb:
                wandb.log(infos)
    return


if __name__ == '__main__':
    start_time = time.time()
    gpu_num = torch.cuda.device_count()
    args = parse_args()

    save_dir = os.path.join(args.model_save_dir, args.model_name, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    ckpt_dir = os.path.join(save_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)

    main(args, args.config_path, ckpt_dir, args.checkpoint)