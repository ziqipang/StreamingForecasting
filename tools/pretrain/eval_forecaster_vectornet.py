import os, argparse, random, time, numpy as np, sys, subprocess, yaml, json, pickle
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data.distributed as DataDDP
from tqdm import tqdm
# ========== Builder ========== #
from streaming_forecasting.forecaster.builder import builder
# ========== Utilities ========== #
from streaming_forecasting.forecaster.utils.eval_utils import LaneGCNFormatData, compute_forecasting_metrics


def parse_args():
    # ==================== Paths ==================== #
    parser = argparse.ArgumentParser(description='Train models')
    parser.add_argument('--model_name', type=str, default='VectorNet')
    parser.add_argument('--config_path', type=str, default='./configs/forecaster/VectorNet.yaml')
    parser.add_argument('--exp_name', type=str, default='debug')
    parser.add_argument('--model_save_dir', type=str, default='./results/')
    parser.add_argument('--weight_path', type=str, default='')
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


def load_pretrain(net, pretrain_dict):
    state_dict = net.state_dict()
    for key in pretrain_dict.keys():
        if key in state_dict and (pretrain_dict[key].size() == state_dict[key].size()):
            value = pretrain_dict[key]
            if not isinstance(value, torch.Tensor):
                value = value.data
            state_dict[key] = value
    net.load_state_dict(state_dict)


def read_configs(config_path):
    configs= dict()
    main_config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    for key in main_config.keys():
        configs[key] = yaml.load(open(main_config[key], 'r'), Loader=yaml.FullLoader)
    return configs


def main(args, config_path, save_dir, weight_path):

    configs = read_configs(config_path)
    dataset, collate_fn, model, loss = builder(args.model_name, configs['data'], configs['model'])

    # ========== Init Dataset ========== #
    val_data_dir = configs['data'].get('val_dir')
    val_map_dir = configs['data'].get('val_map_dir')
    # train_map_dir = None
    val_dataset = dataset(val_data_dir, configs['data'], map_preprocess=val_map_dir, ratio=configs['data']['ratio'])

    # for i in range(10):
    #     data = val_dataset[i]
    #     import pdb
    #     pdb.set_trace()

    val_sampler = DataDDP.DistributedSampler(
        val_dataset, num_replicas=1, shuffle=False, rank=0)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=configs['data']['batch_size'],
        num_workers=configs['data']['workers'],
        collate_fn=collate_fn,
        sampler=val_sampler,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        drop_last=False
    )

    # ========== Init Model ========== #
    device = torch.device(0)
    model = model(configs['model']).to(device)
    model.apply(init_weights)
    ckpt = torch.load(weight_path, map_location=lambda storage, loc: storage)
    load_pretrain(model, ckpt['state_dict'])

    # ========== Evaluate function ========== #
    format_results = LaneGCNFormatData()
    evaluate6 = partial(compute_forecasting_metrics,
                       max_n_guesses=6,
                       horizon=30,
                       miss_threshold=2.0)
    evaluate1 = partial(compute_forecasting_metrics,
                       max_n_guesses=1,
                       horizon=30,
                       miss_threshold=2.0)
    
    # ========== Eval Epoch ========== #
    model.eval()
    progress_bar = tqdm(val_dataloader)
    with torch.no_grad():
        for j, data in enumerate(progress_bar):
            data = dict(data)
            output = model(data)
            format_results(data, output)
    
    # ========== Save results ========== #
    save_path = os.path.join(save_dir, 'val_preds.pkl')
    pickle.dump(format_results.results, open(save_path, 'wb'))

    metrics6 = evaluate6(**format_results.results)
    metrics1 = evaluate1(**format_results.results)
    print('Validation Process Finished!!')

    metrics = {
        'k=6': metrics6,
        'k=1': metrics1
    }
    json.dump(metrics, open(os.path.join(save_dir, 'val_metrics.json'), 'w'))
    return


if __name__ == '__main__':
    start_time = time.time()
    gpu_num = torch.cuda.device_count()
    args = parse_args()

    save_dir = os.path.join(args.model_save_dir, args.model_name, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    main(args, args.config_path, save_dir, args.weight_path)