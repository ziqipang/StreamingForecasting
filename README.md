# [IROS 2023] Streaming Motion Forecasting for Autonomous Driving

[Ziqi Pang](https://ziqipang.github.io/), [Deva Ramanan](https://www.cs.cmu.edu/~deva/), [Mengtian Li](https://mtli.github.io/), [Yu-Xiong Wang](https://yxw.web.illinois.edu/)

## 1. Introduction

This is the official code for our IROS 2023 paper: "Streaming Motion Forecasting for Autonomous Driving." We propose to view the motion forecasting from a **streaming** perspective, where the predictions are made on continuous frames, instead of the conventional **snapshot-based** forecasting.

If you find our code or paper useful, please cite by:
```Tex
@inproceedings{pang2023streaming,
  title={Streaming motion forecasting for autonomous driving},
  author={Pang, Ziqi and Ramanan, Deva and Li, Mengtian and Wang, Yu-Xiong},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2023}
}
```

## 2. Getting Started with Benchmark (Argoverse-SF)

If you wish to evaluate using the protocol of our streaming forecasting, please pay most attention to the [`Result Format and Evaluation`]() section.

### 2.1 Install `streaming_forecasting` Toolkit

* We recommend install our `streaming_forecasting` toolkit locally via:

```bash
pip install -e ./
```

* Then install the `argoverse-api` following their [instructions](https://github.com/argoverse/argoverse-api#installation).

### 2.2 Data Preparation
<details>
<summary>Click to view the details</summary>

You will prepare the argoverse dataset, both tracking and forecasting splits included. **We recommend you putting the tracking and forecasting into separate directories.** For instance, I use directories `~/datasets/argoverse/tracking` and `~/datasets/argoverse/forecasting`. 

Remember to soft-link your data location to `./data/argoverse_forecasting` and `./data/argoverse_tracking`. The final file structure would be similar to:
```
./data
    -- argoverse_tracking
        -- train
           xxx   xxx   xxx ...
        -- val
           xxx   xxx   xxx ...
        
    -- argoverse_forecasting
        -- train
            -- data
               xxx.csv, xxx.csv, xxx.csv
        -- val
            -- data
                xxx.csv, xxx.csv, xxx.csv
```

#### 2.2.1 Tracking data

Argoverse-SF models streaming forecasting by re-purposing the tracking data from Argoverse. Please skip this step if you already have it.

* Download the tracking split from [Argoverse Link](https://www.argoverse.org/av1.html#download-link). You will see 4 `*.tar.gz` for the training set and 1 `*.tar.gz` for the validation set.
* Extract the data from compressed files locally. Take `tracking_val_v1.1.tar.gz` for example:
```bash
# Normal extraction
tar -xvf tracking_val_v1.1.tar.gz -C ./

# Exclude the images if you have limited disk space
tar -xvf tracking_val_v1.1.tar.gz --exclude="*.jpg" -C ./
``` 
* Move everything out of `argoverse_tracking/` and merge the training files 
```bash
# move everything out
mv argoverse_tracking/* ./

# merge training set
mkdir train
mv train1/* train
mv train2/* train
mv train3/* train
mv train4/* train
```

#### 2.2.2 Forecasting data
In the pretraining step, we will use the forecasting data to train a snapshot-based forecasting model, just like normal forecasters on Argoverse.

* Download the forecasting split from [Argoverse Link](https://www.argoverse.org/av1.html#download-link). You will see 1 `*.tar.gz` for the training set and 1 `*.tar.gz` for the validation set.
* Extract the forecasting file locally. Take `forecasting_val_v1.1.tar.gz` for example, the sript is as below.

```bash
tar -xvf forecasting_val_v1.1.tar.gz
```
</details>

### 2.3 Benchmark (Argoverse-SF) Creation

<details>
<summary>Click to view the details</summary>

We will walk you through:
* Generating the Argoverse-SF benchmark files for evaluation and visualization.
* Generating the information files for dataloading during training and inference, recording the ground truth trajectories and orders of frames. 

#### 2.3.1 Benchmark Creation

Please use our `./tools/benchmark/argoverse_sf_creation.py` to create the Argoverse-SF benchmark, which supports evaluation. The commands is as below, if you follow our instructions on softlinking the argoverse datasets to `./data` as described in [[Dataset Preparation]](https://github.com/ziqipang/StreamingForecasting#data-preparation). After this step, you will see `eval_cat_val.pkl` and `eval_cat_train.pkl` popping up in `./data/streaming_forecasting`.
```bash
mkdir ./data/streaming_forecasting

# training set
python tools/benchmark/argoverse_sf_creation.py --data_dir ./data/argoverse_tracking/train --output_dir ./data/streaming_forecasting --save_prefix eval_cat_train --hist_length 20 --fut_length 30 

# validation set
python tools/benchmark/argoverse_sf_creation.py --data_dir ./data/argoverse_tracking/val --output_dir ./data/streaming_forecasting --save_prefix eval_cat_val --hist_length 20 --fut_length 30
```

If you want any customization, please follow the template below.
```bash
# training set
python tools/benchmark/argoverse_sf_creation.py --data_dir $path_to_tracking_train --output_dir $path_to_save_streaming_benchmark --save_prefix eval_cat_train --hist_length $history_length_of_forecasting --fut_length $prediction_horizon 

# validation set
python tools/benchmark/argoverse_sf_creation.py --data_dir $path_to_tracking_val --output_dir $path_to_save_streaming_benchmark --save_prefix eval_cat_val --hist_length $history_length_of_forecasting --fut_length $prediction_horizon 
```

#### 2.3.2 Information File Creation

Please use our `./tools/benchmark/info_file.py` to create the information files for Argoverse-SF. We mimic the style in `mmdetection` and `mmdetection3d` in organizing the information needed for training, inference, and evaluation.

The command is as below, if you follow our instructions to prepare the paths of data. After running these scripts, you will see `infos_train.pkl` and `infos_val.pkl` in `./data/streaming_forecasting`.
```bash
# training set
python tools/benchmark/info_file.py --save_prefix infos_train --benchmark_file eval_cat_train.pkl --data_dir ./data/argoverse_tracking/train

# validation set
python tools/benchmark/info_file.py --save_prefix infos_val --benchmark_file eval_cat_val.pkl --data_dir ./data/argoverse_tracking/val
```

If you want any customization, please follow the template below.
```bash
# training set
python tools/benchmark/info_file.py --data_dir $path_to_tracking_train --output_dir $path_to_save_streaming_benchmark --save_prefix infos_train --benchmark_file eval_cat_train.pkl --hist_length $history_length_of_forecasting --fut_length $prediction_horizon  

# validation set
python tools/benchmark/info_file.py --data_dir $path_to_tracking_validation --output_dir $path_to_save_streaming_benchmark --save_prefix infos_val --benchmark_file eval_cat_val.pkl --hist_length $history_length_of_forecasting --fut_length $prediction_horizon  
```

</details>

### 2.4 Format of Results and Evaluation Protocol

<details>
<summary> Evaluation. Click to view details. </summary>

#### 2.4.1 Format of Results

To evaluate on Argoverse-SF, the result file is `.pkl` file compressing a python list, where each item in the list is the result of one frame. For reference, you can check our [VectorNet result file](https://www.dropbox.com/s/rr4gh7ezm0ooffp/vectornet_streaming_inference.pkl?dl=0) for a rough sense.

Specifically, you would use the information file generated before `infos_val.pkl`. With each item in `infos_val.pkl`, your result file should also be a list corresponding to each sample in `infos_val.pkl`. Please note that the trajectories are in the **world coordinate**.
```python
result = 
[
  # Result for sample[0] in infos_val.pkl 
  {
    ...
  },
  ...
  # Result for sample[i] in infos_val.pkl
  {
    # sequence name in Argoverse, 
    # you should directly copy from sample[i]['seq_name']
    'seq_name': str, 
    # city_name in Argoverse,
    # you should directly copy from sample[i]['city_name']
    'city_name': str,
    # frame number in the sequence,
    # you should directly copy from sample[i]['frame_index']
    'frame_index': int,
    # results of your predictions
    # dictonary with keys being every agent id specified in sample[i]['query_keys']
    'results': {
      # key0 in sample[i]['query_keys]
      sample[i]['query_keys'][0]: ...,
      ...
      # keyj in sample[i]['query_keys]
      sample[i]['query_keys'][j]: {'trajs': numpy array shaped [6, 30, 2], 'confidences': numpy array shaped [6]},
      ...
      sample[i]['query_keys'][-1]: ...
    }
  },
  ...
  # Result for sample[-1] in infos_val.pkl
  {
    ...
  },
]
```

#### 2.4.2 Evaluation

To evaluate the inference results, run the following command and you will find the `metrics.json` in you specified `$directory_to_save_metric_file`
```bash
python tools/evaluation.py --result_file $path_streaming_inference.pkl --metric_file_prefix $directory_to_save_metric_file
```

For example, I want to evaluate my `VectorNet` results and save to `./results/VectorNet`, then:
```bash
python tools/evaluation.py --result_file ./results/VectorNet/streaming_inference.pkl --metric_file_prefix ./results/VectorNet/
```

We structure the metric results in the following way:
* The items of `ade6`, `fde6`, and `mr6` means the minADE, minFDE and miss rate for `K=6`, where the forecaster predicts 6 trajectories for each agent.
* We divide 4 groups of agents to acquire the above metrics. The groups are organized according to moving (`move`) and static (`stay`) motion, and visible (`vis`) and occluded (`occ`) states.
* Finally, the items begin with "`ave-`" are the major metrics that come from averaging all the groups.

</details>


## 3. Getting Started with Streaming Algorithm (Predictive Streamer)

### 3.1 Streaming Inference on Argoverse-SF

<details>

<summary> Inference details. Click to view. </summary>

We use `tools/inference.py` to run streaming forecasting iteratively on every frame. Its template is as below. After running the command, a `streaming_inference.pkl` will appear at the directory of `$path_to_save_results`, which is the result file.
```bash
python tools/inference.py --config $config_path --weight_path $your_checkpoint --save_prefix $path_to_save_results
```

For example, if you use our provided `VectorNet` checkpoint and save the results to `./results/VectorNet`, you could simply run as below.
```bash
python tools/inference.py --config configs/streamer/config.yaml --weight_path ./ckpts/vectornet.ckpt --save_prefix ./results/VectorNet/
```

</details>

### 3.2 Pretraining on the Forecasting Split

Before deploying on the **streaming forecasting** setup, we leverage the forecasting split to pretrain a strong forecasting model. You can skip this step by **directly downloading our pretrained checkpoint.** [[link]](https://www.dropbox.com/s/lsrszkb9emzgy7d/vectornet.ckpt?dl=0)

<details>
<summary> Pretraining details. Click to view. </summary>

#### 3.2.1 Pretraining commands

If you have followed the previous steps, especially the paths to data. Training and evaluating VectorNet on Argoverse's forecasting training/validation sets are as simple as:
```bash
# training
python tools/pretrain/train_forecaster_vectornet.py --exp_name $your_experiment_name --model_save_dir $directory_to_save

# validation
python tools/pretrain/eval_forecaster_vectornet.py --weight_path $path_to_trained_model
```

For example, my command is as simple as:
```
# training
# use wandb for logging
python tools/pretrain/train_forecaster_vectornet.py --exp_name pretrain --model_save_dir ./results --wandb

# validation
python tools/pretrain/eval_forecaster_vectornet.py --weight_path vectornet.ckpt
```

The expected results of the validation process is similar to below. We focus on minADE, minFDE, and MR.
```
------------------------------------------------
Prediction Horizon : 30, Max #guesses (K): 6
------------------------------------------------
{'minADE': 0.7742552296892377, 'minFDE': 1.1925503502421884, 'MR': 0.12593737332792865, 'p-minADE': nan, 'p-minFDE': nan, 'p-MR': 0.8569636313846993, 'brier-minADE': 6.351643563315018, 'brier-minFDE': 6.769938683867964, 'DAC': 0.9879999324415611}
------------------------------------------------
------------------------------------------------
Prediction Horizon : 30, Max #guesses (K): 1
------------------------------------------------
{'minADE': 1.53590666902302, 'minFDE': 3.373533234667678, 'MR': 0.5553810295905959, 'p-minADE': 1.53590666902302, 'p-minFDE': 3.373533234667678, 'p-MR': 0.5553810295905959, 'brier-minADE': 1.53590666902302, 'brier-minFDE': 3.373533234667678, 'DAC': 0.9887515200648561}
------------------------------------------------
```

Suppose you need to customize the training process, such as path to data or optimization details, change the configuration files:
<details>
<summary> Customizing pretraining. Click to view </summary>

We provide configuration via `yaml`-based files.

* `./configs/forecaster/VectorNet.yaml` specify the sub-configuration files controlling the behaviors of data loading, model architecture, and optimization.

* Data loading:

```yaml
# batch size
batch_size: 32 
val_batch_size: 32 

# number of workers in the dataloader
workers: 4
val_workers: 4

# path to the forecasting data
train_dir: ./data/argoverse_forecasting/train/data/
val_dir: ./data/argoverse_forecasting/val/data/
train_map_dir: null
val_map_dir: null

# use all the training data
ratio: 1.0

# perception range [xxyy]
pred_range:
  - -100.0
  - 100.0
  - -100.0
  - 100.0
```

* Model architecture, please use our default VectorNet.

* Optimization:
```yaml
# beginning epoch
epoch: 0
optimizer: adam
# total epoch
num_epochs: 24
# frequency of saving checkpoints
save_freq: 1.0

# base learning rate
lr: 0.0005
weight_decay: 1.0e-4
# iterations for warmup
warmup_iters: 1000
warmup_ratio: 1.0e-3
grad_clip: 1.0e-1

# when to drop the learning rate
lr_decay_epoch:
  - 16
  - 20
# ratio of dropping learning rate
lr_decay_rate: 0.25
```

</details>


</details>

### 3.3 Finetuning and Training Predictive Streamers on Argoverse-SF