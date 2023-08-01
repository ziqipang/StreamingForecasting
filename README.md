# [IROS 2023] Streaming Motion Forecasting for Autonomous Driving

## Introduction

This is the official code for our IROS 2023 paper: "Streaming Motion Forecasting for Autonomous Driving." We propose to view the motion forecasting from a **streaming** perspective, where the predictions are made on continuous frames, instead of the conventional **snapshot-based** forecasting.

If you find our code or paper useful, please cite by:
```Tex
@inproceedings{pang2023streaming,
  title={Streaming motion forecasting for autonomous driving},
  author={Pang, Ziqi and Ramanan, Deva and Mengtian, Li and Yu-Xiong, Wang},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2023}
}
```

## Getting Started

### Install `streaming_forecasting` toolkit

We recommend install it locally via:

```bash
pip install -e ./
```

### Data Preparation
<details>
<summary>Click to view the details</summary>

You will prepare the argoverse dataset, both tracking and forecasting splits included. **We recommend you putting the tracking and forecasting into separate directories.** For instance, I use directories `~/datasets/argoverse/tracking` and `~/datasets/argoverse/forecasting`. 

Remember to soft-link your data location to `./data/argoverse_forecasting` and `./data/argoverse_tracking`. The file structure would be similar to:
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

#### Tracking data

Argoverse-SF models streaming forecasting by re-purposing the tracking data from Argoverse. Please skip this step if you already have it.

* Download the tracking split from [Argoverse Link](https://www.argoverse.org/av1.html#download-link). You will see 4 `*.tar.gz` for the training set and 1 `*.tar.gz` for the validation set.
* Extract the data from compressed files locally. Take `tracking_val_v1.1.tar.gz` for example:
```bash
# Normal extraction
tar -xvf tracking_val_v1.1.tar.gz -C ./

# Exclude the images if you have limited disk space
tar -xvf tracking_val_v1.1.tar.gz --exclude="*.jpg" -C ./
``` 
* Move everything out of `argoverse_tracking/` and Merge the training files 
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

#### Forecasting data
In the pretraining step, we will use the forecasting data to train a snapshot-based forecasting model, just like normal forecasters on Argoverse.

* Download the forecasting split from [Argoverse Link](https://www.argoverse.org/av1.html#download-link). You will see 1 `*.tar.gz` for the training set and 1 `*.tar.gz` for the validation set.
* Extract the forecasting file locally. Take `forecasting_val_v1.1.tar.gz` for example, the sript is as below.

```bash
tar -xvf forecasting_val_v1.1.tar.gz
```

#### Install the `argoverse-api`

* Rigorously follow their [instructions](https://github.com/argoverse/argoverse-api#installation).
</details>

### Benchmark (Argoverse-SF) Creation

<details>
<summary>Click to view the details</summary>

We will walk you through:
* Generating the Argoverse-SF benchmark files for evaluation and visualization.
* Generating the information files for dataloading during training and inference. 

#### Benchmark Creation

Please use our `./tools/argoverse_sf_creation.py` to create the Argoverse-SF benchmark, which supports evaluation. The commands is as below, if you follow our instructions on softlinking the argoverse datasets to `./data` in **Dataset Preparation**. After this step, you will see `eval_cat_val.pkl` and `eval_cat_train.pkl` popping up in `./data/streaming_forecasting`.
```bash
mkdir ./data/streaming_forecasting

# training set
python tools/argoverse_sf_creation.py --data_dir ./data/argoverse_tracking/train --output_dir ./data/streaming_forecasting --save_prefix eval_cat_train --hist_length 20 --fut_length 30 

# validation set
python tools/argoverse_sf_creation.py --data_dir ./data/argoverse_tracking/val --output_dir ./data/streaming_forecasting --save_prefix eval_cat_val --hist_length 20 --fut_length 30
```

If you want any customization, please follow the template below.
```bash
# training set
python tools/argoverse_sf_creation.py --data_dir $path_to_tracking_train --output_dir $path_to_save_streaming_benchmark --save_prefix eval_cat_train --hist_length $history_length_of_forecasting --fut_length $prediction_horizon 

# validation set
python tools/argoverse_sf_creation.py --data_dir $path_to_tracking_val --output_dir $path_to_save_streaming_benchmark --save_prefix eval_cat_val --hist_length $history_length_of_forecasting --fut_length $prediction_horizon 
```

</details>