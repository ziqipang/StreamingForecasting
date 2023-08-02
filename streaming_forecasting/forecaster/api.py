""" Inference API
"""
import numpy as np, os, pickle, torch
from .builder import build_model, build_dataset
from .dataset import ref_copy, collate_fn


class ForecasterAPI:
    def __init__(self, dataset, model, map_reader, device) -> None:
        self.dataset = dataset
        self.model = model.to(device)
        self.map_reader = map_reader
        self.device = device
        return
    
    def inference(self, trajs, steps, timestamps, frame_num, city, idx=-1, preprocessed_map_name=None):
        # load data
        data = {
            'idx': idx, 'city': city, 'trajs': trajs,
            'steps': steps, 'timestamps': timestamps,
            'frame_num': frame_num
        }
        network_data = self.load_data(data, preprocessed_map_name)
        network_data = collate_fn([network_data])

        # infer with models
        results = self.model(network_data, self.device)
        
        # format the results
        preds = [x.detach().cpu().numpy() for x in results['prediction']]
        preds = np.concatenate(preds, axis=0)
        confidences = [x.detach().cpu().numpy() for x in results['confidence']]
        confidences = np.concatenate(confidences, axis=0)
        indexes = results['indexes'][0].detach().cpu().numpy()
        return preds, confidences, indexes
    
    def load_data(self, data, preprocessed_map_path=None):
        feat_data = self.dataset.get_obj_feats(data, self.dataset.config['pred_range'], fut_len=0)
        
        if preprocessed_map_path is None:
            # process graph
            graph = self.map_reader.read_map_from_raw(feat_data, feat_data['city'])
        else:
            # load preprocess graph
            file_path = preprocessed_map_path
            preprocessed_graph = pickle.load(open(file_path, 'rb'))
            graph = self.map_reader.read_map_from_preprocessed(preprocessed_graph, feat_data, feat_data['city'])
        feat_data['graph'] = graph

        # copy the needed information
        result = dict()
        for key in ['idx', 'city', 'norm_center', 'indexes', 'theta', 'norm_rot', 'actors', 'ctrs', 'graph']:
            if key in data:
                result[key] = ref_copy(data[key])
        return result

    def preprocess_map(self, data, preprocessed_map_path=None):
        # load raw map
        feat_data = self.dataset.get_obj_feats(data, self.dataset.config['pred_range'])
        graph = self.map_reader.read_map_from_raw(feat_data, feat_data['city'])
        
        # save
        file_path = preprocessed_map_path
        pickle.dump(graph, open(file_path, 'wb'))


def api_builder(model_name, dataset_config, model_config, weight_path, device='cuda:0') -> ProphetAPI:
    model = build_model(model_name, model_config)(model_config)
    dataset, _ = build_dataset(model_name, dataset_config)
    dataset = dataset(dataset_config['val_dir'], dataset_config, map_preprocess=None)

    ckpt = torch.load(weight_path, map_location=lambda storage, loc: storage)
    load_pretrain(model, ckpt['state_dict'])

    map_reader = dataset.map_reader
    return ForecasterAPI(dataset, model, map_reader, device)


def load_pretrain(net, pretrain_dict):
    state_dict = net.state_dict()
    for key in pretrain_dict.keys():
        if key in state_dict and (pretrain_dict[key].size() == state_dict[key].size()):
            value = pretrain_dict[key]
            if not isinstance(value, torch.Tensor):
                value = value.data
            state_dict[key] = value
    net.load_state_dict(state_dict)