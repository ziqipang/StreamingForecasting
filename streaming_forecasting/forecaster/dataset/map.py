''' Read the map information in the form of subgraphs
'''
import numpy as np
import torch
import os
import copy
from argoverse.map_representation.map_api import ArgoverseMap
import pdb


class VecMapReader:
    def __init__(self, am: ArgoverseMap, config):
        self.config = config
        self.am = am
        return
    
    def read_map_from_raw(self, agt_data, city):
        ''' Function for reading the map
        '''
        # ========== Read the raw connections of maps ========== #
        norm_center, norm_rot = agt_data['norm_center'], agt_data['norm_rot']
        graph = self.get_lane_data(norm_center, city)

        # ========== Translate ids to idcs ========== # 
        # result = self.map_preprocess(norm_center, graph, city)

        # ========== Normalize the centers and segs of the maps ========== #
        norm_graph = self.normalize_map(graph, norm_center, norm_rot)
        
        return norm_graph
    
    def read_map_from_preprocessed(self, graph, agt_data, city):
        # ========== Get the normalization parameters ========== #
        norm_center, norm_rot = agt_data['norm_center'], agt_data['norm_rot']
        
        # ========== Translate ids to idcs ========== # 
        # result = self.map_preprocess(norm_center, graph, city)

        # ========== Normalize the centers and segs of the maps ========== #
        norm_graph = self.normalize_map(graph, norm_center, norm_rot)
        
        return norm_graph
    
    def get_lane_data(self, norm_center, city):
        ''' Get the subgraph information, used for preprocessing
        '''
        # ========== Access the lanes in the range ========== #
        x_min, x_max, y_min, y_max = self.config['pred_range']
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        lane_ids = self.am.get_lane_ids_in_xy_bbox(norm_center[0], norm_center[1], city, radius)
        lane_ids = sorted(copy.deepcopy(lane_ids))
        
        lanes = dict()
        for lane_id in lane_ids:
            centerline = self.am.city_lane_centerlines_dict[city][lane_id].centerline
            displacement = centerline - norm_center
            x, y = displacement[:, 0], displacement[:, 1]
            if x.min() < x_min or x.max() > x_max or y.min() < y_min or y.max() > y_max:
                continue
            else:
                lanes[lane_id] = copy.deepcopy(self.am.city_lane_centerlines_dict[city][lane_id])
        lane_ids = list(lanes.keys())
        
        # ========== Get the lane information ========== #
        centerlines, seg_nums, turns, controls, intersections = list(), list(), list(), list(), list()
        for lane_id in lane_ids:
            lane = lanes[lane_id]
            centerline = lane.centerline[:, :2]
            
            if lane.turn_direction == 'LEFT':
                turn_dir = 1
            elif lane.turn_direction == 'RIGHT':
                turn_dir = 2
            else:
                turn_dir = 0
            
            if lane.has_traffic_control:
                control = 1
            else:
                control = 0
            
            if lane.is_intersection:
                is_intersection = 1
            else:
                is_intersection = 0
            
            centerlines.append(centerline)
            seg_nums.append(centerline.shape[0])
            turns.append(turn_dir)
            controls.append(control)
            intersections.append(is_intersection)
                
        # ========== Compensate for the lanes with less than 10 segments ========== #
        for i, _ in enumerate(lane_ids):
            if seg_nums[i] < 10:
                # N * 2 --> 10 * 2
                centerline = centerlines[i]
                centerline = np.pad(centerline, ((0, 10 - seg_nums[i]), (0, 0)), 'edge')
                centerlines[i] = centerline
                seg_nums[i] = 10
        
        # ========== Save the graph ========== #
        centerlines = np.array(centerlines) # lane_num * 10 * 2
        turns, intersections, controls = np.array(turns), np.array(intersections), np.array(controls)
        graph = {
            'centerlines': centerlines, 'seg_nums': seg_nums,
            'turns': turns, 'intersections': intersections,
            'controls': controls, 'lane_ids': lane_ids
        }
        return graph
    
    def normalize_map(self, graph, norm_center, norm_rot):
        ''' Normalize the map features/data.
            Args:
                graph: Dict, result from get_lane_data
                norm_center: [x, y]
                norm_rot: yaw rotation matrix
            Return:
                graph data, with normalized features and centers.
        '''
        result = copy.deepcopy(graph)
        
        # normalization only changes centerlines
        centerlines = result['centerlines']
        if len(centerlines) == 0:
            result['centerlines'] = centerlines.reshape((0, 10, 2))
            return result

        num, obs_len = centerlines.shape[:2]
        centerlines -= norm_center
        centerlines = np.matmul(norm_rot, centerlines.reshape((-1, 2)).T).T.reshape((num, obs_len, 2))
        result['centerlines'] = centerlines.astype(np.float32)
        return result
    
    def map_preprocess(self, norm_center, graph, city):
        ''' Process the normed connection graph.
            1. Selecting the lane ids lying inside the range 
        '''
        # ========== Access the lanes in the range ========== #
        # Range is smaller than preprocessing.
        x_min, x_max, y_min, y_max = self.config['pred_range']
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        queried_lane_ids = self.am.get_lane_ids_in_xy_bbox(norm_center[0], norm_center[1], city, radius)
        queried_lane_ids = sorted(copy.deepcopy(queried_lane_ids))
        lane_ids = list()
        for lane_id in queried_lane_ids:
            centerline = self.am.city_lane_centerlines_dict[city][lane_id].centerline
            displacement = centerline - norm_center
            x, y = displacement[:, 0], displacement[:, 1]
            if x.min() < x_min or x.max() > x_max or y.min() < y_min or y.max() > y_max:
                continue
            else:
                lane_ids.append(lane_id)
        
        # ========== Select the valid idcs ========== #
        valid_idcs = [i for i, id in enumerate(graph['lane_ids']) if id in lane_ids]
        valid_idcs = np.asarray(valid_idcs).astype(np.int32)
        
        centerlines = graph['centerlines'][valid_idcs].astype(np.float32)
        controls = graph['controls'][valid_idcs].astype(np.float32)
        intersections = graph['intersections'][valid_idcs].astype(np.float32)
        turns = graph['turns'][valid_idcs].astype(np.float32)
        
        result = {
            'centerlines': centerlines, 'controls': controls,
            'intersections': intersections, 'turns': turns, 'lane_ids': lane_ids
        }
        return result