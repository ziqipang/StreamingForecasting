# Modified from https://github.com/argoverse/argoverse-api
# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
import math, numpy as np


LOW_PROB_THRESHOLD_FOR_METRICS = 0.05


def get_ade(forecasted_trajectory: np.ndarray, gt_trajectory: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Average Displacement Error.
    Args:
        forecasted_trajectory: Predicted trajectory with shape (pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (pred_len x 2)
        gt_masks: (pred_len)
    Returns:
        ade: Average Displacement Error
    """
    if np.sum(gt_mask) == 0:
        return -1

    pred_len = forecasted_trajectory.shape[0]
    ade = float(
        sum(
            math.sqrt(
                (forecasted_trajectory[i, 0] - gt_trajectory[i, 0]) ** 2
                + (forecasted_trajectory[i, 1] - gt_trajectory[i, 1]) ** 2
            ) * gt_mask[i]
            for i in range(pred_len)
        )
        / sum(gt_mask)
    )
    return ade


def get_fde(forecasted_trajectory: np.ndarray, gt_trajectory: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Final Displacement Error.
    Args:
        forecasted_trajectory: Predicted trajectory with shape (pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (pred_len x 2)
    Returns:
        fde: Final Displacement Error
    """
    if not gt_mask[-1]:
        return -1
    fde = math.sqrt(
        (forecasted_trajectory[-1, 0] - gt_trajectory[-1, 0]) ** 2
        + (forecasted_trajectory[-1, 1] - gt_trajectory[-1, 1]) ** 2
    )
    return fde


def get_displacement_metrics(
    forecasted_trajectories: np.ndarray,
    forecasted_confidences: np.ndarray,
    gt_trajectories: np.ndarray,
    gt_masks: np.ndarray,
    k: int,
    miss_threshold: float = 2.0):
    """
    Args:
        forecasted_trajectories: N * K * T * 2
        forecasted_confidences: N * K
        gt_trajectories: N * T * 2
        gt_masks: N * T
    """
    N, K, T, _ = forecasted_trajectories.shape
    min_ade, min_fde, n_misses, n_total_num = [], [], [], []

    for idx in range(N):
        cur_min_ade, cur_min_fde = 1e10, 1e10
        min_idx = 0

        # sort according to confidences
        if forecasted_confidences is not None:
            sorted_idx = np.argsort(
                [-x for x in forecasted_confidences[idx]], 
                kind="stable")
        else:
            sorted_idx = np.arange(K)
        pruned_probabilities = [forecasted_confidences[idx][t] for t in sorted_idx[:k]]
        prob_sum = sum(pruned_probabilities)
        pruned_probabilities = [p / prob_sum for p in pruned_probabilities]
        pruned_trajectories = [forecasted_trajectories[idx][t] for t in sorted_idx[:k]]

        # select best prediction
        for j in range(k):
            fde = get_fde(pruned_trajectories[j], gt_trajectories[idx], gt_masks[idx])
            if fde < cur_min_fde:
                min_idx = j
                cur_min_fde = fde
        
        cur_min_ade = get_ade(pruned_trajectories[min_idx], gt_trajectories[idx], gt_masks[idx])
        min_ade.append(cur_min_ade)
        min_fde.append(cur_min_fde)
        n_misses.append(cur_min_fde > miss_threshold)
        n_total_num.append(cur_min_fde > 0)
    
    min_ade = np.array(min_ade)
    min_fde = np.array(min_fde)
    misses = np.array(n_misses)
    total_num = np.array(n_total_num)

    if np.sum(total_num) == 0:
        return -1, -1, -1
    else:
        return np.average(min_ade[min_ade > 0]), np.average(min_fde[min_fde > 0]), np.sum(misses) / np.sum(total_num)

def compute_forecasting_metrics(
    forecasted_trajectories: np.ndarray,
    forecasted_confidences: np.ndarray,
    gt_trajectories: np.ndarray,
    gt_masks: np.ndarray,
    miss_threshold: float = 2.0):
    ade1, fde1, mr1 = get_displacement_metrics(
        forecasted_trajectories, forecasted_confidences, gt_trajectories, gt_masks, 1, miss_threshold)
    ade6, fde6, mr6 = get_displacement_metrics(
        forecasted_trajectories, forecasted_confidences, gt_trajectories, gt_masks, 6, miss_threshold)
    metrics = {
        'ade1': ade1, 'fde1': fde1, 'mr1': mr1,
        'ade6': ade6, 'fde6': fde6, 'mr6': mr6}
    return metrics