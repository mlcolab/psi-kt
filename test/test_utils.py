def test_one_step_and_multi_step_log_probability(
    q_dist,
    transition_h,
    transition_b,
    transition_r,
    p0_mean,
    p0_log_var,
):
    import torch
    from torch import nn, distributions
    from torch.nn import functional as F
    future_tensor = q_dist.rsample((100,))[:,:,:,1:] # [n, bs, 1, time-1, dim_s]
    
    ps_mean = q_dist.mean[:,:,:-1] @ transition_h + transition_b # [bs, 1, time-1, dim_s]
    cov_mat = torch.diag_embed(torch.exp(transition_r)) + EPS
    ps_var = transition_h @ q_dist.covariance_matrix[:,:,:-1] @ transition_h.transpose(-1, -2) + cov_mat # [bs, 1, time-1, dim_s, dim_s]
    dist_one_step = distributions.multivariate_normal.MultivariateNormal(
            loc=ps_mean, 
            covariance_matrix=ps_var
            )
    
    logprob_one_step = dist_one_step.log_prob(future_tensor) # [n, bs, 1, time-1]
    
    p0_var = torch.diag_embed(torch.exp(p0_log_var)) + EPS
    time_step = future_tensor.shape[-2]
    prev_var = p0_var
    prev_mean = p0_mean
    logprob_multi_step = []
    for i in range(time_step):
        pnext_mean = prev_mean @ transition_h + transition_b # [bs, 1, dim_s]
        pnext_var = transition_h @ prev_var @ transition_h.transpose(-1, -2) + cov_mat # [bs, 1, dim_s, dim_s]
        dist_multi_step = distributions.multivariate_normal.MultivariateNormal(
            loc=pnext_mean,
            covariance_matrix=pnext_var,
            )
        logprob = dist_multi_step.log_prob(future_tensor[:, :, :, i]) # [n, bs, 1]
        logprob_multi_step.append(logprob)
    logprob_multi_step = torch.stack(logprob_multi_step, dim=-1) # [n, bs, 1, time-1]
    
    logprob_one_step = logprob_one_step.mean(dim=0) # [bs, 1, time-1]
    logprob_multi_step = logprob_multi_step.mean(dim=0) # [bs, 1, time-1]
    
    
def test_pred_evaluate_method():
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
    from baseline.BaseModel import pred_evaluate_method
    
    # Generate random predictions and true labels
    y_pred = np.random.rand(100)
    y_true = np.random.randint(0, 2, size=100)

    # Define the metrics to evaluate
    metrics = ['rmse', 'mae', 'auc', 'f1', 'accuracy', 'precision', 'recall']

    # Calculate the expected results
    expected_results = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred > 0.5),
        'accuracy': accuracy_score(y_true, y_pred > 0.5),
        'precision': precision_score(y_true, y_pred > 0.5),
        'recall': recall_score(y_true, y_pred > 0.5)
    }

    # Evaluate the predictions using the optimized function
    evaluations = pred_evaluate_method(y_pred, y_true, metrics)

    # Compare the actual and expected results
    for metric in metrics:
        assert np.isclose(evaluations[metric], expected_results[metric]), f"Failed for {metric}: expected {expected_results[metric]}, but got {evaluations[metric]}"


###########################################################################

import torch
from torch.testing import assert_allclose

def test_find_whole_stats():
    # Create test inputs
    all_feature = torch.randn(2, 1, 4, 3)
    t = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    items = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]])
    num_node = 3
    
    # Compute expected output
    expected_stats = torch.zeros((2, num_node, 4, 3))
    expected_last_time = torch.zeros((2, num_node, 5))
    expected_last_time[:, items[:,0], 1] = t[:, 0].float()
    for i in range(1, 4):
        cur_item = items[:, i]
        cur_feat = all_feature[:, 0, i]
        expected_stats[:, :, i] = expected_stats[:, :, i-1] + expected_stats[:, :, i-1]
        expected_stats[:, cur_item, i] = cur_feat
        expected_last_time[:, :, i+1] = expected_last_time[:, :, i] + expected_last_time[:, :, i]
        expected_last_time[:, cur_item, i+1] = t[:, i].float()
    
    # Compute actual output
    # stats, last_time = _find_whole_stats(all_feature, t, items, num_node)
    
    # Check that the outputs are close
    assert_allclose(stats, expected_stats)
    assert_allclose(last_time, expected_last_time)
