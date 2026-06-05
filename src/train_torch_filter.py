import os
import sys
import time
import math
import torch
import matplotlib.pyplot as plt
import numpy as np
from termcolor import cprint
import argparse
from functools import partial

# Import configuration presets
from experiment_configs import get_all_configs
from utils import prepare_data

# Import base TORCHIEKF class
from utils_torch_iekf_base import TORCHIEKF

# CPU-only mode


def load_model_for_config(config_name):
    """Load the appropriate model module based on config name"""
    configs = get_all_configs()
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    config = configs[config_name]
    model_module_name = config['model_file']
    
    # Import base TORCHIEKF class
    from utils_torch_iekf_base import TORCHIEKF
    
    # Import appropriate MesNet class based on config
    if model_module_name == 'utils_torch_filter_dynwinattn':
        from utils_torch_filter_dynwinattn import MesNet
        model_type = 'dynamic_window'
    elif model_module_name == 'utils_torch_filter_winattn':
        from utils_torch_filter_winattn import MesNet
        model_type = 'fixed_window'
    else:
        from utils_torch_filter import MesNet
        model_type = 'baseline'
    
    # Wire config params into MesNet constructor via functools.partial
    params = config['params']
    MesNetWithParams = partial(MesNet, **params)

    # Wrap MesNet to pass it to TORCHIEKF
    class ConfiguredTORCHIEKF(TORCHIEKF):
        def __init__(self, parameter_class=None):
            super().__init__(parameter_class=parameter_class, mes_net_class=MesNetWithParams)
    
    return ConfiguredTORCHIEKF, model_type, config


# Global variables (will be set by load_model_for_config)
MesNetClass = None
MODEL_TYPE = None
CURRENT_CONFIG = None

max_loss = 2e1
max_grad_norm = 1e0
min_lr = 1e-5
criterion = torch.nn.MSELoss(reduction="sum")

# Learning rates for different components
lr_initprocesscov_net = 1e-4
weight_decay_initprocesscov_net = 0e-8

# For fixed/dynamic window attention models
lr_mesnet = {
    'cov_net': 1e-4,
    'cov_lin': 1e-4,
}
weight_decay_mesnet = {
    'cov_net': 1e-8,
    'cov_lin': 1e-8,
}

# Additional learning rates for dynamic window predictor (if exists)
lr_window_predictor = 1e-4
weight_decay_window_predictor = 1e-8


def compute_delta_p(Rot, p):
    list_rpe = [[], [], []]  # [idx_0, idx_end, pose_delta_p]

    # sample at 1 Hz
    Rot = Rot[::10]
    p = p[::10]

    step_size = 10  # every second
    distances = np.zeros(p.shape[0])
    dp = p[1:] - p[:-1]  #  this must be ground truth
    distances[1:] = dp.norm(dim=1).cumsum(0).numpy()

    seq_lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    k_max = int(Rot.shape[0] / step_size) - 1

    for k in range(0, k_max):
        idx_0 = k * step_size
        for seq_length in seq_lengths:
            if seq_length + distances[idx_0] > distances[-1]:
                continue
            idx_shift = np.searchsorted(distances[idx_0:], distances[idx_0] + seq_length)
            idx_end = idx_0 + idx_shift

            list_rpe[0].append(idx_0)
            list_rpe[1].append(idx_end)

        idxs_0 = list_rpe[0]
        idxs_end = list_rpe[1]
        delta_p = Rot[idxs_0].transpose(-1, -2).matmul(
            ((p[idxs_end] - p[idxs_0]).float()).unsqueeze(-1)).squeeze()
        list_rpe[2] = delta_p
    return list_rpe


def get_config_suffix(config_name):
    """Get short suffix for config name to use in file names"""
    suffix_map = {
        'baseline': 'base',
        'fixed_small': 'fws',
        'fixed_standard': 'fwm',
        'fixed_large': 'fwl',
        'dynamic_conservative': 'dwc',
        'dynamic_standard': 'dws',
        'dynamic_aggressive': 'dwa',
        'dynamic_heads_1': 'dwh1',
        'dynamic_heads_8': 'dwh8',
    }
    return suffix_map.get(config_name, config_name[:3])


def monitor_window_statistics(iekf, epoch, path_temp):
    """Monitor and log window statistics for dynamic window attention model."""
    if hasattr(iekf.mes_net, 'dynamic_attention'):
        dyn_attn = iekf.mes_net.dynamic_attention
        cprint("Window stats (epoch {}): range [{}, {}], heads: {}, T={:.1f}, scale={:.1f}".format(
            epoch, dyn_attn.min_window, dyn_attn.max_window, dyn_attn.num_heads,
            dyn_attn.temperature, dyn_attn.suppression_scale), 'cyan')
        # Log window predictor grad status to verify it's learning
        wp = dyn_attn.window_predictor
        first_weight = wp.predictor[0].weight
        has_grad = first_weight.grad is not None
        grad_norm = first_weight.grad.norm().item() if has_grad else 0.0
        cprint("  WindowPredictor grad norm: {:.6f}  (has_grad: {})".format(grad_norm, has_grad), 'cyan')

def train_filter(args, dataset):
    iekf = prepare_filter(args, dataset)
    prepare_loss_data(args, dataset)
    save_iekf(args, iekf)
    optimizer = set_optimizer(iekf)
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loop(args, dataset, epoch, iekf, optimizer, args.seq_dim)
        save_iekf(args, iekf)
        
        # Monitor window statistics for dynamic window model
        if MODEL_TYPE == 'dynamic_window' and hasattr(iekf.mes_net, 'dynamic_attention'):
            monitor_window_statistics(iekf, epoch, args.path_temp)
        
        print("Amount of time spent for 1 epoch: {}s\n".format(int(time.time() - start_time)))
        import gc; gc.collect()
        start_time = time.time()


def prepare_filter(args, dataset):
    global MesNetClass, MODEL_TYPE, CURRENT_CONFIG
    
    # Load MesNet class based on config
    config_name = getattr(args, 'config', 'dynamic_standard')
    MesNetClass, MODEL_TYPE, CURRENT_CONFIG = load_model_for_config(config_name)
    
    # Create TORCHIEKF instance with proper parameter_class
    # MesNetClass is already ConfiguredTORCHIEKF (subclass of TORCHIEKF with MesNet wired in)
    iekf = MesNetClass(
        parameter_class=args.parameter_class,
    )

    # set dataset parameter (already set in __init__ if parameter_class is provided)
    # But we need to convert g to torch tensor if it's numpy
    if type(iekf.g).__module__ == np.__name__:
        iekf.g = torch.from_numpy(iekf.g).double()

    # load model
    if args.continue_training:
        iekf.load(args, dataset)
    iekf.train()
    
    # init u_loc and u_std
    iekf.get_normalize_u(dataset)
    
    cprint(f"\n{'='*80}", 'cyan')
    cprint(f"Configuration: {config_name}", 'cyan')
    cprint(f"Model Type: {MODEL_TYPE}", 'cyan')
    cprint(f"Description: {CURRENT_CONFIG['description']}", 'cyan')
    cprint(f"Parameters: {CURRENT_CONFIG['params']}", 'cyan')
    
    if MODEL_TYPE == 'dynamic_window':
        cprint("[OK] Dynamic Window Attention enabled", 'green')
        if hasattr(iekf.mes_net, 'dynamic_attention'):
            dyn_attn = iekf.mes_net.dynamic_attention
            cprint(f"  - Window range: [{dyn_attn.min_window}, {dyn_attn.max_window}]", 'cyan')
            cprint(f"  - Number of heads: {dyn_attn.num_heads}", 'cyan')
            cprint(f"  - Number of candidates: {dyn_attn.num_candidates}", 'cyan')
    elif MODEL_TYPE == 'fixed_window':
        cprint("[OK] Fixed Window Attention enabled", 'yellow')
        if hasattr(iekf.mes_net, 'window_attention'):
            win_attn = iekf.mes_net.window_attention
            cprint(f"  - Window size: {win_attn.window_size}", 'cyan')
            cprint(f"  - Number of heads: {win_attn.num_heads}", 'cyan')
    else:
        cprint("[OK] Baseline CNN model", 'red')
    
    cprint(f"{'='*80}\n", 'cyan')
    
    return iekf


def prepare_loss_data(args, dataset):
    # Add config suffix to file names
    config_suffix = get_config_suffix(getattr(args, 'config', 'dynamic_standard'))
    
    file_delta_p = os.path.join(args.path_temp, f'delta_p_{config_suffix}.p')
    if os.path.isfile(file_delta_p):
        mondict = dataset.load(file_delta_p)
        dataset.list_rpe = mondict['list_rpe']
        dataset.list_rpe_validation = mondict['list_rpe_validation']
        if set(dataset.datasets_train_filter.keys()) <= set(dataset.list_rpe.keys()): 
            return

    # prepare delta_p_gt
    list_rpe = {}
    for dataset_name, Ns in dataset.datasets_train_filter.items():
        t, ang_gt, p_gt, v_gt, u = prepare_data(args, dataset, dataset_name, 0)
        p_gt = p_gt.double()
        
        # 修复：如果 Ns[1] 为 None，使用实际数据长度
        N_end = Ns[1] if Ns[1] is not None else len(t)
        Rot_gt = torch.zeros(N_end, 3, 3)
        for k in range(N_end):
            ang_k = ang_gt[k]
            Rot_gt[k] = TORCHIEKF.from_rpy(ang_k[0], ang_k[1], ang_k[2]).double()
        list_rpe[dataset_name] = compute_delta_p(Rot_gt[:N_end], p_gt[:N_end])
        del Rot_gt

    list_rpe_validation = {}
    for dataset_name, Ns in dataset.datasets_validatation_filter.items():
        t, ang_gt, p_gt, v_gt, u = prepare_data(args, dataset, dataset_name, 0)
        p_gt = p_gt.double()
        
        # 修复：如果 Ns[1] 为 None，使用实际数据长度
        N_end = Ns[1] if Ns[1] is not None else len(t)
        Rot_gt = torch.zeros(N_end, 3, 3)
        for k in range(N_end):
            ang_k = ang_gt[k]
            Rot_gt[k] = TORCHIEKF.from_rpy(ang_k[0], ang_k[1], ang_k[2]).double()
        list_rpe_validation[dataset_name] = compute_delta_p(Rot_gt[:N_end], p_gt[:N_end])
        del Rot_gt
    
    dataset.list_rpe = {}
    for dataset_name, rpe in list_rpe.items():
        if len(rpe[0]) != 0:  # 修复：使用 != 而不是 is not
            dataset.list_rpe[dataset_name] = list_rpe[dataset_name]
        else:
            dataset.datasets_train_filter.pop(dataset_name)
            list_rpe.pop(dataset_name)
            cprint("%s has too much dirty data, it's removed from training list" % dataset_name, 'yellow')

    dataset.list_rpe_validation = {}
    for dataset_name, rpe in list_rpe_validation.items():
        if len(rpe[0]) != 0:  # 修复：使用 != 而不是 is not
            dataset.list_rpe_validation[dataset_name] = list_rpe_validation[dataset_name]
        else:
            dataset.datasets_validatation_filter.pop(dataset_name)
            list_rpe_validation.pop(dataset_name)
            cprint("%s has too much dirty data, it's removed from validation list" % dataset_name, 'yellow')
    mondict = {
        'list_rpe': list_rpe, 'list_rpe_validation': list_rpe_validation,
        }
    dataset.dump(mondict, file_delta_p)


def train_loop(args, dataset, epoch, iekf, optimizer, seq_dim):
    """训练循环，按照论文要求每批次采样 9 个 1 分钟序列
    
    论文要求 (C. Training):
    "we sample a batch of nine 1 min sequences, each sequence starts at a random arbitrary time"
    """
    import numpy as np
    
    # 从所有训练序列中随机采样 9 个（可重复采样）
    dataset_names = list(dataset.datasets_train_filter.keys())
    if len(dataset_names) == 0:
        cprint("No training data available!", 'red')
        return
    
    # 随机采样 9 个序列（允许重复）
    batch_size = 9
    sampled_datasets = np.random.choice(dataset_names, size=batch_size, replace=True)
    
    loss_train = 0
    optimizer.zero_grad()
    
    for i, dataset_name in enumerate(sampled_datasets):
        Ns = dataset.datasets_train_filter[dataset_name]
        t, ang_gt, p_gt, v_gt, u, N0 = prepare_data_filter(dataset, dataset_name, Ns,
                                                                  iekf, seq_dim)

        loss = mini_batch_step(dataset, dataset_name, iekf,
                               dataset.list_rpe[dataset_name], t, ang_gt, p_gt, v_gt, u, N0)

        if loss == -1 or torch.isnan(loss):  # fix: use == not is
            cprint("{} loss is invalid".format(i), 'yellow')
            continue
        elif loss > max_loss:
            cprint("{} loss is too high {:.5f}".format(i, loss), 'yellow')
            continue
        else:
            loss_train += loss

    if loss_train == 0:
        cprint("All losses are invalid in this batch", 'yellow')
        return
    loss_train.backward()

    # Debug: verify WindowPredictor gradient flows through soft mask
    if MODEL_TYPE == 'dynamic_window' and hasattr(iekf.mes_net, 'dynamic_attention'):
        wp = iekf.mes_net.dynamic_attention.window_predictor
        w = wp.predictor[0].weight
        if w.grad is not None:
            cprint("  [OK] WindowPredictor gradient: {:.6f}".format(w.grad.norm().item()), 'green')
        else:
            cprint("  [WARN] WindowPredictor gradient is None!", 'red')

    g_norm = torch.nn.utils.clip_grad_norm_(iekf.parameters(), max_grad_norm)
    g_norm_val = g_norm.item() if g_norm is not None else float('inf')
    if math.isnan(g_norm_val) or g_norm_val > 3*max_grad_norm:
        cprint("gradient norm: {:.5f}".format(g_norm_val), 'yellow')
        optimizer.zero_grad()

    else:
        optimizer.step()
        optimizer.zero_grad()
        cprint("gradient norm: {:.5f}".format(g_norm.item()))
    print('Train Epoch: {:2d} \tLoss: {:.5f}'.format(epoch, loss_train))
    return loss_train


def save_iekf(args, iekf):
    # Add config suffix to file name
    config_suffix = get_config_suffix(getattr(args, 'config', 'dynamic_standard'))
    file_name = os.path.join(args.path_temp, f"iekfnets_{config_suffix}.p")
    torch.save(iekf.state_dict(), file_name)
    print(f"The IEKF nets ({config_suffix}) are saved in the file " + file_name)


def mini_batch_step(dataset, dataset_name, iekf, list_rpe, t, ang_gt, p_gt, v_gt, u, N0):
    iekf.set_Q()
    measurements_covs = iekf.forward_nets(u)
    Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = iekf.run(t, u,measurements_covs,
                                                            v_gt, p_gt, t.shape[0],
                                                            ang_gt[0])
    delta_p, delta_p_gt = precompute_lost(Rot, p, list_rpe, N0)
    if delta_p is None:
        return -1
    loss = criterion(delta_p, delta_p_gt)

    # Soft regularization: gently penalize large windows to prevent collapse to max_window
    if MODEL_TYPE == 'dynamic_window' and hasattr(iekf, '_last_expected_windows'):
        ew = iekf._last_expected_windows
        dyn_attn = iekf.mes_net.dynamic_attention
        min_w, max_w = dyn_attn.min_window, dyn_attn.max_window
        reg = 1e-4 * ((ew - min_w) / (max_w - min_w)).mean()
        loss = loss + reg

    return loss


def set_optimizer(iekf):
    """设置优化器，配置不同网络层的学习率和权重衰减"""
    param_list = [{'params': iekf.initprocesscov_net.parameters(),
                           'lr': lr_initprocesscov_net,
                           'weight_decay': weight_decay_initprocesscov_net}]
    # Add ALL mes_net parameters (cov_net, cov_lin, fc1, window_attention, dynamic_attention, etc.)
    param_list.append({'params': iekf.mes_net.parameters(),
                       'lr': 1e-4,
                       'weight_decay': 1e-8})
    optimizer = torch.optim.Adam(param_list)
    return optimizer


def prepare_data_filter(dataset, dataset_name, Ns, iekf, seq_dim):
    """准备训练数据，包括数据提取、子采样和噪声添加"""
    # get data with trainable instant
    t, ang_gt, p_gt, v_gt, u = dataset.get_data(dataset_name)
    t = t[Ns[0]: Ns[1]]
    ang_gt = ang_gt[Ns[0]: Ns[1]]
    p_gt = p_gt[Ns[0]: Ns[1]] - p_gt[Ns[0]]
    v_gt = v_gt[Ns[0]: Ns[1]]
    u = u[Ns[0]: Ns[1]]

    # subsample data
    N0, N = get_start_and_end(seq_dim, u)
    t = t[N0: N].double()
    ang_gt = ang_gt[N0: N].double()
    p_gt = (p_gt[N0: N] - p_gt[N0]).double()
    v_gt = v_gt[N0: N].double()
    u = u[N0: N].double()

    # add noise
    if iekf.mes_net.training:
        u = dataset.add_noise(u)

    return t, ang_gt, p_gt, v_gt, u, N0


def get_start_and_end(seq_dim, u):
    """获取训练序列的起始和结束索引"""
    if seq_dim is None:
        N0 = 0
        N = u.shape[0]
    else:  # training sequence
        N0 = 10 * int(np.random.randint(0, (u.shape[0] - seq_dim)/10))
        N = N0 + seq_dim
    return N0, N


def precompute_lost(Rot, p, list_rpe, N0):
    """预计算相对位姿误差（RPE）用于损失计算"""
    device = Rot.device
    N = p.shape[0]
    Rot_10_Hz = Rot[::10]
    p_10_Hz = p[::10]
    idxs_0 = torch.Tensor(list_rpe[0]).clone().long().to(device) - int(N0 / 10)
    idxs_end = torch.Tensor(list_rpe[1]).clone().long().to(device) - int(N0 / 10)
    delta_p_gt = list_rpe[2].to(device)
    
    idxs = torch.ones(idxs_0.shape[0], device=device, dtype=torch.bool)
    idxs[idxs_0 < 0] = False
    idxs[idxs_end >= int(N / 10)] = False
    
    delta_p_gt = delta_p_gt[idxs]
    idxs_end_bis = idxs_end[idxs]
    idxs_0_bis = idxs_0[idxs]
    if len(idxs_0_bis) == 0:  # 修复：使用 == 而不是 is
        return None, None     
    else:
        delta_p = Rot_10_Hz[idxs_0_bis].transpose(-1, -2).matmul(
        (p_10_Hz[idxs_end_bis] - p_10_Hz[idxs_0_bis]).unsqueeze(-1)).squeeze()
        distance = delta_p_gt.norm(dim=1).unsqueeze(-1)
        return delta_p.double() / distance.double(), delta_p_gt.double() / distance.double() 


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AI-IMU-DR Training with Config Selection')
    
    # Configuration selection
    
    configs = get_all_configs()
    parser.add_argument('--config', type=str, default='dynamic_standard',
                        choices=list(configs.keys()),
                        help='Configuration preset to use (default: dynamic_standard)')
    
    # List available configs
    parser.add_argument('--list-configs', action='store_true',
                        help='List all available configurations and exit')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=400,
                        help='Number of training epochs (default: 400)')
    parser.add_argument('--seq-dim', type=int, default=6000,
                        help='Sequence dimension for training (default: 6000)')
    parser.add_argument('--no-continue', action='store_true',
                        help='Do not continue from previous checkpoint')
    
    # What to do
    parser.add_argument('--train', action='store_true',
                        help='Train the filter')
    parser.add_argument('--test', action='store_true',
                        help='Test the filter')
    parser.add_argument('--results', action='store_true',
                        help='Generate results')
    parser.add_argument('--read-data', action='store_true',
                        help='Read and preprocess data')
    
    # Paths
    parser.add_argument('--path-data-base', type=str, default=None,
                        help='Base path for KITTI data')
    parser.add_argument('--path-data-save', type=str, default='../data',
                        help='Path to save processed data')
    parser.add_argument('--path-results', type=str, default='../results',
                        help='Path to save results')
    parser.add_argument('--path-temp', type=str, default='../temp',
                        help='Path for temporary files')
    
    return parser.parse_args()
