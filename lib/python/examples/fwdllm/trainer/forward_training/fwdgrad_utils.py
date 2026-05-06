import torch
import math

from torch.nn import CrossEntropyLoss
from typing import Callable, Tuple
from torch.cuda.amp import autocast
import logging

logger = logging.getLogger(__name__)


def _get_loss(x: torch.Tensor, t: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """Compute cross-entropy loss.

    Args:
        x (torch.Tensor): Output of the model.
        t (torch.Tensor): Targets.
        num_classes (int, optional): Maximum number of classes. Defaults to 10.

    Returns:
        torch.Tensor: Cross-entropy loss.
    """
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(x.view(-1, num_classes), t.view(-1))
    return loss


def get_loss(
    model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor, num_classes: int = 10
) -> torch.Tensor:
    """Cross-entropy loss. Given a pytorch model, it computes the cross-entropy loss.

    Args:
        model (torch.nn.Module): PyTorch model.
        x (torch.Tensor): Input tensor for the PyTorch model.
        t (torch.Tensor): Targets.
        num_classes (int, optional): Maximum number of classes. Defaults to 10.

    Returns:
        torch.Tensor: Cross-entropy loss.
    """
    y = model(x)[0]
    return _get_loss(y, t, num_classes)


def functional_get_loss(
    params: Tuple[torch.nn.Parameter, ...],
    model: Callable[[Tuple[torch.nn.Parameter, ...], torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    num_classes: int,
    buffers: list,
) -> torch.Tensor:
    """Functional cross-entropy loss. Given a functional version of a pytorch model, which can be obtained with
    `fmodel, params = functorch.make_functional(model)`, it computes the cross-entropy loss.

    Args:
        params (Tuple[torch.nn.Parameter, ...]): Model parameters obtained by `fmodel, params = fc.make_functional(model)`.
        model (Callable[[Tuple[torch.nn.Parameter, ...], torch.Tensor], torch.Tensor]): Functional version of a pytorch model,
            obtained by fmodel, `params = fc.make_functional(model)`
        x (torch.Tensor): Input tensor for the PyTorch model.
        t (torch.Tensor): Targets.
        num_classes (int, optional): Maximum number of classes. Defaults to 10.
        buffers (list): Model buffers.

    Returns:
        torch.Tensor: Cross-entropy loss.
    """
    y = model(params, buffers, x)[0]
    return _get_loss(y, t, num_classes)


def calculate_jvp(func, params, v):
    """
    Calculations Jacobian-vector product using numerical differentiation
    """
    h = 0.01
    with torch.no_grad(), autocast():
        loss = func(tuple([params[i] - h * v[i] for i in range(len(params))]))
        terbulence_loss = func(
            tuple([params[i] + h * v[i] for i in range(len(params))])
        )
    avg_loss = (terbulence_loss + loss) / 2
    jvp = (terbulence_loss - loss) / (2 * h)
    return avg_loss, jvp


def calculate_jvp_after_actual_update(func, params, v, jvp_scalar):
    """
    Calculations Jacobian-vector product using numerical differentiation
    """
    h = 0.01 # learning rate factor
    with torch.no_grad(), autocast():
        loss = func(tuple([params[i] - h * jvp_scalar * v[i] for i in range(len(params))]))
    return loss

def calculate_jvp_before_actual_update(func, params):
    """
    Calculations Jacobian-vector product using numerical differentiation
    """
    with torch.no_grad(), autocast():
        loss = func(tuple([params[i] for i in range(len(params))]))
    return loss

# Might contain useful memory optimizations. Look at this only if you're running into a memory bottleneck & you need ideas
# def calculate_jvp_experiment(func, params, v):
#     """
#     This implementation is mathematically similar to the implementation provided by Vanilla FwdLLM with some memory optimizations
#     Calculations Jacobian-vector product using numerical differentiation
#     """
#     h = 0.01
#     # logger.info(f"[MEM] Before: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
#     with torch.no_grad(), autocast():
#         # logger.info(f"params[0].device = {params[0].device}, v[0].device = {v[0].device}")
#         device = torch.device("cuda:0")
#         params = [p.to(device) for p in params]
#         v = [vi.to(device) for vi in v]
#         loss = func(tuple([params[i] - h * v[i] for i in range(len(params))]))
#         torch.cuda.empty_cache()  # optional, but can help with fragmentation
#         # logger.info(f"[MEM] After loss: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
#         terbulence_loss = func(tuple([params[i] + h * v[i] for i in range(len(params))]))
#         torch.cuda.empty_cache()
#         # logger.info(f"[MEM] After turbulence loss: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

#     avg_loss = (terbulence_loss + loss) / 2
#     jvp = (terbulence_loss - loss) / (2 * h)
#     del loss, terbulence_loss
#     torch.cuda.empty_cache()
#     # logger.info(f"[MEM] After cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
#     return avg_loss, jvp


# Does not work for n == 1
def calculate_var(fwdgrad_list):
    n = len(fwdgrad_list)

    # 计算前一半tensor的平均值
    first_half_mean = torch.mean(torch.stack(fwdgrad_list[: n // 2]), dim=0)

    # 计算后一半tensor的平均值
    second_half_mean = torch.mean(torch.stack(fwdgrad_list[n // 2 :]), dim=0)

    # 计算两个平均值之间的方差
    var = torch.var(torch.stack([first_half_mean, second_half_mean]), dim=0).mean()

    return var

# Does not work for n == 1
def calculate_real_var(fwdgrad_list):
    n = len(fwdgrad_list)

    # 计算两个平均值之间的方差
    var = torch.var(torch.stack(fwdgrad_list), dim=0).mean()

    return var

def log_dist(name, tensor):
    """Helper to log distribution statistics of a tensor."""
    if tensor.numel() == 0:
        return
    
    # Flatten to ensure we are looking at the distribution of all scalar values
    flat = tensor.detach().float().view(-1)
    
    # Define percentiles to track
    q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to(flat.device)
    percentiles = torch.quantile(flat, q)
    
    logger.info(
        f"{name:15} | Mean: {flat.mean():.6f} | "
        f"P10: {percentiles[0]:.6f} | P50: {percentiles[2]:.6f} | P90: {percentiles[4]:.6f} | P95: {percentiles[5]:.6f} | P99: {percentiles[6]:.6f}"
    )

def calculate_snr(fwdgrad_list):
    """
    Calculates SNR using the variance of all individual updates.
    This factors in magnitude outliers and client-to-client disagreement.
    """
    n = len(fwdgrad_list)
    
    
    # Requirement: Need at least 2 updates to calculate variance
    if n < 2:
        return 0.0

    # 1. Stack all individual updates: shape (N, Parameters)
    all_grads_stacked = torch.stack(fwdgrad_list)
    
    # 2. Calculate the Global Mean (The Signal)
    global_mean = torch.mean(all_grads_stacked, dim=0)
    
    # 4. Actual Variance: Variance across all N updates
    # We calculate variance for each parameter (dim=0), 
    # then take the mean to get a single scalar representing total noise.
    actual_var = torch.var(all_grads_stacked, dim=0)

    logger.info(f"shape of actual_var: {actual_var.shape}")
    logger.info("--- Gradient Distribution Stats ---")
    logger.info(f"JVP of all updates so far {all_grads_stacked}")
    log_dist("Signal (Mean)", all_grads_stacked)
    log_dist("Signal^2", all_grads_stacked**2)
    # log_dist("Variance", actual_var)

    mean_of_mean = torch.mean(global_mean)
    mean_of_mean_2 = torch.mean(global_mean ** 2)
    mean_of_var = torch.mean(actual_var)

    logger.info(f"number of updates: {n} - mean_of_mean : {mean_of_mean} mean_of_mean_squared : {mean_of_mean_2} and  mean_of_var : {mean_of_var}")

    snr = torch.mean((global_mean ** 2) / (actual_var))

    return snr.item()

def calculate_snr_gradients(fwdgrad_list):
    """
    Calculates SNR using the variance of all individual updates.
    This factors in magnitude outliers and client-to-client disagreement.
    """
    n = len(fwdgrad_list)
    
    
    # Requirement: Need at least 2 updates to calculate variance
    if n < 2:
        return 0.0

    # 1. Stack all individual updates: shape (N, Parameters)
    all_grads_stacked = torch.stack(fwdgrad_list)
    
    # 2. Calculate the Global Mean (The Signal)
    global_mean = torch.mean(all_grads_stacked, dim=0)
    
    # 4. Actual Variance: Variance across all N updates
    # We calculate variance for each parameter (dim=0), 
    # then take the mean to get a single scalar representing total noise.
    actual_var = torch.var(all_grads_stacked, dim=0)

    logger.info(f"shape of actual_var: {actual_var.shape}")
    logger.info("--- Gradient Distribution Stats ---")
    logger.info(f"JVP of all updates so far {all_grads_stacked}")
    # log_dist("Signal (Mean)", all_grads_stacked)
    # log_dist("Signal^2", all_grads_stacked**2)
    # log_dist("Variance", actual_var)

    mean_of_mean = torch.mean(global_mean)
    mean_of_mean_2 = torch.mean(global_mean ** 2)
    mean_of_var = torch.mean(actual_var)
    snr = torch.mean((global_mean ** 2) / (actual_var))

    logger.info(f"number of gradient updates: {n} - mean_of_mean : {mean_of_mean} mean_of_mean_squared : {mean_of_mean_2} and  mean_of_var : {mean_of_var} and snr : {snr}")

    return snr.item()

def calculate_cv(fwdgrad_list):
    n = len(fwdgrad_list)
    
    # Does not work for n == 1 (need at least 2 to compute standard deviation)
    if n < 2:
        return 0.0

    # 将所有tensor堆叠在一起
    stacked_grads = torch.stack(fwdgrad_list)

    # 计算所有tensor在各个维度上的标准差 (Standard Deviation: sigma)
    std_dev = torch.std(stacked_grads, dim=0)

    # 计算所有tensor在各个维度上的平均值 (Mean: mu)
    mean_val = torch.mean(stacked_grads, dim=0)

    # 计算变异系数 CV = Std / |Mean|
    # Note: We use torch.abs() because means can be negative, and CV should be positive.
    # Added 1e-8 to prevent division by zero when the mean is exactly 0.
    cv_tensor = std_dev / (torch.abs(mean_val) + 1e-8)

    # 求所有维度的平均值，返回一个标量 (scalar)
    cv = cv_tensor.mean()

    return cv.item()


def calculate_cos_sim(A, target_grad, device):
    batch_size = 1000

    # 计算总批次数
    num_batches = math.ceil(A.size(0) / batch_size)

    # 创建一个空的结果张量
    result = torch.empty(A.size(0))

    # 逐批次计算余弦相似度
    for i in range(num_batches):
        # 获取当前批次的起始索引和结束索引
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, A.size(0))

        # 提取当前批次的向量
        # TODO: See why .to(device) was called here
        batch = A[start_idx:end_idx]  # .to(device)

        # 计算当前批次的余弦相似度
        similarity = torch.cosine_similarity(batch, target_grad, dim=-1)

        # 将结果保存到结果张量的对应位置
        result[start_idx:end_idx] = similarity

    return similarity
