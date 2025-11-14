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
    num_classes: int = 10,
    buffers=None,
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
        loss = func(tuple([params[i]-h*v[i] for i in range(len(params))]))
        terbulence_loss = func(tuple([params[i]+h*v[i] for i in range(len(params))]))
    avg_loss = (terbulence_loss + loss)/2
    jvp = (terbulence_loss - loss)/(2*h)
    return avg_loss, jvp

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
