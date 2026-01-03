"""
评估指标模块：MSLE, ACP, NLL, PICP, MPIW等
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ..config import LOG_OFFSET


def compute_msle(y_true, y_pred):
    """计算MSLE (Mean Squared Logarithmic Error)

    MSLE = mean((log(y_true + c) - log(y_pred + c))^2)

    适合长尾分布数据，关注相对误差
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.clip(y_pred, 0, None)  # 确保非负

    log_true = np.log(y_true + LOG_OFFSET)
    log_pred = np.log(y_pred + LOG_OFFSET)

    return np.mean((log_true - log_pred) ** 2)


def compute_acp(y_true, y_pred, alpha=0.2, delta=5):
    """计算ACP (Accuracy within Coverage Percentage)

    如果 |y_pred - y_true| <= max(alpha * y_true, delta)，则视为"命中"
    返回命中率

    参数:
        y_true: 真实值
        y_pred: 预测值
        alpha: 相对容忍度 (默认20%)
        delta: 绝对容忍度 (默认5)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tolerance = np.maximum(alpha * y_true, delta)
    within_tolerance = np.abs(y_pred - y_true) <= tolerance

    return np.mean(within_tolerance)


def compute_log_nll(y_true, y_pred, y_std):
    """计算对数空间的NLL (Negative Log Likelihood)

    L = 0.5 * log(σ²) + (log(y+c) - log(μ+c))² / (2σ²)

    参数:
        y_true: 真实值（原始空间）
        y_pred: 预测均值（原始空间）
        y_std: 预测标准差（log空间）
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_std = np.array(y_std)

    # 确保数值稳定
    y_pred = np.clip(y_pred, 0, None)
    y_std = np.clip(y_std, 1e-4, None)

    # 转换到log空间
    log_true = np.log(y_true + LOG_OFFSET)
    log_pred = np.log(y_pred + LOG_OFFSET)

    # 计算NLL
    nll = 0.5 * np.log(y_std ** 2) + ((log_true - log_pred) ** 2) / (2 * y_std ** 2)

    return np.mean(nll)


def compute_picp(y_true, y_pred, y_std, confidence=0.95):
    """计算PICP (Prediction Interval Coverage Probability)

    计算真实值落在置信区间内的比例
    置信区间: [μ - z*σ, μ + z*σ]

    参数:
        y_true: 真实值
        y_pred: 预测均值
        y_std: 预测标准差
        confidence: 置信水平 (默认95%)

    返回:
        覆盖率 (理想情况下应接近confidence)
    """
    from scipy import stats

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_std = np.array(y_std)

    # 计算z分数
    z = stats.norm.ppf((1 + confidence) / 2)

    # 在log空间计算置信区间
    log_true = np.log(y_true + LOG_OFFSET)
    log_pred = np.log(y_pred + LOG_OFFSET)

    lower = log_pred - z * y_std
    upper = log_pred + z * y_std

    # 计算覆盖率
    within_interval = (log_true >= lower) & (log_true <= upper)

    return np.mean(within_interval)


def compute_mpiw(y_pred, y_std, confidence=0.95):
    """计算MPIW (Mean Prediction Interval Width)

    将对数尺度的置信区间映射回原始尺度后计算宽度

    对数空间置信区间: [log(μ+c) - z*σ, log(μ+c) + z*σ]
    原始空间置信区间: [exp(lower) - c, exp(upper) - c]
    原始空间宽度: exp(upper) - exp(lower)

    参数:
        y_pred: 预测均值（原始空间）
        y_std: 预测标准差（log空间）
        confidence: 置信水平

    返回:
        平均置信区间宽度（原始空间）
    """
    from scipy import stats

    y_pred = np.array(y_pred)
    y_std = np.array(y_std)

    # 确保数值稳定
    y_pred = np.clip(y_pred, 0, None)
    # 限制标准差范围，避免极端值导致溢出（log空间标准差超过10已经是极端情况）
    y_std = np.clip(y_std, 1e-4, 10.0)

    # z分数
    z = stats.norm.ppf((1 + confidence) / 2)

    # 在log空间计算置信区间
    log_pred = np.log(y_pred + LOG_OFFSET)
    lower_log = log_pred - z * y_std
    upper_log = log_pred + z * y_std

    # 限制upper_log避免exp溢出（exp(50) ≈ 5e21，已经足够大）
    upper_log = np.clip(upper_log, None, 50.0)

    # 映射回原始空间
    lower_orig = np.exp(lower_log) - LOG_OFFSET
    upper_orig = np.exp(upper_log) - LOG_OFFSET

    # 确保下界非负
    lower_orig = np.clip(lower_orig, 0, None)

    # 计算原始空间的区间宽度
    width = upper_orig - lower_orig

    return np.mean(width)


def evaluate(y_true, y_pred, prefix='', y_std=None):
    """综合评估函数

    计算所有评估指标

    参数:
        y_true: 真实值
        y_pred: 预测值
        prefix: 指标名称前缀（如 'train_', 'val_', 'test_'）
        y_std: 预测标准差（可选，用于概率预测模型）

    返回:
        包含所有指标的字典
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {}

    # 基础指标
    # metrics[f'{prefix}MSE'] = mean_squared_error(y_true, y_pred)
    # metrics[f'{prefix}RMSE'] = np.sqrt(metrics[f'{prefix}MSE'])
    metrics[f'{prefix}MAE'] = mean_absolute_error(y_true, y_pred)
    # metrics[f'{prefix}R2'] = r2_score(y_true, y_pred)
    metrics[f'{prefix}MSLE'] = compute_msle(y_true, y_pred)

    # ACP指标（多个阈值）
    metrics[f'{prefix}ACP@20%'] = compute_acp(y_true, y_pred, alpha=0.2, delta=5)
    metrics[f'{prefix}ACP@50%'] = compute_acp(y_true, y_pred, alpha=0.5, delta=5)

    # 如果有不确定性估计
    if y_std is not None:
        y_std = np.array(y_std)
        metrics[f'{prefix}LogNLL'] = compute_log_nll(y_true, y_pred, y_std)
        metrics[f'{prefix}PICP@95%'] = compute_picp(y_true, y_pred, y_std, confidence=0.95)
        metrics[f'{prefix}MPIW@95%'] = compute_mpiw(y_pred, y_std, confidence=0.95)

    return metrics
