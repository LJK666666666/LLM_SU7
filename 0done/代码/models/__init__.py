"""
模型模块：提供各类预测模型

包含:
- MODEL_REGISTRY: 模型注册表
- register_model: 模型注册装饰器
- 传统机器学习模型: Ridge, Lasso, RF, GBDT, XGBoost, LightGBM, NGBoost
- 深度学习模型: BGE_NN, BGE_Mini
"""

from .traditional import (
    RidgeModel,
    LassoModel,
    RandomForestModel,
    GBDTModel,
)
from .bge_nn import BGENNModel, BGEMiniModel

# 模型注册表
MODEL_REGISTRY = {}


def register_model(name):
    """模型注册装饰器"""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


# 注册传统模型
MODEL_REGISTRY['ridge'] = RidgeModel
MODEL_REGISTRY['lasso'] = LassoModel
MODEL_REGISTRY['rf'] = RandomForestModel
MODEL_REGISTRY['gbdt'] = GBDTModel

# 尝试注册XGBoost
try:
    from .traditional import XGBoostModel
    MODEL_REGISTRY['xgboost'] = XGBoostModel
except ImportError:
    pass

# 尝试注册LightGBM
try:
    from .traditional import LightGBMModel
    MODEL_REGISTRY['lightgbm'] = LightGBMModel
except ImportError:
    pass

# 尝试注册NGBoost
try:
    from .traditional import NGBoostModel
    MODEL_REGISTRY['ngboost'] = NGBoostModel
except ImportError:
    pass

# 注册BGE神经网络模型
MODEL_REGISTRY['bge_nn'] = BGENNModel
MODEL_REGISTRY['bge_mini'] = BGEMiniModel


# 辅助函数
def log_transform(y):
    """对数变换"""
    import numpy as np
    from ..config import LOG_OFFSET
    return np.log(y + LOG_OFFSET)


def inverse_log_transform(y_log):
    """逆对数变换"""
    import numpy as np
    from ..config import LOG_OFFSET
    return np.exp(y_log) - LOG_OFFSET


__all__ = [
    'MODEL_REGISTRY',
    'register_model',
    'log_transform',
    'inverse_log_transform',
    'RidgeModel',
    'LassoModel',
    'RandomForestModel',
    'GBDTModel',
    'BGENNModel',
    'BGEMiniModel',
]
