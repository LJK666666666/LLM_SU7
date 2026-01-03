"""
数据加载模块：加载训练/验证/测试数据及外部特征
"""
import pandas as pd
from pathlib import Path

from ..config import (
    DATA_DIR, TRAIN_FILE, VAL_FILE, TEST_FILE,
    TRAIN_PKL, VAL_PKL, TEST_PKL
)


def load_data(use_pkl=True):
    """加载训练/验证/测试数据

    参数:
        use_pkl: 是否使用pickle格式（更快）

    返回:
        train_df, val_df, test_df
    """
    if use_pkl:
        train_path = DATA_DIR / TRAIN_PKL
        val_path = DATA_DIR / VAL_PKL
        test_path = DATA_DIR / TEST_PKL

        if train_path.exists() and val_path.exists() and test_path.exists():
            train_df = pd.read_pickle(train_path)
            val_df = pd.read_pickle(val_path)
            test_df = pd.read_pickle(test_path)
            return train_df, val_df, test_df

    # 回退到CSV
    train_df = pd.read_csv(DATA_DIR / TRAIN_FILE)
    val_df = pd.read_csv(DATA_DIR / VAL_FILE)
    test_df = pd.read_csv(DATA_DIR / TEST_FILE)

    return train_df, val_df, test_df


def load_lda_features():
    """加载LDA主题特征

    返回:
        train_lda, val_lda, test_lda: 包含'序号'和'主题'列的DataFrame
    """
    train_lda = pd.read_pickle(DATA_DIR / 'train_lda.pkl')
    val_lda = pd.read_pickle(DATA_DIR / 'val_lda.pkl')
    test_lda = pd.read_pickle(DATA_DIR / 'test_lda.pkl')

    return train_lda, val_lda, test_lda


def load_density_features(method='minhash'):
    """加载时间密度特征

    参数:
        method: 'bge' 或 'minhash'

    返回:
        train_density, val_density, test_density: 包含时间密度特征的DataFrame
    """
    suffix = f'_time_density_{method}.pkl'

    train_density = pd.read_pickle(DATA_DIR / f'train{suffix}')
    val_density = pd.read_pickle(DATA_DIR / f'val{suffix}')
    test_density = pd.read_pickle(DATA_DIR / f'test{suffix}')

    return train_density, val_density, test_density
