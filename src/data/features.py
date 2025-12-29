"""
特征提取模块：从原始数据提取各类特征
"""
import re
import numpy as np
import pandas as pd

from ..config import (
    FEATURE_GROUPS, BASE_FEATURE_COLS, TEXT_FEATURE_COLS,
    LDA_FEATURE_COLS, DENSITY_FEATURE_COLS, XIAOMI_KEYWORDS
)


def extract_time_features(df):
    """提取时间特征

    从'发布时间'列提取：发布小时、发布星期、是否工作日
    """
    df = df.copy()
    df['发布时间'] = pd.to_datetime(df['发布时间'])
    df['发布小时'] = df['发布时间'].dt.hour
    df['发布星期'] = df['发布时间'].dt.dayofweek
    df['是否工作日'] = (df['发布星期'] < 5).astype(int)
    return df


def extract_text_features(df):
    """提取文本特征

    从'评论文案'列提取：
    - 评论长度
    - 感叹号数
    - 问号数
    - 表情数
    - 话题标签有无
    - 小米相关词数
    """
    df = df.copy()

    # 评论长度
    df['评论长度'] = df['评论文案'].fillna('').apply(len)

    # 感叹号数
    df['感叹号数'] = df['评论文案'].fillna('').apply(lambda x: x.count('！') + x.count('!'))

    # 问号数
    df['问号数'] = df['评论文案'].fillna('').apply(lambda x: x.count('？') + x.count('?'))

    # 表情数（匹配[xxx]格式的表情）
    emoji_pattern = re.compile(r'\[.+?\]')
    df['表情数'] = df['评论文案'].fillna('').apply(lambda x: len(emoji_pattern.findall(x)))

    # 话题标签有无
    topic_pattern = re.compile(r'#.+?#')
    df['话题标签有无'] = df['评论文案'].fillna('').apply(lambda x: 1 if topic_pattern.search(x) else 0)

    # 小米相关词数
    def count_xiaomi_keywords(text):
        if not text:
            return 0
        text = str(text).lower()
        count = 0
        for keyword in XIAOMI_KEYWORDS:
            count += text.count(keyword.lower())
        return count

    df['小米相关词数'] = df['评论文案'].fillna('').apply(count_xiaomi_keywords)

    return df


def merge_external_features(df, lda_df, density_df):
    """合并外部特征（LDA主题、时间密度）

    参数:
        df: 原始数据
        lda_df: LDA特征DataFrame（包含'序号'和'主题'）
        density_df: 时间密度特征DataFrame

    返回:
        合并后的DataFrame
    """
    df = df.copy()

    # 合并LDA特征
    if lda_df is not None and '序号' in lda_df.columns:
        df = df.merge(lda_df[['序号', '主题']], on='序号', how='left')
        df['主题'] = df['主题'].fillna(0).astype(int)

    # 合并时间密度特征
    if density_df is not None and '序号' in density_df.columns:
        density_cols = ['序号', '时间顺序索引', '最大相似度', '重复次数']
        density_cols = [c for c in density_cols if c in density_df.columns]
        df = df.merge(density_df[density_cols], on='序号', how='left')

        # 填充缺失值
        for col in ['时间顺序索引', '最大相似度', '重复次数']:
            if col in df.columns:
                df[col] = df[col].fillna(0)

    return df


def get_feature_cols(feature_groups):
    """根据特征组名称获取特征列列表

    参数:
        feature_groups: 特征组名称列表，如 ['base', 'text'] 或 'all'

    返回:
        特征列名称列表
    """
    if feature_groups == 'all' or 'all' in feature_groups:
        feature_groups = list(FEATURE_GROUPS.keys())

    if isinstance(feature_groups, str):
        feature_groups = [g.strip() for g in feature_groups.split(',')]

    feature_cols = []
    for group in feature_groups:
        if group in FEATURE_GROUPS:
            feature_cols.extend(FEATURE_GROUPS[group])

    return feature_cols


def prepare_features(df, feature_cols, target_col='子评论数'):
    """准备特征矩阵和目标变量

    参数:
        df: 数据DataFrame
        feature_cols: 特征列名称列表
        target_col: 目标变量列名

    返回:
        X: 特征矩阵 (numpy array)
        y: 目标变量 (numpy array)
    """
    # 检查缺失列
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"缺失特征列: {missing_cols}")

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    return X, y
