"""
评论数预测模型训练脚本
预测目标：评论的子评论数

不可用特征（转发数/点赞数相关）：
    - 用户总转发数、用户总点赞数
    - 微博转发数、微博点赞数

可用输入特征：
    基础特征组 (--features base):
        - 用户总评论数、用户是否认证
        - 是否一级评论、楼层数
        - 微博评论数
        - 发布时间（提取小时、星期等）

    文本特征组 (--features text):
        - 评论长度、感叹号数、问号数、表情数
        - 话题标签有无、小米汽车相关词汇数

    LDA特征组 (--features lda):
        - LDA主题 (需要 train_lda.pkl 等文件)

    时间密度特征组 (--features density):
        - 时间顺序索引、相似程度、重复次数 (需要 train_time_density.pkl 等文件)

使用方法：
    python src/train_comment_predictor.py --model rf
    python src/train_comment_predictor.py --model rf --features base,text
    python src/train_comment_predictor.py --model rf --features base,text,lda,density
    python src/train_comment_predictor.py --model xgboost --features all
    python src/train_comment_predictor.py --model ngboost --features base,text,density
"""

import argparse
import os
import pickle
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

# PyTorch相关导入
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 配置 ====================
ROOT_DIR = Path('D:/010_CodePrograms/L/LLM_su7')
RESULTS_BASE = ROOT_DIR / 'results'

# 数据中的列：
# 0: 评论文案, 1: 微博文案, 2: 根评论文案, 3: 父评论文案
# 4: 发布时间, 5: 用户总转发数, 6: 用户总评论数, 7: 用户总点赞数
# 8: 用户是否认证, 9: 是否一级评论, 10: 子评论数, 11: 点赞数
# 12: 微博转发数, 13: 微博评论数, 14: 微博点赞数

# 不可用特征（转发数/点赞数相关）：
#   - 用户总转发数、用户总点赞数
#   - 微博转发数、微博点赞数
#   - 点赞数（评论本身的点赞数）

# ==================== 特征定义 ====================
# 基础特征（从原始数据直接读取或提取）
BASE_FEATURE_COLS = [
    '用户总评论数',      # 用户历史评论数
    '用户是否认证',      # 是否为认证用户
    '是否一级评论',      # 是否直接评论微博
    '微博评论数',        # 该微博的总评论数
    '发布小时',          # 从发布时间提取
    '发布星期',          # 从发布时间提取
    '是否工作日',        # 从发布时间提取
]

# 文本特征（从评论文案提取）
TEXT_FEATURE_COLS = [
    '评论长度',          # 评论文案字符数
    '感叹号数',          # 感叹号数量
    '问号数',            # 问号数量
    '表情数',            # 表情符号数量
    '话题标签有无',      # 是否包含#话题#
    '小米相关词数',      # 小米汽车相关词汇数量
]

# LDA特征（从 train_lda.pkl 等文件读取）
LDA_FEATURE_COLS = [
    '主题',              # LDA主题编号
]

# 时间密度特征（从 train_time_density.pkl 等文件读取）
DENSITY_FEATURE_COLS = [
    '时间顺序索引',      # 按时间排序的索引
    '最大相似度',        # 与之前评论的最大相似度
    '重复次数',          # 相似评论的数量
]

# 特征组定义
FEATURE_GROUPS = {
    'base': BASE_FEATURE_COLS,
    'text': TEXT_FEATURE_COLS,
    'lda': LDA_FEATURE_COLS,
    'density': DENSITY_FEATURE_COLS,
}

TARGET_COL = '子评论数'

# 小米汽车相关词汇
XIAOMI_KEYWORDS = [
    '小米', 'SU7', 'su7', '雷军', '雷总', '米车', '小米汽车',
    '智驾', '智能驾驶', '自动驾驶', 'NIO', '蔚来', '特斯拉', 'Tesla',
    '比亚迪', 'BYD', '新能源', '电动车', '电车', '续航', '充电',
    '车机', '中控', '座舱', '辅助驾驶', 'AEB', '预碰撞'
]

# ==================== BGE神经网络模型配置 ====================
# VIP用户白名单（被@超过20次的高频用户）
VIP_USERS = [
    '小米法务部', '雷军', '小米汽车', '王化', '鸿蒙智行法务',
    '薛定谔的英短咕咕咕', '余承东', '小米公司发言人', '小蒜苗长',
    '万能的大熊', '我是大彬同学', '科技新一', '美国驻华大使馆',
    'AI逃逸', '小米公司', '羊驼的睡衣', '卢伟冰',
    '诗雨370491153', '不会武功的武功李云飞'
]
VIP_USER_TO_ID = {user: i for i, user in enumerate(VIP_USERS)}

# 特殊Token
SPECIAL_TOKEN_USER = '_USER_'  # 非VIP用户统一替换

# BGE模型路径
BGE_MODEL_PATH = ROOT_DIR / 'bge-base-zh-v1.5'

# @用户提取正则
AT_USER_PATTERN = re.compile(r'@([^\s:：,，。！!?？\[\]]+)')
# 表情符号正则（微博格式 [xxx]）
EMOJI_PATTERN = re.compile(r'\[[^\[\]]+\]')


def preprocess_text_for_bge(text, replace_users=True):
    """预处理文本用于BGE编码

    处理：
    1. @用户：VIP保留，其他替换为_USER_
    2. 保留表情符号和小米关键词（BGE会学习其语义）

    参数:
        text: 原始文本
        replace_users: 是否替换非VIP用户

    返回:
        processed_text: 处理后的文本
    """
    if not isinstance(text, str) or len(text) == 0:
        return ""

    if replace_users:
        # 替换非VIP用户为_USER_
        def replace_at_user(match):
            username = match.group(1)
            if username in VIP_USER_TO_ID:
                return f'@{username}'  # VIP用户保留
            return f'@{SPECIAL_TOKEN_USER}'  # 非VIP用户替换

        text = AT_USER_PATTERN.sub(replace_at_user, text)

    return text


# ==================== 模型注册 ====================
MODEL_REGISTRY = {}


def register_model(name):
    """模型注册装饰器"""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


# Log变换常量
LOG_OFFSET = 10


def log_transform(y):
    """对目标变量进行log变换"""
    return np.log(np.maximum(y, 0) + LOG_OFFSET)


def inverse_log_transform(y_log):
    """log变换的逆变换，确保结果非负"""
    return np.maximum(np.exp(y_log) - LOG_OFFSET, 0)


@register_model('ridge')
class RidgeModel:
    """Ridge回归 (使用log变换目标，优化MSLE)"""
    def __init__(self, alpha=1.0, **kwargs):
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.name = 'Ridge'
        self.use_log_target = True  # 标记使用log变换

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        y_log = log_transform(y)
        self.model.fit(X_scaled, y_log)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        y_log_pred = self.model.predict(X_scaled)
        return inverse_log_transform(y_log_pred)


@register_model('lasso')
class LassoModel:
    """Lasso回归 (使用log变换目标，优化MSLE)"""
    def __init__(self, alpha=1.0, **kwargs):
        self.model = Lasso(alpha=alpha)
        self.scaler = StandardScaler()
        self.name = 'Lasso'
        self.use_log_target = True

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        y_log = log_transform(y)
        self.model.fit(X_scaled, y_log)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        y_log_pred = self.model.predict(X_scaled)
        return inverse_log_transform(y_log_pred)


@register_model('rf')
class RandomForestModel:
    """随机森林 (使用log变换目标，优化MSLE)"""
    def __init__(self, n_estimators=100, max_depth=10, random_state=42, **kwargs):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.name = 'RandomForest'
        self.use_log_target = True

    def fit(self, X, y):
        y_log = log_transform(y)
        self.model.fit(X, y_log)

    def predict(self, X):
        y_log_pred = self.model.predict(X)
        return inverse_log_transform(y_log_pred)

    def get_feature_importance(self):
        return self.model.feature_importances_


@register_model('gbdt')
class GBDTModel:
    """梯度提升树 (使用log变换目标，优化MSLE)"""
    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, **kwargs):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state
        )
        self.name = 'GBDT'
        self.use_log_target = True

    def fit(self, X, y):
        y_log = log_transform(y)
        self.model.fit(X, y_log)

    def predict(self, X):
        y_log_pred = self.model.predict(X)
        return inverse_log_transform(y_log_pred)

    def get_feature_importance(self):
        return self.model.feature_importances_


# 可选：XGBoost支持
try:
    import xgboost as xgb

    @register_model('xgboost')
    class XGBoostModel:
        """XGBoost (使用log变换目标，优化MSLE)"""
        def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, **kwargs):
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                n_jobs=-1
            )
            self.name = 'XGBoost'
            self.use_log_target = True

        def fit(self, X, y):
            y_log = log_transform(y)
            self.model.fit(X, y_log)

        def predict(self, X):
            y_log_pred = self.model.predict(X)
            return inverse_log_transform(y_log_pred)

        def get_feature_importance(self):
            return self.model.feature_importances_

except ImportError:
    print("[提示] XGBoost未安装，如需使用请运行: pip install xgboost")


# 可选：LightGBM支持
try:
    import lightgbm as lgb

    @register_model('lightgbm')
    class LightGBMModel:
        """LightGBM (使用log变换目标，优化MSLE)"""
        def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, **kwargs):
            self.model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1
            )
            self.name = 'LightGBM'
            self.use_log_target = True

        def fit(self, X, y):
            y_log = log_transform(y)
            self.model.fit(X, y_log)

        def predict(self, X):
            y_log_pred = self.model.predict(X)
            return inverse_log_transform(y_log_pred)

        def get_feature_importance(self):
            return self.model.feature_importances_

except ImportError:
    print("[提示] LightGBM未安装，如需使用请运行: pip install lightgbm")


# 可选：NGBoost支持（同时预测数值和不确定性）
try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from ngboost.scores import LogScore  # NLL损失

    @register_model('ngboost')
    class NGBoostModel:
        """NGBoost - 自然梯度提升，同时预测数值和不确定性

        使用对数尺度训练：在log(y+10)空间建模正态分布
        损失函数: L = 0.5 * log(σ²) + (log(y+10) - log(μ+10))² / (2σ²)
        输出：预测均值(原始空间)和标准差(log空间)
        """
        def __init__(self, n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, **kwargs):
            from sklearn.tree import DecisionTreeRegressor
            self.model = NGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
                Dist=Normal,           # 使用正态分布
                Score=LogScore,        # NLL损失函数
                verbose=False,
                natural_gradient=True,
                Base=DecisionTreeRegressor(max_depth=max_depth, random_state=random_state),
                minibatch_frac=1.0
            )
            self.scaler = StandardScaler()  # 特征标准化
            self.name = 'NGBoost'
            self.supports_uncertainty = True
            self.use_log_target = True  # 标记使用log变换

        def fit(self, X, y):
            """在log空间训练"""
            X_scaled = self.scaler.fit_transform(X)
            y_log = log_transform(y)  # 转换到log空间
            self.model.fit(X_scaled, y_log)

        def predict(self, X):
            """返回预测均值（原始空间）"""
            X_scaled = self.scaler.transform(X)
            y_log_pred = self.model.predict(X_scaled)
            return inverse_log_transform(y_log_pred)  # 转换回原始空间

        def predict_dist(self, X):
            """返回完整分布参数

            返回:
                mu: 预测均值（原始空间）
                sigma: 预测标准差（log空间，用于不确定性度量）
            """
            X_scaled = self.scaler.transform(X)
            dist = self.model.pred_dist(X_scaled)
            # loc是log空间的均值，scale是log空间的标准差
            mu_log = dist.loc
            sigma_log = dist.scale
            # 均值转换回原始空间
            mu = inverse_log_transform(mu_log)
            # 标准差保持在log空间（用于不确定性评估）
            return mu, sigma_log

        def get_feature_importance(self):
            # NGBoost的feature_importances_可能是多维的，取均值或处理为1D
            fi = self.model.feature_importances_
            if fi.ndim > 1:
                fi = fi.mean(axis=0)  # 多个分布参数的重要性取平均
            return fi

        def compute_nll(self, X, y):
            """计算对数尺度的NLL损失

            L = 0.5 * log(σ²) + (log(y+10) - μ_log)² / (2σ²)
            """
            X_scaled = self.scaler.transform(X)
            dist = self.model.pred_dist(X_scaled)
            y_log = log_transform(y)
            # 使用NGBoost的内置logpdf计算NLL
            nll = -dist.logpdf(y_log).mean()
            return nll

except ImportError:
    print("[提示] NGBoost未安装，如需使用请运行: pip install ngboost")


# ==================== BGE神经网络模型 ====================
class BertEmbeddings(nn.Module):
    """BERT嵌入层"""
    def __init__(self, config, pad_token_id=0):
        super().__init__()
        self.word_embeddings = nn.Embedding(config['vocab_size'], config['hidden_size'], padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(config['max_position_embeddings'], config['hidden_size'])
        self.token_type_embeddings = nn.Embedding(config['type_vocab_size'], config['hidden_size'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=config.get('layer_norm_eps', 1e-12))
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(position_ids) + self.token_type_embeddings(token_type_ids)
        return self.dropout(self.LayerNorm(embeddings))


class BertSelfAttention(nn.Module):
    """BERT自注意力层"""
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config['num_attention_heads']
        self.attention_head_size = config['hidden_size'] // config['num_attention_heads']
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config['hidden_size'], self.all_head_size)
        self.key = nn.Linear(config['hidden_size'], self.all_head_size)
        self.value = nn.Linear(config['hidden_size'], self.all_head_size)
        self.dropout = nn.Dropout(config['attention_probs_dropout_prob'])

    def forward(self, hidden_states, attention_mask=None):
        def transpose_for_scores(x):
            new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            return x.view(*new_shape).permute(0, 2, 1, 3)

        q = transpose_for_scores(self.query(hidden_states))
        k = transpose_for_scores(self.key(hidden_states))
        v = transpose_for_scores(self.value(hidden_states))
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.attention_head_size ** 0.5)
        if attention_mask is not None:
            scores = scores + attention_mask
        context = torch.matmul(self.dropout(F.softmax(scores, dim=-1)), v)
        return context.permute(0, 2, 1, 3).contiguous().view(context.size(0), -1, self.all_head_size)


class BertLayer(nn.Module):
    """BERT Transformer层"""
    def __init__(self, config):
        super().__init__()
        self.attention = BertSelfAttention(config)
        self.attention_dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.attention_norm = nn.LayerNorm(config['hidden_size'], eps=config.get('layer_norm_eps', 1e-12))
        self.intermediate = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.output_dense = nn.Linear(config['intermediate_size'], config['hidden_size'])
        self.output_norm = nn.LayerNorm(config['hidden_size'], eps=config.get('layer_norm_eps', 1e-12))
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, hidden_states, attention_mask=None):
        attn_out = self.attention(hidden_states, attention_mask)
        hidden_states = self.attention_norm(hidden_states + self.dropout(self.attention_dense(attn_out)))
        intermediate = F.gelu(self.intermediate(hidden_states))
        return self.output_norm(hidden_states + self.dropout(self.output_dense(intermediate)))


class BertModel(nn.Module):
    """BERT模型"""
    def __init__(self, config, pad_token_id=0):
        super().__init__()
        self.embeddings = BertEmbeddings(config, pad_token_id)
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(config['num_hidden_layers'])])

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
        hidden_states = self.embeddings(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class CrossAttentionFusion(nn.Module):
    """Cross-Attention融合层：评论作为Query，上下文（微博/根评论/父评论）作为Key/Value"""
    def __init__(self, hidden_size=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, comment_emb, context_embs):
        """
        参数:
            comment_emb: [batch, hidden_size] 评论embedding
            context_embs: [batch, 3, hidden_size] 微博/根评论/父评论embedding

        返回:
            fused: [batch, hidden_size] 融合后的特征
        """
        # 扩展comment_emb为 [batch, 1, hidden_size] 作为query
        query = comment_emb.unsqueeze(1)
        # context_embs 作为 key 和 value
        attn_out, _ = self.attention(query, context_embs, context_embs)
        # 残差连接
        fused = self.norm(comment_emb + self.dropout(attn_out.squeeze(1)))
        return fused


class DualPredictionHead(nn.Module):
    """双预测头：同时预测均值和方差"""
    def __init__(self, input_size, hidden_size=256, dropout=0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(hidden_size // 2, 1)      # 均值（log空间）
        self.sigma_head = nn.Linear(hidden_size // 2, 1)   # 方差（log空间）

    def forward(self, x):
        shared = self.shared(x)
        mu = self.mu_head(shared).squeeze(-1)
        # 使用softplus确保sigma为正值，加上小常数避免数值问题
        sigma = F.softplus(self.sigma_head(shared)).squeeze(-1) + 1e-4
        return mu, sigma


class CommentPredictorNN(nn.Module):
    """评论数预测神经网络

    架构:
        4个文本 → BGE编码(4×768) → Cross-Attention融合 → 拼接数值特征 → 双预测头 → (均值, 方差)
    """
    def __init__(self, bert_model, num_numeric_features, hidden_size=256, dropout=0.1, freeze_bert=True):
        super().__init__()
        self.bert = bert_model
        self.bert_hidden_size = 768

        # 冻结BERT
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Cross-Attention融合层
        self.fusion = CrossAttentionFusion(self.bert_hidden_size, num_heads=8, dropout=dropout)

        # 数值特征投影层
        self.numeric_proj = nn.Sequential(
            nn.Linear(num_numeric_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 特征融合后的维度: 768 (文本融合) + 64 (数值投影)
        fusion_size = self.bert_hidden_size + 64

        # 双预测头
        self.prediction_head = DualPredictionHead(fusion_size, hidden_size, dropout)

    def encode_text(self, input_ids, attention_mask):
        """编码单个文本，返回[CLS]的embedding"""
        output = self.bert(input_ids, attention_mask)
        return output[:, 0, :]  # 取[CLS]位置的特征

    def forward(self, comment_ids, comment_mask,
                weibo_ids, weibo_mask,
                root_ids, root_mask,
                parent_ids, parent_mask,
                numeric_features):
        """
        前向传播

        参数:
            comment_ids, comment_mask: 评论文案的tokenized输入
            weibo_ids, weibo_mask: 微博文案的tokenized输入
            root_ids, root_mask: 根评论文案的tokenized输入
            parent_ids, parent_mask: 父评论文案的tokenized输入
            numeric_features: [batch, num_numeric_features] 数值特征

        返回:
            mu: [batch] 预测均值（log空间）
            sigma: [batch] 预测标准差（log空间）
        """
        # 编码4个文本
        comment_emb = self.encode_text(comment_ids, comment_mask)
        weibo_emb = self.encode_text(weibo_ids, weibo_mask)
        root_emb = self.encode_text(root_ids, root_mask)
        parent_emb = self.encode_text(parent_ids, parent_mask)

        # 堆叠上下文embedding: [batch, 3, 768]
        context_embs = torch.stack([weibo_emb, root_emb, parent_emb], dim=1)

        # Cross-Attention融合
        text_fused = self.fusion(comment_emb, context_embs)

        # 数值特征投影
        numeric_proj = self.numeric_proj(numeric_features)

        # 拼接所有特征
        combined = torch.cat([text_fused, numeric_proj], dim=1)

        # 双预测头
        mu, sigma = self.prediction_head(combined)

        return mu, sigma


class CommentDataset(Dataset):
    """评论预测数据集"""
    def __init__(self, df, tokenizer, density_df=None, max_length=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 合并时间密度特征
        if density_df is not None:
            self.df = self.df.merge(density_df, on='序号', how='left')
            self.df['时间顺序索引'] = self.df['时间顺序索引'].fillna(0)
            self.df['最大相似度'] = self.df['最大相似度'].fillna(0)
            self.df['重复次数'] = self.df['重复次数'].fillna(0)

        # 提取时间特征
        self.df['发布时间'] = pd.to_datetime(self.df['发布时间'])
        self.df['发布小时'] = self.df['发布时间'].dt.hour
        self.df['发布星期'] = self.df['发布时间'].dt.dayofweek

        # 预处理文本
        self.comment_texts = [preprocess_text_for_bge(str(t)) for t in self.df['评论文案'].fillna('')]
        self.weibo_texts = [preprocess_text_for_bge(str(t)) for t in self.df['微博文案'].fillna('')]
        self.root_texts = [preprocess_text_for_bge(str(t)) for t in self.df['根评论文案'].fillna('')]
        self.parent_texts = [preprocess_text_for_bge(str(t)) for t in self.df['父评论文案'].fillna('')]

        # 目标变量
        self.targets = self.df['子评论数'].values.astype(np.float32)

        # 准备数值特征
        self._prepare_numeric_features()

    def _prepare_numeric_features(self):
        """准备数值特征"""
        # 用户总评论数（log变换）
        user_comments = np.log1p(self.df['用户总评论数'].fillna(0).values)

        # 用户是否认证
        is_verified = self.df['用户是否认证'].fillna(0).astype(float).values

        # 是否一级评论
        is_first_level = self.df['是否一级评论'].fillna(0).astype(float).values

        # 发布小时（周期编码）
        hour = self.df['发布小时'].values
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # 发布星期（周期编码）
        weekday = self.df['发布星期'].values
        weekday_sin = np.sin(2 * np.pi * weekday / 7)
        weekday_cos = np.cos(2 * np.pi * weekday / 7)

        # 时间顺序索引（标准化）
        time_idx = self.df.get('时间顺序索引', pd.Series([0] * len(self.df))).values
        time_idx = (time_idx - time_idx.mean()) / (time_idx.std() + 1e-8)

        # 最大相似度
        max_sim = self.df.get('最大相似度', pd.Series([0] * len(self.df))).values

        # 重复次数（log变换）
        repeat_count = np.log1p(self.df.get('重复次数', pd.Series([0] * len(self.df))).values)

        # 合并所有数值特征
        self.numeric_features = np.column_stack([
            user_comments,    # 0
            is_verified,      # 1
            is_first_level,   # 2
            hour_sin,         # 3
            hour_cos,         # 4
            weekday_sin,      # 5
            weekday_cos,      # 6
            time_idx,         # 7
            max_sim,          # 8
            repeat_count,     # 9
        ]).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def _tokenize(self, text):
        """Tokenize单个文本"""
        if not text or text == "":
            text = "空"
        encoded = self.tokenizer.encode(text)
        ids = encoded.ids[:self.max_length]
        # padding
        pad_len = self.max_length - len(ids)
        ids = ids + [0] * pad_len
        mask = [1.0] * (self.max_length - pad_len) + [0.0] * pad_len
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float)

    def __getitem__(self, idx):
        # Tokenize 4个文本
        comment_ids, comment_mask = self._tokenize(self.comment_texts[idx])
        weibo_ids, weibo_mask = self._tokenize(self.weibo_texts[idx])
        root_ids, root_mask = self._tokenize(self.root_texts[idx])
        parent_ids, parent_mask = self._tokenize(self.parent_texts[idx])

        return {
            'comment_ids': comment_ids,
            'comment_mask': comment_mask,
            'weibo_ids': weibo_ids,
            'weibo_mask': weibo_mask,
            'root_ids': root_ids,
            'root_mask': root_mask,
            'parent_ids': parent_ids,
            'parent_mask': parent_mask,
            'numeric_features': torch.tensor(self.numeric_features[idx], dtype=torch.float),
            'target': torch.tensor(self.targets[idx], dtype=torch.float),
        }


def nll_loss(y_true, mu, sigma):
    """对数尺度NLL损失

    L = 0.5 * log(σ²) + (log(y+10) - μ)² / (2σ²)

    参数:
        y_true: 真实值（原始空间）
        mu: 预测均值（log空间）
        sigma: 预测标准差（log空间）
    """
    y_log = torch.log(y_true + LOG_OFFSET)
    nll = 0.5 * torch.log(sigma ** 2) + ((y_log - mu) ** 2) / (2 * sigma ** 2)
    return nll.mean()


@register_model('bge_nn')
class BGENNModel:
    """BGE + 神经网络预测模型

    使用BGE-base-zh-v1.5编码4个文本（评论/微博/根评论/父评论），
    通过Cross-Attention融合，结合数值特征，双预测头输出均值和方差。
    """
    def __init__(self, freeze_bert=True, hidden_size=256, dropout=0.1, **kwargs):
        self.name = 'BGE_NN'
        self.freeze_bert = freeze_bert
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.supports_uncertainty = True
        self.use_log_target = True

        # 训练参数
        self.epochs = kwargs.get('epochs', 30)
        self.batch_size = kwargs.get('batch_size', 32)
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.patience = kwargs.get('patience', 5)

    def _load_bge_model(self):
        """加载BGE模型"""
        from tokenizers import Tokenizer

        model_path = str(BGE_MODEL_PATH)
        print(f"加载BGE模型: {model_path}")

        # 加载tokenizer
        self.tokenizer = Tokenizer.from_file(os.path.join(model_path, 'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=128)

        # 加载vocab获取pad_token_id
        with open(os.path.join(model_path, 'vocab.txt'), 'r', encoding='utf-8') as f:
            vocab = {line.strip(): idx for idx, line in enumerate(f)}
        self.pad_token_id = vocab.get('[PAD]', 0)

        # 加载配置
        with open(os.path.join(model_path, 'config.json'), 'r') as f:
            config = json.load(f)

        # 创建BERT模型
        bert_model = BertModel(config, self.pad_token_id)

        # 加载预训练权重
        state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location='cpu')

        # 权重映射
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('bert.', '')
            if 'encoder.layer.' in new_key:
                new_key = new_key.replace('encoder.layer.', 'layers.')
                new_key = new_key.replace('.attention.self.', '.attention.')
                new_key = new_key.replace('.attention.output.dense', '.attention_dense')
                new_key = new_key.replace('.attention.output.LayerNorm', '.attention_norm')
                new_key = new_key.replace('.intermediate.dense', '.intermediate')
                new_key = new_key.replace('.output.dense', '.output_dense')
                new_key = new_key.replace('.output.LayerNorm', '.output_norm')
            new_state_dict[new_key] = value

        model_state = bert_model.state_dict()
        matched = {k: v for k, v in new_state_dict.items() if k in model_state and model_state[k].shape == v.shape}
        model_state.update(matched)
        bert_model.load_state_dict(model_state)

        print(f"BGE权重加载完成，匹配: {len(matched)}/{len(model_state)}")

        return bert_model

    def fit(self, train_df, val_df, train_density=None, val_density=None):
        """训练模型"""
        print(f"\n使用设备: {self.device}")
        print(f"冻结BGE: {self.freeze_bert}")

        # 加载BGE模型
        bert_model = self._load_bge_model()

        # 创建数据集
        print("创建数据集...")
        train_dataset = CommentDataset(train_df, self.tokenizer, train_density, max_length=128)
        val_dataset = CommentDataset(val_df, self.tokenizer, val_density, max_length=128)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        # 创建模型
        num_numeric_features = train_dataset.numeric_features.shape[1]
        self.model = CommentPredictorNN(
            bert_model,
            num_numeric_features,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            freeze_bert=self.freeze_bert
        ).to(self.device)

        # 优化器
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=0.01
        )

        # 学习率调度
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )

        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            # 训练
            self.model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}'):
                # 移动数据到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}

                optimizer.zero_grad()

                mu, sigma = self.model(
                    batch['comment_ids'], batch['comment_mask'],
                    batch['weibo_ids'], batch['weibo_mask'],
                    batch['root_ids'], batch['root_mask'],
                    batch['parent_ids'], batch['parent_mask'],
                    batch['numeric_features']
                )

                loss = nll_loss(batch['target'], mu, sigma)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # 验证
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    mu, sigma = self.model(
                        batch['comment_ids'], batch['comment_mask'],
                        batch['weibo_ids'], batch['weibo_mask'],
                        batch['root_ids'], batch['root_mask'],
                        batch['parent_ids'], batch['parent_mask'],
                        batch['numeric_features']
                    )

                    loss = nll_loss(batch['target'], mu, sigma)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            # 学习率调度
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # 恢复最佳模型
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)
            self.model.to(self.device)

        self.train_losses = train_losses
        self.val_losses = val_losses

    def predict(self, df, density_df=None):
        """预测（返回均值）"""
        mu, _ = self.predict_dist(df, density_df)
        return mu

    def predict_dist(self, df, density_df=None):
        """预测分布参数

        返回:
            mu: 预测均值（原始空间）
            sigma: 预测标准差（log空间）
        """
        dataset = CommentDataset(df, self.tokenizer, density_df, max_length=128)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.model.eval()
        all_mu = []
        all_sigma = []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                mu, sigma = self.model(
                    batch['comment_ids'], batch['comment_mask'],
                    batch['weibo_ids'], batch['weibo_mask'],
                    batch['root_ids'], batch['root_mask'],
                    batch['parent_ids'], batch['parent_mask'],
                    batch['numeric_features']
                )

                # 转换回原始空间
                mu_orig = torch.exp(mu) - LOG_OFFSET
                mu_orig = torch.clamp(mu_orig, min=0)

                all_mu.append(mu_orig.cpu().numpy())
                all_sigma.append(sigma.cpu().numpy())

        return np.concatenate(all_mu), np.concatenate(all_sigma)

    def compute_nll(self, df, density_df=None):
        """计算NLL损失"""
        dataset = CommentDataset(df, self.tokenizer, density_df, max_length=128)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.model.eval()
        total_nll = 0
        count = 0

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                mu, sigma = self.model(
                    batch['comment_ids'], batch['comment_mask'],
                    batch['weibo_ids'], batch['weibo_mask'],
                    batch['root_ids'], batch['root_mask'],
                    batch['parent_ids'], batch['parent_mask'],
                    batch['numeric_features']
                )

                loss = nll_loss(batch['target'], mu, sigma)
                total_nll += loss.item() * len(batch['target'])
                count += len(batch['target'])

        return total_nll / count


# ==================== 数据加载 ====================
def load_data(use_pkl=True):
    """加载训练/验证/测试数据"""
    print("=" * 60)
    print("【加载数据】")
    print("=" * 60)

    ext = 'pkl' if use_pkl else 'csv'

    if use_pkl:
        train_df = pd.read_pickle(ROOT_DIR / f'train.{ext}')
        val_df = pd.read_pickle(ROOT_DIR / f'val.{ext}')
        test_df = pd.read_pickle(ROOT_DIR / f'test.{ext}')
    else:
        train_df = pd.read_csv(ROOT_DIR / f'train.{ext}', encoding='utf-8-sig')
        val_df = pd.read_csv(ROOT_DIR / f'val.{ext}', encoding='utf-8-sig')
        test_df = pd.read_csv(ROOT_DIR / f'test.{ext}', encoding='utf-8-sig')

    print(f"训练集: {len(train_df):,} 条")
    print(f"验证集: {len(val_df):,} 条")
    print(f"测试集: {len(test_df):,} 条")

    return train_df, val_df, test_df


def load_lda_features():
    """加载LDA主题特征"""
    try:
        train_lda = pd.read_pickle(ROOT_DIR / 'train_lda.pkl')
        val_lda = pd.read_pickle(ROOT_DIR / 'val_lda.pkl')
        test_lda = pd.read_pickle(ROOT_DIR / 'test_lda.pkl')
        print(f"  LDA特征已加载")
        return train_lda, val_lda, test_lda
    except FileNotFoundError:
        print(f"  [警告] LDA特征文件不存在，跳过LDA特征")
        return None, None, None


def load_density_features(method='minhash'):
    """加载时间密度特征

    参数:
        method: 'bge' 使用BGE语义相似度, 'minhash' 使用MinHash Jaccard相似度
    """
    suffix = '_minhash' if method == 'minhash' else ''
    try:
        train_density = pd.read_pickle(ROOT_DIR / f'train_time_density{suffix}.pkl')
        val_density = pd.read_pickle(ROOT_DIR / f'val_time_density{suffix}.pkl')
        test_density = pd.read_pickle(ROOT_DIR / f'test_time_density{suffix}.pkl')
        print(f"  时间密度特征已加载 (method={method})")
        return train_density, val_density, test_density
    except FileNotFoundError:
        print(f"  [警告] 时间密度特征文件不存在 (method={method})，跳过density特征")
        return None, None, None


def extract_time_features(df):
    """从发布时间提取时间特征"""
    df = df.copy()

    # 确保发布时间是datetime类型
    if '发布时间' in df.columns:
        df['发布时间'] = pd.to_datetime(df['发布时间'])
        df['发布小时'] = df['发布时间'].dt.hour
        df['发布星期'] = df['发布时间'].dt.dayofweek  # 0=周一, 6=周日
        df['是否工作日'] = (df['发布星期'] < 5).astype(int)  # 周一到周五为工作日

    return df


def extract_text_features(df):
    """从评论文案提取文本特征"""
    df = df.copy()

    # 表情符号正则表达式（匹配微博表情 [xxx] 格式）
    emoji_pattern = re.compile(r'\[[\u4e00-\u9fa5a-zA-Z]+\]')

    # 话题标签正则表达式
    topic_pattern = re.compile(r'#[^#]+#')

    def count_features(text):
        if not isinstance(text, str):
            return 0, 0, 0, 0, 0, 0

        # 评论长度
        length = len(text)

        # 感叹号数（中英文）
        exclamation = text.count('!') + text.count('！')

        # 问号数（中英文）
        question = text.count('?') + text.count('？')

        # 表情数
        emojis = len(emoji_pattern.findall(text))

        # 话题标签有无
        has_topic = 1 if topic_pattern.search(text) else 0

        # 小米相关词汇数
        xiaomi_count = sum(1 for kw in XIAOMI_KEYWORDS if kw in text)

        return length, exclamation, question, emojis, has_topic, xiaomi_count

    # 批量提取特征
    features = df['评论文案'].apply(count_features)
    df['评论长度'] = features.apply(lambda x: x[0])
    df['感叹号数'] = features.apply(lambda x: x[1])
    df['问号数'] = features.apply(lambda x: x[2])
    df['表情数'] = features.apply(lambda x: x[3])
    df['话题标签有无'] = features.apply(lambda x: x[4])
    df['小米相关词数'] = features.apply(lambda x: x[5])

    return df


def merge_external_features(df, lda_df, density_df):
    """合并外部特征（LDA和时间密度）"""
    df = df.copy()

    # 合并LDA特征
    if lda_df is not None and '序号' in df.columns:
        df = df.merge(lda_df[['序号', '主题']], on='序号', how='left')
        df['主题'] = df['主题'].fillna(0).astype(int)

    # 合并时间密度特征
    if density_df is not None and '序号' in df.columns:
        density_cols = ['序号', '时间顺序索引', '最大相似度', '重复次数']
        df = df.merge(density_df[density_cols], on='序号', how='left')
        df['时间顺序索引'] = df['时间顺序索引'].fillna(0)
        df['最大相似度'] = df['最大相似度'].fillna(0)
        df['重复次数'] = df['重复次数'].fillna(0)

    return df


def get_feature_cols(feature_groups):
    """根据特征组列表获取所有特征列名"""
    all_cols = []
    for group in feature_groups:
        if group in FEATURE_GROUPS:
            all_cols.extend(FEATURE_GROUPS[group])
    return all_cols


def prepare_features(df, feature_cols, target_col=TARGET_COL):
    """准备特征和标签"""
    # 提取特征
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # 处理布尔值转换为数值
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    # 处理缺失值
    X = X.fillna(0)
    y = y.fillna(0)

    return X.values, y.values


# ==================== 评估指标 ====================
# 注：LOG_OFFSET和safe_log/safe_exp已在模型定义部分定义


def compute_msle(y_true, y_pred):
    """计算MSLE (Mean Squared Logarithmic Error)

    公式: MSLE = mean((log(y_true + 10) - log(y_pred + 10))^2)
    适合长尾分布，关注比例而非差值
    """
    log_true = log_transform(y_true)
    log_pred = log_transform(y_pred)
    return np.mean((log_true - log_pred) ** 2)


def compute_acp(y_true, y_pred, alpha=0.2, delta=5):
    """计算ACP (Accuracy within X% - 命中率)

    定义：如果 |y_pred - y_true| <= max(alpha * y_true, delta)，则视为"命中"

    参数:
        alpha: 相对误差容忍度 (默认20%)
        delta: 绝对误差容忍度 (默认5)

    返回:
        命中率 (0-1之间)
    """
    tolerance = np.maximum(alpha * np.abs(y_true), delta)
    hits = np.abs(y_pred - y_true) <= tolerance
    return np.mean(hits)


def compute_log_nll(y_true, y_pred, y_std):
    """计算对数尺度的NLL (Negative Log Likelihood)

    公式: L = 0.5 * log(σ²) + (log(y+10) - log(μ+10))² / (2σ²)

    参数:
        y_true: 真实值（原始空间）
        y_pred: 预测均值（原始空间）
        y_std: 预测标准差（log空间）
    """
    log_true = log_transform(y_true)
    log_pred = log_transform(y_pred)

    # 避免σ过小导致数值问题
    sigma = np.maximum(y_std, 1e-6)

    # NLL = 0.5 * log(σ²) + (log_true - log_pred)² / (2σ²)
    nll = 0.5 * np.log(sigma ** 2) + ((log_true - log_pred) ** 2) / (2 * sigma ** 2)
    return np.mean(nll)


def compute_picp(y_true, y_pred, y_std, confidence=0.95):
    """计算PICP (Prediction Interval Coverage Probability - 区间覆盖率)

    在log空间构建置信区间，检查log(y_true+10)是否落在区间内

    参数:
        y_true: 真实值（原始空间）
        y_pred: 预测均值（原始空间）
        y_std: 预测标准差（log空间）
        confidence: 置信水平 (默认95%)

    返回:
        覆盖率 (0-1之间)
    """
    from scipy import stats
    z = stats.norm.ppf((1 + confidence) / 2)  # 95% -> z ≈ 1.96

    # 在log空间构建置信区间
    log_pred = log_transform(y_pred)
    log_true = log_transform(y_true)

    lower = log_pred - z * y_std
    upper = log_pred + z * y_std

    covered = (log_true >= lower) & (log_true <= upper)
    return np.mean(covered)


def compute_mpiw(y_std, confidence=0.95):
    """计算MPIW (Mean Prediction Interval Width - 平均区间宽度)

    log空间的区间宽度 = 2 * z * σ
    """
    from scipy import stats
    z = stats.norm.ppf((1 + confidence) / 2)

    widths = 2 * z * y_std
    return np.mean(widths)


def evaluate(y_true, y_pred, prefix='', y_std=None):
    """计算评估指标

    基础指标: MSE, RMSE, MAE, R2, MSLE, ACP
    不确定性指标 (需要y_std): Log-NLL, PICP, MPIW
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    msle = compute_msle(y_true, y_pred)
    acp_20 = compute_acp(y_true, y_pred, alpha=0.2, delta=5)  # 20%容忍度
    acp_50 = compute_acp(y_true, y_pred, alpha=0.5, delta=5)  # 50%容忍度

    metrics = {
        f'{prefix}MSE': mse,
        f'{prefix}RMSE': rmse,
        f'{prefix}MAE': mae,
        f'{prefix}R2': r2,
        f'{prefix}MSLE': msle,
        f'{prefix}ACP@20%': acp_20,
        f'{prefix}ACP@50%': acp_50,
    }

    # 如果有不确定性估计，计算额外指标
    if y_std is not None:
        log_nll = compute_log_nll(y_true, y_pred, y_std)
        picp = compute_picp(y_true, y_pred, y_std, confidence=0.95)
        mpiw = compute_mpiw(y_std, confidence=0.95)

        metrics[f'{prefix}LogNLL'] = log_nll
        metrics[f'{prefix}PICP@95%'] = picp
        metrics[f'{prefix}MPIW'] = mpiw

    return metrics


# ==================== 结果保存 ====================
def get_result_dir(model_name):
    """获取结果保存目录"""
    base_name = f"comment_pred_{model_name}"

    # 查找已存在的目录编号
    existing = list(RESULTS_BASE.glob(f"{base_name}_*"))
    if existing:
        nums = [int(str(p).split('_')[-1]) for p in existing if str(p).split('_')[-1].isdigit()]
        next_num = max(nums) + 1 if nums else 1
    else:
        next_num = 1

    result_dir = RESULTS_BASE / f"{base_name}_{next_num}"
    result_dir.mkdir(parents=True, exist_ok=True)

    return result_dir


def save_results(result_dir, model, metrics, args, X_test, y_test, y_pred, y_std=None, feature_cols=None):
    """保存训练结果"""
    print(f"\n【保存结果到 {result_dir}】")

    # 1. 保存模型
    model_path = result_dir / 'model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  模型已保存: {model_path}")

    # 2. 保存指标
    metrics_path = result_dir / 'metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  指标已保存: {metrics_path}")

    # 3. 保存参数
    args_dict = vars(args)
    args_path = result_dir / 'args.json'
    with open(args_path, 'w', encoding='utf-8') as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)
    print(f"  参数已保存: {args_path}")

    # 4. 绘制预测效果图
    if y_std is not None:
        # NGBoost: 包含不确定性的可视化
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 实际值 vs 预测值（带误差棒）
        ax = axes[0, 0]
        # 采样展示（数据太多时）
        sample_size = min(1000, len(y_test))
        idx = np.random.choice(len(y_test), sample_size, replace=False)
        ax.errorbar(y_test[idx], y_pred[idx], yerr=y_std[idx], fmt='o', alpha=0.3,
                    markersize=2, elinewidth=0.5, capsize=0)
        max_val = min(y_test.max(), 100)
        ax.plot([0, max_val], [0, max_val], 'r--', label='完美预测', linewidth=2)
        ax.set_xlabel('实际子评论数')
        ax.set_ylabel('预测子评论数')
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val * 0.8)
        ax.legend()

        # 残差分布
        ax = axes[0, 1]
        residuals = y_test - y_pred
        ax.hist(residuals, bins=100, edgecolor='white', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_xlabel('残差 (实际 - 预测)')
        ax.set_ylabel('频数')
        ax.set_xlim(-50, 50)

        # 预测标准差分布
        ax = axes[1, 0]
        ax.hist(y_std, bins=100, edgecolor='white', alpha=0.7, color='orange')
        ax.axvline(x=y_std.mean(), color='r', linestyle='--', label=f'均值: {y_std.mean():.2f}')
        ax.set_xlabel('预测标准差 (不确定性)')
        ax.set_ylabel('频数')
        ax.legend()

        # 标准化残差分布 (z-score)
        ax = axes[1, 1]
        z_scores = (y_test - y_pred) / (y_std + 1e-6)
        ax.hist(z_scores, bins=100, edgecolor='white', alpha=0.7, color='green', density=True)
        # 叠加标准正态分布
        x_norm = np.linspace(-5, 5, 100)
        from scipy.stats import norm
        ax.plot(x_norm, norm.pdf(x_norm), 'r--', linewidth=2, label='标准正态分布')
        ax.set_xlabel('标准化残差 (z-score)')
        ax.set_ylabel('密度')
        ax.set_xlim(-5, 5)
        ax.legend()

    else:
        # 普通模型可视化
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 实际值 vs 预测值
        ax = axes[0]
        ax.scatter(y_test, y_pred, alpha=0.1, s=1)
        max_val = min(y_test.max(), y_pred.max(), 100)
        ax.plot([0, max_val], [0, max_val], 'r--', label='完美预测')
        ax.set_xlabel('实际子评论数')
        ax.set_ylabel('预测子评论数')
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val * 0.5)
        ax.legend()

        # 残差分布
        ax = axes[1]
        residuals = y_test - y_pred
        ax.hist(residuals, bins=100, edgecolor='white', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_xlabel('残差 (实际 - 预测)')
        ax.set_ylabel('频数')
        ax.set_xlim(-50, 50)

    plt.tight_layout()
    fig_path = result_dir / 'prediction_plot.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {fig_path}")

    # 5. 保存特征重要性（如果支持）
    if hasattr(model, 'get_feature_importance') and feature_cols is not None:
        importance = model.get_feature_importance()
        importance_df = pd.DataFrame({
            '特征': feature_cols,
            '重要性': importance
        }).sort_values('重要性', ascending=False)

        importance_path = result_dir / 'feature_importance.csv'
        importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
        print(f"  特征重要性已保存: {importance_path}")

        print("\n特征重要性:")
        for _, row in importance_df.iterrows():
            bar = '█' * int(row['重要性'] * 50)
            print(f"  {row['特征']}: {row['重要性']:.4f} {bar}")


# ==================== 主训练流程 ====================
def train(args):
    """主训练流程"""
    # 解析特征组
    if args.features == 'all':
        feature_groups = ['base', 'text', 'lda', 'density']
    else:
        feature_groups = [g.strip() for g in args.features.split(',')]

    # 获取特征列
    feature_cols = get_feature_cols(feature_groups)

    print("=" * 60)
    print(f"【训练模型: {args.model}】")
    print("=" * 60)
    print(f"特征组: {feature_groups}")
    print(f"特征列 ({len(feature_cols)}个): {feature_cols}")
    print(f"标签: {TARGET_COL}")

    # 1. 加载数据
    train_df, val_df, test_df = load_data(use_pkl=True)

    # 2. 加载外部特征（如需要）
    train_lda, val_lda, test_lda = None, None, None
    train_density, val_density, test_density = None, None, None

    if 'lda' in feature_groups:
        train_lda, val_lda, test_lda = load_lda_features()
        if train_lda is None:
            feature_groups.remove('lda')
            feature_cols = get_feature_cols(feature_groups)
            print(f"  更新特征列: {feature_cols}")

    if 'density' in feature_groups:
        train_density, val_density, test_density = load_density_features(args.density_method)
        if train_density is None:
            feature_groups.remove('density')
            feature_cols = get_feature_cols(feature_groups)
            print(f"  更新特征列: {feature_cols}")

    # 3. 特征工程
    print("\n【特征工程】")

    # 基础时间特征
    if 'base' in feature_groups:
        print("  提取时间特征...")
        train_df = extract_time_features(train_df)
        val_df = extract_time_features(val_df)
        test_df = extract_time_features(test_df)

    # 文本特征
    if 'text' in feature_groups:
        print("  提取文本特征...")
        train_df = extract_text_features(train_df)
        val_df = extract_text_features(val_df)
        test_df = extract_text_features(test_df)

    # 合并外部特征
    if 'lda' in feature_groups or 'density' in feature_groups:
        print("  合并外部特征...")
        train_df = merge_external_features(train_df, train_lda, train_density)
        val_df = merge_external_features(val_df, val_lda, val_density)
        test_df = merge_external_features(test_df, test_lda, test_density)

    # 4. 准备特征矩阵
    print("\n【准备特征】")
    X_train, y_train = prepare_features(train_df, feature_cols)
    X_val, y_val = prepare_features(val_df, feature_cols)
    X_test, y_test = prepare_features(test_df, feature_cols)

    print(f"训练集特征形状: {X_train.shape}")
    print(f"验证集特征形状: {X_val.shape}")
    print(f"测试集特征形状: {X_test.shape}")

    # 5. 创建模型
    if args.model not in MODEL_REGISTRY:
        print(f"错误: 未知模型 '{args.model}'")
        print(f"可用模型: {list(MODEL_REGISTRY.keys())}")
        return

    model_cls = MODEL_REGISTRY[args.model]

    # BGE_NN模型需要特殊处理
    if args.model == 'bge_nn':
        model = model_cls(
            freeze_bert=not args.finetune_bge,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience
        )
        print(f"\n模型: {model.name}")
        print(f"  冻结BGE: {model.freeze_bert}")
        print(f"  隐藏层大小: {model.hidden_size}")
        print(f"  Dropout: {model.dropout}")
        print(f"  Epochs: {model.epochs}")
        print(f"  Batch Size: {model.batch_size}")
        print(f"  学习率: {model.learning_rate}")

        # 6. 训练（BGE_NN使用DataFrame）
        print("\n【训练中...】")
        model.fit(train_df, val_df, train_density, val_density)
        print("训练完成!")

        # 7. 评估（BGE_NN使用DataFrame）
        print("\n【评估结果】")
        y_train_pred = model.predict(train_df, train_density)
        y_val_pred = model.predict(val_df, val_density)
        y_test_pred = model.predict(test_df, test_density)

        # 获取不确定性估计
        _, y_train_std = model.predict_dist(train_df, train_density)
        _, y_val_std = model.predict_dist(val_df, val_density)
        _, y_test_std = model.predict_dist(test_df, test_density)

        # 获取真实值
        y_train = train_df['子评论数'].values
        y_val = val_df['子评论数'].values
        y_test = test_df['子评论数'].values

        # 计算评估指标
        train_metrics = evaluate(y_train, y_train_pred, prefix='train_', y_std=y_train_std)
        val_metrics = evaluate(y_val, y_val_pred, prefix='val_', y_std=y_val_std)
        test_metrics = evaluate(y_test, y_test_pred, prefix='test_', y_std=y_test_std)

        all_metrics = {**train_metrics, **val_metrics, **test_metrics}
        all_metrics['model'] = 'bge_nn'
        all_metrics['freeze_bert'] = model.freeze_bert

        # 添加NLL指标
        train_nll = model.compute_nll(train_df, train_density)
        val_nll = model.compute_nll(val_df, val_density)
        test_nll = model.compute_nll(test_df, test_density)
        all_metrics['train_NLL'] = train_nll
        all_metrics['val_NLL'] = val_nll
        all_metrics['test_NLL'] = test_nll

        # 打印基础指标
        print(f"\n{'数据集':<8} {'RMSE':<10} {'MAE':<10} {'MSLE':<10} {'R2':<10}")
        print("-" * 50)
        print(f"{'训练集':<8} {train_metrics['train_RMSE']:<10.4f} {train_metrics['train_MAE']:<10.4f} {train_metrics['train_MSLE']:<10.4f} {train_metrics['train_R2']:<10.4f}")
        print(f"{'验证集':<8} {val_metrics['val_RMSE']:<10.4f} {val_metrics['val_MAE']:<10.4f} {val_metrics['val_MSLE']:<10.4f} {val_metrics['val_R2']:<10.4f}")
        print(f"{'测试集':<8} {test_metrics['test_RMSE']:<10.4f} {test_metrics['test_MAE']:<10.4f} {test_metrics['test_MSLE']:<10.4f} {test_metrics['test_R2']:<10.4f}")

        # 打印ACP指标
        print(f"\n{'数据集':<8} {'ACP@20%':<12} {'ACP@50%':<12}")
        print("-" * 35)
        print(f"{'训练集':<8} {train_metrics['train_ACP@20%']*100:<12.2f}% {train_metrics['train_ACP@50%']*100:<12.2f}%")
        print(f"{'验证集':<8} {val_metrics['val_ACP@20%']*100:<12.2f}% {val_metrics['val_ACP@50%']*100:<12.2f}%")
        print(f"{'测试集':<8} {test_metrics['test_ACP@20%']*100:<12.2f}% {test_metrics['test_ACP@50%']*100:<12.2f}%")

        # 打印不确定性指标
        print("\n【不确定性估计】")
        print(f"{'数据集':<8} {'NLL':<12} {'LogNLL':<12} {'PICP@95%':<12} {'MPIW':<10}")
        print("-" * 55)
        print(f"{'训练集':<8} {all_metrics['train_NLL']:<12.4f} {train_metrics['train_LogNLL']:<12.4f} {train_metrics['train_PICP@95%']*100:<12.2f}% {train_metrics['train_MPIW']:<10.4f}")
        print(f"{'验证集':<8} {all_metrics['val_NLL']:<12.4f} {val_metrics['val_LogNLL']:<12.4f} {val_metrics['val_PICP@95%']*100:<12.2f}% {val_metrics['val_MPIW']:<10.4f}")
        print(f"{'测试集':<8} {all_metrics['test_NLL']:<12.4f} {test_metrics['test_LogNLL']:<12.4f} {test_metrics['test_PICP@95%']*100:<12.2f}% {test_metrics['test_MPIW']:<10.4f}")

        print(f"\n预测标准差统计 (log空间):")
        print(f"  均值: {y_test_std.mean():.4f}")
        print(f"  中位数: {np.median(y_test_std):.4f}")
        print(f"  最小值: {y_test_std.min():.4f}")
        print(f"  最大值: {y_test_std.max():.4f}")

        # 8. 保存结果
        result_dir = get_result_dir(args.model)

        # 保存模型
        model_path = result_dir / 'model.pt'
        torch.save({
            'model_state_dict': model.model.state_dict(),
            'train_losses': model.train_losses,
            'val_losses': model.val_losses,
        }, model_path)
        print(f"\n模型已保存: {model_path}")

        # 保存指标
        metrics_path = result_dir / 'metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"指标已保存: {metrics_path}")

        # 保存参数
        args_dict = vars(args)
        args_path = result_dir / 'args.json'
        with open(args_path, 'w', encoding='utf-8') as f:
            json.dump(args_dict, f, indent=2, ensure_ascii=False)
        print(f"参数已保存: {args_path}")

        # 绘制训练曲线
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 损失曲线
        ax = axes[0]
        ax.plot(model.train_losses, label='训练损失')
        ax.plot(model.val_losses, label='验证损失')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('NLL Loss')
        ax.legend()

        # 预测效果图
        ax = axes[1]
        sample_size = min(1000, len(y_test))
        idx = np.random.choice(len(y_test), sample_size, replace=False)
        ax.errorbar(y_test[idx], y_test_pred[idx], yerr=y_test_std[idx], fmt='o', alpha=0.3,
                    markersize=2, elinewidth=0.5, capsize=0)
        max_val = min(y_test.max(), 100)
        ax.plot([0, max_val], [0, max_val], 'r--', label='完美预测', linewidth=2)
        ax.set_xlabel('实际子评论数')
        ax.set_ylabel('预测子评论数')
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val * 0.8)
        ax.legend()

        plt.tight_layout()
        fig_path = result_dir / 'training_plot.png'
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"图表已保存: {fig_path}")

        print("\n" + "=" * 60)
        print("【训练完成】")
        print("=" * 60)
        return

    # 其他模型的正常处理
    model = model_cls(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=args.seed
    )
    print(f"\n模型: {model.name}")

    # 6. 训练
    print("\n【训练中...】")
    model.fit(X_train, y_train)
    print("训练完成!")

    # 7. 评估
    print("\n【评估结果】")
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # NGBoost: 获取不确定性估计
    y_train_std, y_val_std, y_test_std = None, None, None
    if hasattr(model, 'supports_uncertainty') and model.supports_uncertainty:
        _, y_train_std = model.predict_dist(X_train)
        _, y_val_std = model.predict_dist(X_val)
        _, y_test_std = model.predict_dist(X_test)

    # 计算评估指标（包含不确定性指标）
    train_metrics = evaluate(y_train, y_train_pred, prefix='train_', y_std=y_train_std)
    val_metrics = evaluate(y_val, y_val_pred, prefix='val_', y_std=y_val_std)
    test_metrics = evaluate(y_test, y_test_pred, prefix='test_', y_std=y_test_std)

    all_metrics = {**train_metrics, **val_metrics, **test_metrics}
    all_metrics['feature_groups'] = ','.join(feature_groups)
    all_metrics['feature_cols'] = feature_cols

    # NGBoost: 添加NLL指标
    if hasattr(model, 'supports_uncertainty') and model.supports_uncertainty:
        train_nll = model.compute_nll(X_train, y_train)
        val_nll = model.compute_nll(X_val, y_val)
        test_nll = model.compute_nll(X_test, y_test)

        all_metrics['train_NLL'] = train_nll
        all_metrics['val_NLL'] = val_nll
        all_metrics['test_NLL'] = test_nll

    # 打印基础指标
    print(f"\n{'数据集':<8} {'RMSE':<10} {'MAE':<10} {'MSLE':<10} {'R2':<10}")
    print("-" * 50)
    print(f"{'训练集':<8} {train_metrics['train_RMSE']:<10.4f} {train_metrics['train_MAE']:<10.4f} {train_metrics['train_MSLE']:<10.4f} {train_metrics['train_R2']:<10.4f}")
    print(f"{'验证集':<8} {val_metrics['val_RMSE']:<10.4f} {val_metrics['val_MAE']:<10.4f} {val_metrics['val_MSLE']:<10.4f} {val_metrics['val_R2']:<10.4f}")
    print(f"{'测试集':<8} {test_metrics['test_RMSE']:<10.4f} {test_metrics['test_MAE']:<10.4f} {test_metrics['test_MSLE']:<10.4f} {test_metrics['test_R2']:<10.4f}")

    # 打印ACP指标
    print(f"\n{'数据集':<8} {'ACP@20%':<12} {'ACP@50%':<12}")
    print("-" * 35)
    print(f"{'训练集':<8} {train_metrics['train_ACP@20%']*100:<12.2f}% {train_metrics['train_ACP@50%']*100:<12.2f}%")
    print(f"{'验证集':<8} {val_metrics['val_ACP@20%']*100:<12.2f}% {val_metrics['val_ACP@50%']*100:<12.2f}%")
    print(f"{'测试集':<8} {test_metrics['test_ACP@20%']*100:<12.2f}% {test_metrics['test_ACP@50%']*100:<12.2f}%")

    # NGBoost: 打印不确定性指标
    if hasattr(model, 'supports_uncertainty') and model.supports_uncertainty:
        print("\n【不确定性估计】")
        print(f"{'数据集':<8} {'NLL':<12} {'LogNLL':<12} {'PICP@95%':<12} {'MPIW':<10}")
        print("-" * 55)
        print(f"{'训练集':<8} {all_metrics['train_NLL']:<12.4f} {train_metrics['train_LogNLL']:<12.4f} {train_metrics['train_PICP@95%']*100:<12.2f}% {train_metrics['train_MPIW']:<10.4f}")
        print(f"{'验证集':<8} {all_metrics['val_NLL']:<12.4f} {val_metrics['val_LogNLL']:<12.4f} {val_metrics['val_PICP@95%']*100:<12.2f}% {val_metrics['val_MPIW']:<10.4f}")
        print(f"{'测试集':<8} {all_metrics['test_NLL']:<12.4f} {test_metrics['test_LogNLL']:<12.4f} {test_metrics['test_PICP@95%']*100:<12.2f}% {test_metrics['test_MPIW']:<10.4f}")

        print(f"\n预测标准差统计 (log空间):")
        print(f"  均值: {y_test_std.mean():.4f}")
        print(f"  中位数: {np.median(y_test_std):.4f}")
        print(f"  最小值: {y_test_std.min():.4f}")
        print(f"  最大值: {y_test_std.max():.4f}")

    # 8. 保存结果
    result_dir = get_result_dir(args.model)
    save_results(result_dir, model, all_metrics, args, X_test, y_test, y_test_pred,
                 y_std=y_test_std, feature_cols=feature_cols)

    print("\n" + "=" * 60)
    print("【训练完成】")
    print("=" * 60)


# ==================== 命令行参数 ====================
def parse_args():
    parser = argparse.ArgumentParser(description='评论数预测模型训练')

    # 模型选择
    parser.add_argument('--model', type=str, default='rf',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='模型类型 (default: rf)')

    # 特征组选择
    parser.add_argument('--features', type=str, default='base,text',
                        help='特征组，用逗号分隔: base,text,lda,density 或 all (default: base,text)')

    # 时间密度特征方法
    parser.add_argument('--density_method', type=str, default='minhash',
                        choices=['bge', 'minhash'],
                        help='时间密度特征计算方法: bge(语义相似度) 或 minhash(Jaccard相似度) (default: minhash)')

    # 模型参数
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='树的数量 (default: 100)')
    parser.add_argument('--max_depth', type=int, default=10,
                        help='树的最大深度 (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='学习率 (default: 0.1)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='正则化系数，用于Ridge/Lasso (default: 1.0)')

    # BGE神经网络模型参数
    parser.add_argument('--finetune_bge', action='store_true',
                        help='是否微调BGE模型 (default: False，冻结BGE)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数，用于bge_nn (default: 30)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小，用于bge_nn (default: 32)')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='隐藏层大小，用于bge_nn (default: 256)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout率，用于bge_nn (default: 0.1)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping耐心值 (default: 5)')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (default: 42)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
