"""
BGE神经网络模型：使用BGE-base-zh-v1.5进行文本编码的深度学习模型

包含:
- BERT相关类（Embeddings, SelfAttention, Layer, Model）
- CrossAttentionFusion: 跨注意力融合层
- DualPredictionHead: 双预测头（均值+方差）
- CommentPredictorNN: 评论预测神经网络
- BGENNModel: 封装类
"""
import os
import re
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ..config import (
    BGE_MODEL_PATH, LOG_OFFSET,
    VIP_USERS, VIP_USER_TO_ID, SPECIAL_TOKEN_USER,
    WEIBO_EMOJI_LIST, WEIBO_EMOJI_TO_ID,
    UNICODE_EMOJI_LIST, UNICODE_EMOJI_TO_ID,
    XIAOMI_EMBED_KEYWORDS, XIAOMI_KEYWORD_TO_ID,
    TOTAL_SPECIAL_EMBEDDINGS,
    VIP_EMBED_OFFSET, WEIBO_EMOJI_EMBED_OFFSET, UNICODE_EMOJI_EMBED_OFFSET,
    XIAOMI_EMBED_OFFSET, USER_EMBED_ID
)


# ==================== 特殊Token提取 ====================
def extract_special_token_ids(text):
    """从文本中提取特殊token的ID列表

    提取内容:
    1. @VIP用户 -> 对应VIP嵌入ID
    2. @普通用户 -> 统一USER嵌入ID
    3. [微博表情] -> 微博表情嵌入ID
    4. Unicode Emoji (😂🌿等) -> Unicode表情嵌入ID
    5. 小米关键词 -> 关键词嵌入ID

    返回:
        special_ids: 出现的特殊token ID列表（不重复）
    """
    if not text or pd.isna(text):
        return []

    text = str(text)
    special_ids = set()

    # 1. 提取@用户
    at_pattern = re.compile(r'@([^\s@:：,，。！!?？\[\]]+)')
    for match in at_pattern.finditer(text):
        username = match.group(1)
        if username in VIP_USER_TO_ID:
            special_ids.add(VIP_EMBED_OFFSET + VIP_USER_TO_ID[username])
        else:
            special_ids.add(USER_EMBED_ID)  # 非VIP用户

    # 2. 提取微博方括号表情 [xxx]
    weibo_emoji_pattern = re.compile(r'\[([^\[\]]+)\]')
    for match in weibo_emoji_pattern.finditer(text):
        emoji_name = match.group(1)
        if emoji_name in WEIBO_EMOJI_TO_ID:
            special_ids.add(WEIBO_EMOJI_EMBED_OFFSET + WEIBO_EMOJI_TO_ID[emoji_name])

    # 3. 提取Unicode Emoji（真实emoji字符）
    for emoji in UNICODE_EMOJI_LIST:
        if emoji in text:
            special_ids.add(UNICODE_EMOJI_EMBED_OFFSET + UNICODE_EMOJI_TO_ID[emoji])

    # 4. 提取小米关键词
    text_lower = text.lower()
    for keyword in XIAOMI_EMBED_KEYWORDS:
        if keyword.lower() in text_lower:
            special_ids.add(XIAOMI_EMBED_OFFSET + XIAOMI_KEYWORD_TO_ID[keyword])

    return list(special_ids)


# ==================== 文本预处理 ====================
def preprocess_text_for_bge(text, replace_users=True):
    """预处理文本，用于BGE编码

    处理:
    1. @用户 -> 保留VIP用户，其他替换为 _USER_
    2. 保留表情、特殊字符（由BGE模型处理）

    参数:
        text: 原始文本
        replace_users: 是否替换非VIP用户

    返回:
        处理后的文本
    """
    if not text or pd.isna(text):
        return ""

    text = str(text)

    if replace_users:
        # 匹配 @用户名 模式
        def replace_user(match):
            username = match.group(1)
            if username in VIP_USER_TO_ID:
                return f"@{username}"  # 保留VIP用户
            return SPECIAL_TOKEN_USER  # 替换为特殊token

        text = re.sub(r'@([^\s@]+)', replace_user, text)

    return text.strip()


# ==================== BERT模型组件 ====================
class BertEmbeddings(nn.Module):
    """BERT Embeddings层"""
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config['vocab_size'], config['hidden_size'], padding_idx=0)
        self.position_embeddings = nn.Embedding(config['max_position_embeddings'], config['hidden_size'])
        self.token_type_embeddings = nn.Embedding(config['type_vocab_size'], config['hidden_size'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        embeddings = self.word_embeddings(input_ids)
        embeddings += self.position_embeddings(position_ids)
        embeddings += self.token_type_embeddings(token_type_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    """BERT Self-Attention层"""
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config['num_attention_heads']
        self.attention_head_size = config['hidden_size'] // config['num_attention_heads']
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config['hidden_size'], self.all_head_size)
        self.key = nn.Linear(config['hidden_size'], self.all_head_size)
        self.value = nn.Linear(config['hidden_size'], self.all_head_size)
        self.dropout = nn.Dropout(config['attention_probs_dropout_prob'])

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        return context_layer


class BertLayer(nn.Module):
    """BERT Transformer层"""
    def __init__(self, config):
        super().__init__()
        self.attention = BertSelfAttention(config)
        self.attention_output = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.attention_norm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.intermediate = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.output = nn.Linear(config['intermediate_size'], config['hidden_size'])
        self.output_norm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.dropout(self.attention_output(attention_output))
        hidden_states = self.attention_norm(hidden_states + attention_output)

        intermediate_output = F.gelu(self.intermediate(hidden_states))
        layer_output = self.dropout(self.output(intermediate_output))
        hidden_states = self.output_norm(hidden_states + layer_output)
        return hidden_states


class BertModel(nn.Module):
    """BERT模型（仅编码器）"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(config['num_hidden_layers'])])
        self.pooler = nn.Linear(config['hidden_size'], config['hidden_size'])

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        hidden_states = self.embeddings(input_ids, token_type_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, extended_attention_mask)

        pooled_output = torch.tanh(self.pooler(hidden_states[:, 0]))
        return pooled_output, hidden_states


# ==================== 特殊Token嵌入层 ====================
class SpecialTokenEmbedding(nn.Module):
    """特殊Token可训练嵌入层

    为VIP用户、表情符号、小米关键词提供独立的可训练嵌入。
    这些嵌入与BGE无关，可以在BGE冻结时单独训练。
    """
    def __init__(self, num_embeddings=TOTAL_SPECIAL_EMBEDDINGS, embedding_dim=768, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # Xavier初始化
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, special_ids, special_mask):
        """
        参数:
            special_ids: [batch, max_special_tokens] 特殊token ID
            special_mask: [batch, max_special_tokens] 有效位置掩码 (1=有效, 0=padding)

        返回:
            pooled: [batch, embedding_dim] 池化后的特殊token嵌入
        """
        # 获取嵌入 [batch, max_special_tokens, embedding_dim]
        embeddings = self.embedding(special_ids)
        embeddings = self.dropout(embeddings)

        # 掩码池化：对有效token取平均
        mask_expanded = special_mask.unsqueeze(-1).float()  # [batch, max_special, 1]
        sum_embeddings = (embeddings * mask_expanded).sum(dim=1)  # [batch, dim]
        count = special_mask.sum(dim=1, keepdim=True).float().clamp(min=1)  # [batch, 1]
        pooled = sum_embeddings / count  # [batch, dim]

        return pooled


# ==================== 注意力融合与预测头 ====================
class CrossAttentionFusion(nn.Module):
    """跨注意力融合层

    将评论embedding作为Query，上下文（微博/根评论/父评论）作为Key/Value
    """
    def __init__(self, hidden_size=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, comment_emb, context_embs):
        """
        参数:
            comment_emb: [batch, 768] 评论embedding
            context_embs: [batch, 3, 768] 上下文embeddings (微博, 根评论, 父评论)

        返回:
            fused: [batch, 768] 融合后的embedding
        """
        # 扩展comment_emb为 [batch, 1, 768] 作为Query
        query = comment_emb.unsqueeze(1)

        # context_embs作为Key和Value
        attn_output, _ = self.attention(query, context_embs, context_embs)

        # 残差连接
        fused = self.norm(comment_emb + self.dropout(attn_output.squeeze(1)))
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
        self.mu_head = nn.Linear(hidden_size // 2, 1)
        self.sigma_head = nn.Linear(hidden_size // 2, 1)

        # 初始化：让mu初始输出接近log(10)≈2.3（对应子评论数为0）
        # sigma初始输出接近1（合理的不确定性）
        nn.init.zeros_(self.mu_head.weight)
        nn.init.constant_(self.mu_head.bias, 2.3)  # log(10) ≈ 2.3
        nn.init.zeros_(self.sigma_head.weight)
        nn.init.constant_(self.sigma_head.bias, 0.5)  # softplus(0.5) ≈ 0.97

    def forward(self, x):
        """
        返回:
            mu: [batch] 预测均值（log空间）
            sigma: [batch] 预测标准差（log空间，通过softplus保证正值）
        """
        shared = self.shared(x)
        mu = self.mu_head(shared).squeeze(-1)
        sigma = F.softplus(self.sigma_head(shared)).squeeze(-1) + 1e-4
        return mu, sigma


# ==================== Mini轻量化模型 ====================
class CommentPredictorMini(nn.Module):
    """轻量化评论预测神经网络

    相比完整版CommentPredictorNN的简化:
    1. 移除Cross-Attention，使用简单的加权平均融合
    2. 更小的隐藏层维度（128 vs 256）
    3. 可选只使用评论文本（不使用微博/根评论/父评论）
    4. 更少的参数量，更快的训练和推理速度

    适用场景:
    - 快速实验和原型验证
    - 资源受限环境
    - 作为基线模型对比
    """
    def __init__(self, bert_model, num_numeric_features, hidden_size=128, dropout=0.1,
                 freeze_bert=True, use_special_embeddings=True, use_context=True):
        super().__init__()
        self.bert = bert_model
        self.freeze_bert = freeze_bert
        self.use_special_embeddings = use_special_embeddings
        self.use_context = use_context  # 是否使用上下文文本（微博/根评论/父评论）

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # 特殊Token嵌入层（可选，维度更小）
        if use_special_embeddings:
            self.special_embedding = SpecialTokenEmbedding(
                num_embeddings=TOTAL_SPECIAL_EMBEDDINGS,
                embedding_dim=64,  # Mini版使用更小的维度
                dropout=dropout
            )
            special_dim = 64
        else:
            self.special_embedding = None
            special_dim = 0

        # 文本嵌入维度
        if use_context:
            # 使用可学习的融合权重
            self.fusion_weights = nn.Parameter(torch.ones(4) / 4)
            text_dim = 768
        else:
            # 只使用评论文本
            text_dim = 768

        # 文本降维投影（768 -> hidden_size）
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 数值特征投影
        self.numeric_proj = nn.Sequential(
            nn.Linear(num_numeric_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 预测头
        total_dim = hidden_size + 32 + special_dim
        self.prediction_head = nn.Sequential(
            nn.Linear(total_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 输出层
        self.mu_head = nn.Linear(hidden_size, 1)
        self.sigma_head = nn.Linear(hidden_size, 1)

        # 初始化输出层
        nn.init.zeros_(self.mu_head.weight)
        nn.init.constant_(self.mu_head.bias, 2.3)
        nn.init.zeros_(self.sigma_head.weight)
        nn.init.constant_(self.sigma_head.bias, 0.5)

    def encode_text(self, input_ids, attention_mask):
        """编码单个文本，返回[CLS] embedding"""
        pooled_output, _ = self.bert(input_ids, attention_mask)
        return pooled_output

    def forward(self, comment_ids, comment_mask, weibo_ids, weibo_mask,
                root_ids, root_mask, parent_ids, parent_mask, numeric_features,
                special_ids=None, special_mask=None):
        """
        参数与CommentPredictorNN相同，保持接口一致
        """
        # 编码评论文本
        comment_emb = self.encode_text(comment_ids, comment_mask)

        if self.use_context:
            # 编码上下文文本
            weibo_emb = self.encode_text(weibo_ids, weibo_mask)
            root_emb = self.encode_text(root_ids, root_mask)
            parent_emb = self.encode_text(parent_ids, parent_mask)

            # 加权平均融合（可学习权重）
            weights = F.softmax(self.fusion_weights, dim=0)
            text_fused = (weights[0] * comment_emb +
                         weights[1] * weibo_emb +
                         weights[2] * root_emb +
                         weights[3] * parent_emb)
        else:
            # 只使用评论文本
            text_fused = comment_emb

        # 文本降维投影
        text_proj = self.text_proj(text_fused)

        # 数值特征投影
        numeric_proj = self.numeric_proj(numeric_features)

        # 特殊Token嵌入
        if self.use_special_embeddings and special_ids is not None:
            special_emb = self.special_embedding(special_ids, special_mask)
            combined = torch.cat([text_proj, numeric_proj, special_emb], dim=1)
        else:
            combined = torch.cat([text_proj, numeric_proj], dim=1)

        # 预测
        hidden = self.prediction_head(combined)
        mu = self.mu_head(hidden).squeeze(-1)
        sigma = F.softplus(self.sigma_head(hidden)).squeeze(-1) + 1e-4

        return mu, sigma


class CommentPredictorNN(nn.Module):
    """评论预测神经网络

    结构:
    1. BGE编码4个文本（评论/微博/根评论/父评论）
    2. 特殊Token嵌入（VIP用户/表情/关键词，可训练，独立于BGE）
    3. Cross-Attention融合评论与上下文
    4. 拼接数值特征和特殊嵌入
    5. 双预测头输出均值和方差
    """
    def __init__(self, bert_model, num_numeric_features, hidden_size=256, dropout=0.1,
                 freeze_bert=True, use_special_embeddings=True):
        super().__init__()
        self.bert = bert_model
        self.freeze_bert = freeze_bert
        self.use_special_embeddings = use_special_embeddings

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # 特殊Token嵌入层（独立于BGE，始终可训练）
        if use_special_embeddings:
            self.special_embedding = SpecialTokenEmbedding(
                num_embeddings=TOTAL_SPECIAL_EMBEDDINGS,
                embedding_dim=128,  # 较小的维度，避免过拟合
                dropout=dropout
            )
            special_dim = 128
        else:
            self.special_embedding = None
            special_dim = 0

        # Cross-Attention融合层
        self.fusion = CrossAttentionFusion(hidden_size=768, num_heads=8, dropout=dropout)

        # 数值特征投影
        self.numeric_proj = nn.Sequential(
            nn.Linear(num_numeric_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 双预测头（输入维度 = 768 + 64 + special_dim）
        self.prediction_head = DualPredictionHead(
            input_size=768 + 64 + special_dim,
            hidden_size=hidden_size,
            dropout=dropout
        )

    def encode_text(self, input_ids, attention_mask):
        """编码单个文本，返回[CLS] embedding"""
        pooled_output, _ = self.bert(input_ids, attention_mask)
        return pooled_output

    def forward(self, comment_ids, comment_mask, weibo_ids, weibo_mask,
                root_ids, root_mask, parent_ids, parent_mask, numeric_features,
                special_ids=None, special_mask=None):
        """
        参数:
            comment_ids, comment_mask: 评论文案的tokenized输入
            weibo_ids, weibo_mask: 微博文案的tokenized输入
            root_ids, root_mask: 根评论文案的tokenized输入
            parent_ids, parent_mask: 父评论文案的tokenized输入
            numeric_features: [batch, num_numeric_features] 数值特征
            special_ids: [batch, max_special] 特殊token ID（可选）
            special_mask: [batch, max_special] 特殊token掩码（可选）

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

        # 特殊Token嵌入
        if self.use_special_embeddings and special_ids is not None:
            special_emb = self.special_embedding(special_ids, special_mask)
            # 拼接所有特征
            combined = torch.cat([text_fused, numeric_proj, special_emb], dim=1)
        else:
            # 拼接文本和数值特征
            combined = torch.cat([text_fused, numeric_proj], dim=1)

        # 双预测头
        mu, sigma = self.prediction_head(combined)

        return mu, sigma


# ==================== NLL损失函数 ====================
def nll_loss(y_true, mu, sigma):
    """对数尺度NLL损失（数值稳定版）

    L = 0.5 * log(σ²) + (log(y+10) - μ)² / (2σ²)

    参数:
        y_true: 真实值（原始空间）
        mu: 预测均值（log空间）
        sigma: 预测标准差（log空间）
    """
    # 确保y_true非负
    y_true = torch.clamp(y_true, min=0)
    y_log = torch.log(y_true + LOG_OFFSET)

    # 限制sigma范围，避免数值问题
    sigma = torch.clamp(sigma, min=1e-4, max=100.0)

    # 限制mu范围，避免极端值
    mu = torch.clamp(mu, min=-10.0, max=20.0)

    # 计算NLL
    nll = 0.5 * torch.log(sigma ** 2 + 1e-8) + ((y_log - mu) ** 2) / (2 * sigma ** 2 + 1e-8)

    return nll.mean()


# ==================== BGENNModel封装类 ====================
class BGENNModel:
    """BGE + 神经网络预测模型

    使用BGE-base-zh-v1.5编码4个文本（评论/微博/根评论/父评论），
    通过Cross-Attention融合，结合数值特征，双预测头输出均值和方差。
    支持可训练的特殊Token嵌入（VIP用户/表情/关键词）。
    支持BF16混合精度训练（需GPU支持）。
    """
    def __init__(self, freeze_bert=True, hidden_size=256, dropout=0.1,
                 use_special_embeddings=True, use_bf16=False, **kwargs):
        self.name = 'BGE_NN'
        self.freeze_bert = freeze_bert
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_special_embeddings = use_special_embeddings
        self.use_bf16 = use_bf16  # BF16混合精度训练（默认关闭）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.supports_uncertainty = True
        self.use_log_target = True

        # 检查BF16支持
        if self.use_bf16:
            if not torch.cuda.is_available():
                print("警告: BF16需要CUDA支持，已自动禁用")
                self.use_bf16 = False
            elif not torch.cuda.is_bf16_supported():
                print("警告: 当前GPU不支持BF16，已自动禁用")
                self.use_bf16 = False

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
        bert_model = BertModel(config)

        # 加载预训练权重
        state_dict = torch.load(
            os.path.join(model_path, 'pytorch_model.bin'),
            map_location='cpu'
        )

        # 映射权重名称
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith('bert.'):
                new_key = key[5:]
            if 'encoder.layer' in new_key:
                new_key = new_key.replace('encoder.layer', 'layers')
            if 'attention.self' in new_key:
                new_key = new_key.replace('attention.self', 'attention')
            if 'attention.output.dense' in new_key:
                new_key = new_key.replace('attention.output.dense', 'attention_output')
            if 'attention.output.LayerNorm' in new_key:
                new_key = new_key.replace('attention.output.LayerNorm', 'attention_norm')
            if 'intermediate.dense' in new_key:
                new_key = new_key.replace('intermediate.dense', 'intermediate')
            if 'output.dense' in new_key and 'attention' not in new_key:
                new_key = new_key.replace('output.dense', 'output')
            if 'output.LayerNorm' in new_key and 'attention' not in new_key:
                new_key = new_key.replace('output.LayerNorm', 'output_norm')
            if 'pooler.dense' in new_key:
                new_key = new_key.replace('pooler.dense', 'pooler')
            new_state_dict[new_key] = value

        # 加载权重
        missing, unexpected = bert_model.load_state_dict(new_state_dict, strict=False)
        print(f"BGE权重加载完成，匹配: {len(new_state_dict) - len(missing)}/{len(new_state_dict)}")

        return bert_model

    def fit(self, train_df, val_df, train_density=None, val_density=None, save_dir=None,
            test_df=None, test_density=None):
        """训练模型

        参数:
            train_df: 训练数据
            val_df: 验证数据
            train_density: 训练集密度特征
            val_density: 验证集密度特征
            save_dir: 权重保存目录（如果提供，每个epoch后保存best和last权重）
            test_df: 测试数据（可选，提前分词以加速评估）
            test_density: 测试集密度特征
        """
        from ..data.dataset import CommentDataset

        print(f"\n使用设备: {self.device}")
        print(f"冻结BGE: {self.freeze_bert}")

        # 加载BGE模型
        bert_model = self._load_bge_model()

        # 创建数据集（一次性完成所有分词）
        print("创建数据集...")
        train_dataset = CommentDataset(train_df, self.tokenizer, train_density, max_length=128)
        val_dataset = CommentDataset(val_df, self.tokenizer, val_density, max_length=128)

        # 如果提供了测试集，也一并创建（避免评估时重新分词）
        if test_df is not None:
            print("创建测试数据集（预分词）...")
            self._test_dataset = CommentDataset(test_df, self.tokenizer, test_density, max_length=128)
        else:
            self._test_dataset = None

        # 保存训练/验证数据集供评估使用
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset

        # 优化的DataLoader配置
        num_workers = min(8, os.cpu_count() or 4)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False
        )

        # 创建模型
        num_numeric_features = train_dataset.numeric_features.shape[1]
        self.model = CommentPredictorNN(
            bert_model,
            num_numeric_features,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            freeze_bert=self.freeze_bert,
            use_special_embeddings=self.use_special_embeddings
        ).to(self.device)

        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
        if self.use_special_embeddings:
            special_params = sum(p.numel() for p in self.model.special_embedding.parameters())
            print(f"  特殊嵌入参数: {special_params:,} (VIP用户/表情/关键词)")

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

        # BF16混合精度训练设置
        # 注意: BF16不需要GradScaler，因为其动态范围足够大
        if self.use_bf16:
            print("启用BF16混合精度训练")
            autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        else:
            # 使用空上下文管理器（不改变精度）
            from contextlib import nullcontext
            autocast_ctx = nullcontext()

        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            # 训练
            self.model.train()
            train_loss = 0
            nan_count = 0
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}'):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                optimizer.zero_grad()

                # BF16混合精度前向传播
                with autocast_ctx:
                    mu, sigma = self.model(
                        batch['comment_ids'], batch['comment_mask'],
                        batch['weibo_ids'], batch['weibo_mask'],
                        batch['root_ids'], batch['root_mask'],
                        batch['parent_ids'], batch['parent_mask'],
                        batch['numeric_features'],
                        batch.get('special_ids'), batch.get('special_mask')
                    )
                    loss = nll_loss(batch['target'], mu, sigma)

                # NaN检测
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    if nan_count <= 3:
                        print(f"\n警告: 检测到NaN/Inf损失，跳过此批次")
                        print(f"  mu范围: [{mu.min().item():.4f}, {mu.max().item():.4f}]")
                        print(f"  sigma范围: [{sigma.min().item():.4f}, {sigma.max().item():.4f}]")
                        print(f"  target范围: [{batch['target'].min().item():.4f}, {batch['target'].max().item():.4f}]")
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

            if nan_count > 0:
                print(f"  本epoch共有 {nan_count} 个批次出现NaN/Inf，已跳过")

            train_loss /= max(len(train_loader) - nan_count, 1)
            train_losses.append(train_loss)

            # 验证
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    # BF16混合精度验证
                    with autocast_ctx:
                        mu, sigma = self.model(
                            batch['comment_ids'], batch['comment_mask'],
                            batch['weibo_ids'], batch['weibo_mask'],
                            batch['root_ids'], batch['root_mask'],
                            batch['parent_ids'], batch['parent_mask'],
                            batch['numeric_features'],
                            batch.get('special_ids'), batch.get('special_mask')
                        )
                        loss = nll_loss(batch['target'], mu, sigma)

                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            scheduler.step(val_loss)

            # 保存 last 权重（每个epoch都保存）
            if save_dir is not None:
                last_path = Path(save_dir) / 'model_last.pt'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                }, last_path)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

                # 保存 best 权重
                if save_dir is not None:
                    best_path = Path(save_dir) / 'model_best.pt'
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                    }, best_path)
                    print(f"  保存最佳模型 (val_loss={val_loss:.4f})")
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
        """预测分布参数"""
        from ..data.dataset import CommentDataset
        from contextlib import nullcontext

        dataset = CommentDataset(df, self.tokenizer, density_df, max_length=128)
        num_workers = min(4, os.cpu_count() or 2)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        # BF16推理上下文
        if self.use_bf16:
            autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        else:
            autocast_ctx = nullcontext()

        self.model.eval()
        all_mu = []
        all_sigma = []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with autocast_ctx:
                    mu, sigma = self.model(
                        batch['comment_ids'], batch['comment_mask'],
                        batch['weibo_ids'], batch['weibo_mask'],
                        batch['root_ids'], batch['root_mask'],
                        batch['parent_ids'], batch['parent_mask'],
                        batch['numeric_features'],
                        batch.get('special_ids'), batch.get('special_mask')
                    )

                # 转回原始空间，确保非负（在FP32下进行）
                mu_orig = torch.exp(torch.clamp(mu.float(), max=20.0)) - LOG_OFFSET
                mu_orig = torch.clamp(mu_orig, min=0)  # 确保预测值非负
                all_mu.append(mu_orig.cpu().numpy())
                all_sigma.append(sigma.float().cpu().numpy())

        return np.concatenate(all_mu), np.concatenate(all_sigma)

    def evaluate_all(self, df=None, density_df=None, use_cached=None):
        """一次性评估：返回预测均值、标准差和NLL（避免重复创建Dataset和分词）

        参数:
            df: DataFrame数据（如果use_cached为None则必须提供）
            density_df: 密度特征（如果use_cached为None则需要提供）
            use_cached: 使用缓存的数据集 ('train', 'val', 'test')，如果提供则忽略df和density_df
        """
        from ..data.dataset import CommentDataset
        from contextlib import nullcontext

        # 使用缓存的数据集或创建新的
        if use_cached is not None:
            if use_cached == 'train' and hasattr(self, '_train_dataset'):
                dataset = self._train_dataset
            elif use_cached == 'val' and hasattr(self, '_val_dataset'):
                dataset = self._val_dataset
            elif use_cached == 'test' and hasattr(self, '_test_dataset') and self._test_dataset is not None:
                dataset = self._test_dataset
            else:
                raise ValueError(f"缓存数据集 '{use_cached}' 不存在，请先调用fit并传入相应数据")
        else:
            dataset = CommentDataset(df, self.tokenizer, density_df, max_length=128)

        # 推理时可以使用更大的批次
        eval_batch_size = self.batch_size * 2
        num_workers = min(4, os.cpu_count() or 2)
        loader = DataLoader(
            dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        if self.use_bf16:
            autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        else:
            autocast_ctx = nullcontext()

        self.model.eval()
        all_mu = []
        all_sigma = []
        total_nll = 0
        count = 0

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with autocast_ctx:
                    mu, sigma = self.model(
                        batch['comment_ids'], batch['comment_mask'],
                        batch['weibo_ids'], batch['weibo_mask'],
                        batch['root_ids'], batch['root_mask'],
                        batch['parent_ids'], batch['parent_mask'],
                        batch['numeric_features'],
                        batch.get('special_ids'), batch.get('special_mask')
                    )
                    loss = nll_loss(batch['target'], mu, sigma)

                total_nll += loss.item() * len(batch['target'])
                count += len(batch['target'])

                mu_orig = torch.exp(torch.clamp(mu.float(), max=20.0)) - LOG_OFFSET
                mu_orig = torch.clamp(mu_orig, min=0)
                all_mu.append(mu_orig.cpu().numpy())
                all_sigma.append(sigma.float().cpu().numpy())

        y_pred = np.concatenate(all_mu)
        y_std = np.concatenate(all_sigma)
        nll = total_nll / count

        return y_pred, y_std, nll

    def compute_nll(self, df, density_df=None):
        """计算NLL损失"""
        from ..data.dataset import CommentDataset
        from contextlib import nullcontext

        dataset = CommentDataset(df, self.tokenizer, density_df, max_length=128)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # BF16推理上下文
        if self.use_bf16:
            autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        else:
            autocast_ctx = nullcontext()

        self.model.eval()
        total_nll = 0
        count = 0

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with autocast_ctx:
                    mu, sigma = self.model(
                        batch['comment_ids'], batch['comment_mask'],
                        batch['weibo_ids'], batch['weibo_mask'],
                        batch['root_ids'], batch['root_mask'],
                        batch['parent_ids'], batch['parent_mask'],
                        batch['numeric_features'],
                        batch.get('special_ids'), batch.get('special_mask')
                    )
                    loss = nll_loss(batch['target'], mu, sigma)

                total_nll += loss.item() * len(batch['target'])
                count += len(batch['target'])

        return total_nll / count


# ==================== BGEMiniModel封装类 ====================
class BGEMiniModel:
    """BGE + Mini轻量化神经网络预测模型

    相比BGENNModel的简化:
    1. 使用加权平均融合代替Cross-Attention（减少约60%参数）
    2. 更小的隐藏层维度（128 vs 256）
    3. 可选只使用评论文本（use_context=False时推理速度提升4倍）
    4. 特殊嵌入维度更小（64 vs 128）

    适用场景:
    - 快速实验和原型验证
    - 资源受限环境（显存不足）
    - 作为基线模型对比
    """
    def __init__(self, freeze_bert=True, hidden_size=128, dropout=0.1,
                 use_special_embeddings=True, use_context=True, use_bf16=False, **kwargs):
        self.name = 'BGE_Mini'
        self.freeze_bert = freeze_bert
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_special_embeddings = use_special_embeddings
        self.use_context = use_context
        self.use_bf16 = use_bf16
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.supports_uncertainty = True
        self.use_log_target = True

        # 检查BF16支持
        if self.use_bf16:
            if not torch.cuda.is_available():
                print("警告: BF16需要CUDA支持，已自动禁用")
                self.use_bf16 = False
            elif not torch.cuda.is_bf16_supported():
                print("警告: 当前GPU不支持BF16，已自动禁用")
                self.use_bf16 = False

        # 训练参数（Mini版默认更多epoch，更大学习率）
        self.epochs = kwargs.get('epochs', 50)
        self.batch_size = kwargs.get('batch_size', 64)
        self.learning_rate = kwargs.get('learning_rate', 2e-4)
        self.patience = kwargs.get('patience', 7)

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
        bert_model = BertModel(config)

        # 加载预训练权重
        state_dict = torch.load(
            os.path.join(model_path, 'pytorch_model.bin'),
            map_location='cpu'
        )

        # 映射权重名称
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith('bert.'):
                new_key = key[5:]
            if 'encoder.layer' in new_key:
                new_key = new_key.replace('encoder.layer', 'layers')
            if 'attention.self' in new_key:
                new_key = new_key.replace('attention.self', 'attention')
            if 'attention.output.dense' in new_key:
                new_key = new_key.replace('attention.output.dense', 'attention_output')
            if 'attention.output.LayerNorm' in new_key:
                new_key = new_key.replace('attention.output.LayerNorm', 'attention_norm')
            if 'intermediate.dense' in new_key:
                new_key = new_key.replace('intermediate.dense', 'intermediate')
            if 'output.dense' in new_key and 'attention' not in new_key:
                new_key = new_key.replace('output.dense', 'output')
            if 'output.LayerNorm' in new_key and 'attention' not in new_key:
                new_key = new_key.replace('output.LayerNorm', 'output_norm')
            if 'pooler.dense' in new_key:
                new_key = new_key.replace('pooler.dense', 'pooler')
            new_state_dict[new_key] = value

        # 加载权重
        missing, unexpected = bert_model.load_state_dict(new_state_dict, strict=False)
        print(f"BGE权重加载完成，匹配: {len(new_state_dict) - len(missing)}/{len(new_state_dict)}")

        return bert_model

    def fit(self, train_df, val_df, train_density=None, val_density=None, save_dir=None,
            test_df=None, test_density=None):
        """训练模型

        参数:
            train_df: 训练数据
            val_df: 验证数据
            train_density: 训练集密度特征
            val_density: 验证集密度特征
            save_dir: 权重保存目录（如果提供，每个epoch后保存best和last权重）
            test_df: 测试数据（可选，提前分词以加速评估）
            test_density: 测试集密度特征
        """
        from ..data.dataset import CommentDataset
        from contextlib import nullcontext

        print(f"\n[Mini模型] 使用设备: {self.device}")
        print(f"  冻结BGE: {self.freeze_bert}")
        print(f"  使用上下文: {self.use_context}")
        print(f"  特殊嵌入: {self.use_special_embeddings}")

        # 加载BGE模型
        bert_model = self._load_bge_model()

        # 创建数据集（一次性完成所有分词）
        print("创建数据集...")
        train_dataset = CommentDataset(train_df, self.tokenizer, train_density, max_length=128)
        val_dataset = CommentDataset(val_df, self.tokenizer, val_density, max_length=128)

        # 如果提供了测试集，也一并创建（避免评估时重新分词）
        if test_df is not None:
            print("创建测试数据集（预分词）...")
            self._test_dataset = CommentDataset(test_df, self.tokenizer, test_density, max_length=128)
        else:
            self._test_dataset = None

        # 保存训练/验证数据集供评估使用
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset

        # DataLoader配置
        num_workers = min(8, os.cpu_count() or 4)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False
        )

        # 创建Mini模型
        num_numeric_features = train_dataset.numeric_features.shape[1]
        self.model = CommentPredictorMini(
            bert_model,
            num_numeric_features,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            freeze_bert=self.freeze_bert,
            use_special_embeddings=self.use_special_embeddings,
            use_context=self.use_context
        ).to(self.device)

        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
        if self.use_special_embeddings:
            special_params = sum(p.numel() for p in self.model.special_embedding.parameters())
            print(f"  特殊嵌入参数: {special_params:,}")

        # 优化器
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=0.01
        )

        # 学习率调度
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        # BF16混合精度
        if self.use_bf16:
            print("启用BF16混合精度训练")
            autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        else:
            autocast_ctx = nullcontext()

        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            # 训练
            self.model.train()
            train_loss = 0
            nan_count = 0
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}'):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                optimizer.zero_grad()

                with autocast_ctx:
                    mu, sigma = self.model(
                        batch['comment_ids'], batch['comment_mask'],
                        batch['weibo_ids'], batch['weibo_mask'],
                        batch['root_ids'], batch['root_mask'],
                        batch['parent_ids'], batch['parent_mask'],
                        batch['numeric_features'],
                        batch.get('special_ids'), batch.get('special_mask')
                    )
                    loss = nll_loss(batch['target'], mu, sigma)

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            if nan_count > 0:
                print(f"  本epoch共有 {nan_count} 个批次出现NaN/Inf，已跳过")

            train_loss /= max(len(train_loader) - nan_count, 1)
            train_losses.append(train_loss)

            # 验证
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    with autocast_ctx:
                        mu, sigma = self.model(
                            batch['comment_ids'], batch['comment_mask'],
                            batch['weibo_ids'], batch['weibo_mask'],
                            batch['root_ids'], batch['root_mask'],
                            batch['parent_ids'], batch['parent_mask'],
                            batch['numeric_features'],
                            batch.get('special_ids'), batch.get('special_mask')
                        )
                        loss = nll_loss(batch['target'], mu, sigma)

                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            scheduler.step(val_loss)

            # 保存 last 权重（每个epoch都保存）
            if save_dir is not None:
                last_path = Path(save_dir) / 'model_last.pt'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                }, last_path)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

                # 保存 best 权重
                if save_dir is not None:
                    best_path = Path(save_dir) / 'model_best.pt'
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                    }, best_path)
                    print(f"  保存最佳模型 (val_loss={val_loss:.4f})")
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
        """预测分布参数"""
        from ..data.dataset import CommentDataset
        from contextlib import nullcontext

        dataset = CommentDataset(df, self.tokenizer, density_df, max_length=128)
        num_workers = min(4, os.cpu_count() or 2)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        if self.use_bf16:
            autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        else:
            autocast_ctx = nullcontext()

        self.model.eval()
        all_mu = []
        all_sigma = []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with autocast_ctx:
                    mu, sigma = self.model(
                        batch['comment_ids'], batch['comment_mask'],
                        batch['weibo_ids'], batch['weibo_mask'],
                        batch['root_ids'], batch['root_mask'],
                        batch['parent_ids'], batch['parent_mask'],
                        batch['numeric_features'],
                        batch.get('special_ids'), batch.get('special_mask')
                    )

                mu_orig = torch.exp(torch.clamp(mu.float(), max=20.0)) - LOG_OFFSET
                mu_orig = torch.clamp(mu_orig, min=0)
                all_mu.append(mu_orig.cpu().numpy())
                all_sigma.append(sigma.float().cpu().numpy())

        return np.concatenate(all_mu), np.concatenate(all_sigma)

    def evaluate_all(self, df=None, density_df=None, use_cached=None):
        """一次性评估：返回预测均值、标准差和NLL（避免重复创建Dataset和分词）

        参数:
            df: DataFrame数据（如果use_cached为None则必须提供）
            density_df: 密度特征（如果use_cached为None则需要提供）
            use_cached: 使用缓存的数据集 ('train', 'val', 'test')，如果提供则忽略df和density_df
        """
        from ..data.dataset import CommentDataset
        from contextlib import nullcontext

        # 使用缓存的数据集或创建新的
        if use_cached is not None:
            if use_cached == 'train' and hasattr(self, '_train_dataset'):
                dataset = self._train_dataset
            elif use_cached == 'val' and hasattr(self, '_val_dataset'):
                dataset = self._val_dataset
            elif use_cached == 'test' and hasattr(self, '_test_dataset') and self._test_dataset is not None:
                dataset = self._test_dataset
            else:
                raise ValueError(f"缓存数据集 '{use_cached}' 不存在，请先调用fit并传入相应数据")
        else:
            dataset = CommentDataset(df, self.tokenizer, density_df, max_length=128)

        # 推理时可以使用更大的批次
        eval_batch_size = self.batch_size * 2
        num_workers = min(4, os.cpu_count() or 2)
        loader = DataLoader(
            dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        if self.use_bf16:
            autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        else:
            autocast_ctx = nullcontext()

        self.model.eval()
        all_mu = []
        all_sigma = []
        total_nll = 0
        count = 0

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with autocast_ctx:
                    mu, sigma = self.model(
                        batch['comment_ids'], batch['comment_mask'],
                        batch['weibo_ids'], batch['weibo_mask'],
                        batch['root_ids'], batch['root_mask'],
                        batch['parent_ids'], batch['parent_mask'],
                        batch['numeric_features'],
                        batch.get('special_ids'), batch.get('special_mask')
                    )
                    loss = nll_loss(batch['target'], mu, sigma)

                total_nll += loss.item() * len(batch['target'])
                count += len(batch['target'])

                mu_orig = torch.exp(torch.clamp(mu.float(), max=20.0)) - LOG_OFFSET
                mu_orig = torch.clamp(mu_orig, min=0)
                all_mu.append(mu_orig.cpu().numpy())
                all_sigma.append(sigma.float().cpu().numpy())

        y_pred = np.concatenate(all_mu)
        y_std = np.concatenate(all_sigma)
        nll = total_nll / count

        return y_pred, y_std, nll

    def compute_nll(self, df, density_df=None):
        """计算NLL损失"""
        from ..data.dataset import CommentDataset
        from contextlib import nullcontext

        dataset = CommentDataset(df, self.tokenizer, density_df, max_length=128)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        if self.use_bf16:
            autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        else:
            autocast_ctx = nullcontext()

        self.model.eval()
        total_nll = 0
        count = 0

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with autocast_ctx:
                    mu, sigma = self.model(
                        batch['comment_ids'], batch['comment_mask'],
                        batch['weibo_ids'], batch['weibo_mask'],
                        batch['root_ids'], batch['root_mask'],
                        batch['parent_ids'], batch['parent_mask'],
                        batch['numeric_features'],
                        batch.get('special_ids'), batch.get('special_mask')
                    )
                    loss = nll_loss(batch['target'], mu, sigma)

                total_nll += loss.item() * len(batch['target'])
                count += len(batch['target'])

        return total_nll / count
