"""
PyTorch数据集类：用于BGE神经网络模型训练
"""
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

from ..models.bge_nn import preprocess_text_for_bge, extract_special_token_ids


class CommentDataset(Dataset):
    """评论预测数据集（优化版：支持预分词缓存）"""

    def __init__(self, df, tokenizer, density_df=None, max_length=128, max_special_tokens=20,
                 use_density_features=True, use_context=True, cache_dir=None):
        """
        参数:
            df: 原始数据DataFrame
            tokenizer: BGE tokenizer（如果使用缓存可以为None）
            density_df: 时间密度特征DataFrame（可选）
            max_length: 文本最大长度
            max_special_tokens: 特殊token最大数量
            use_density_features: 是否使用时间密度特征（消融实验：w/o 重复特征）
            use_context: 是否使用上下文文本（消融实验：w/o 上下文）
            cache_dir: 预分词缓存目录路径（如果提供则使用缓存）
        """
        self.df = df.reset_index(drop=True)
        self.max_length = max_length
        self.max_special_tokens = max_special_tokens
        self.use_density_features = use_density_features  # 消融参数
        self.use_context = use_context  # 消融参数

        # 合并时间密度特征（如果提供且启用）
        if density_df is not None and use_density_features:
            self.df = self.df.merge(density_df, on='序号', how='left')
            self.df['时间顺序索引'] = self.df['时间顺序索引'].fillna(0)
            self.df['最大相似度'] = self.df['最大相似度'].fillna(0)
            self.df['重复次数'] = self.df['重复次数'].fillna(0)

        # 提取时间特征
        self.df['发布时间'] = pd.to_datetime(self.df['发布时间'])
        self.df['发布小时'] = self.df['发布时间'].dt.hour
        self.df['发布星期'] = self.df['发布时间'].dt.dayofweek

        # 准备数值特征
        self._prepare_numeric_features()

        # 文本预处理：优先使用缓存
        if cache_dir is not None:
            cache_path = Path(cache_dir)
            if (cache_path / 'tokenized_texts.pt').exists():
                self._load_from_cache(cache_path)
            elif tokenizer is not None:
                print(f"  警告: 缓存目录 {cache_dir} 不完整，使用实时分词")
                self._tokenize_texts(tokenizer)
            else:
                raise ValueError(f"缓存目录 {cache_dir} 不完整且未提供tokenizer")
        elif tokenizer is not None:
            self._tokenize_texts(tokenizer)
        else:
            raise ValueError("必须提供tokenizer或有效的cache_dir")

        # 目标变量和数值特征转为Tensor
        self.targets = torch.tensor(self.df['子评论数'].values.astype(np.float32), dtype=torch.float)
        self.numeric_features_tensor = torch.tensor(self.numeric_features, dtype=torch.float)

    def _load_from_cache(self, cache_dir):
        """从预分词缓存加载"""
        print(f"  从缓存加载分词结果: {cache_dir}")

        # 加载缓存文件
        cache_data = torch.load(cache_dir / 'tokenized_texts.pt')
        with open(cache_dir / 'text_mapping.pkl', 'rb') as f:
            mapping = pickle.load(f)

        text_to_idx = mapping['text_to_idx']
        all_ids = cache_data['input_ids']
        all_masks = cache_data['attention_mask']

        # 加载特殊Token ID缓存（如果存在）
        special_ids_file = cache_dir / 'special_token_ids.pkl'
        if special_ids_file.exists():
            with open(special_ids_file, 'rb') as f:
                special_mapping = pickle.load(f)
            text_to_special_ids = special_mapping['text_to_special_ids']
            self._has_special_cache = True
        else:
            text_to_special_ids = None
            self._has_special_cache = False

        # 默认索引（用于未找到的文本）
        default_idx = text_to_idx.get("空", 0)

        def get_indices(col_name):
            """获取文本对应的缓存索引"""
            texts = self.df[col_name].fillna('')
            indices = []
            for t in texts:
                processed = preprocess_text_for_bge(str(t)) if t else "空"
                idx = text_to_idx.get(processed, default_idx)
                indices.append(idx)
            return indices

        n_samples = len(self.df)

        # 始终加载评论文案
        print("    加载评论文案缓存...")
        comment_indices = get_indices('评论文案')
        self.comment_data = (all_ids[comment_indices].clone(), all_masks[comment_indices].clone())

        # 上下文文本（消融实验可选）
        if self.use_context:
            print("    加载微博文案缓存...")
            weibo_indices = get_indices('微博文案')
            self.weibo_data = (all_ids[weibo_indices].clone(), all_masks[weibo_indices].clone())

            print("    加载根评论文案缓存...")
            root_indices = get_indices('根评论文案')
            self.root_data = (all_ids[root_indices].clone(), all_masks[root_indices].clone())

            print("    加载父评论文案缓存...")
            parent_indices = get_indices('父评论文案')
            self.parent_data = (all_ids[parent_indices].clone(), all_masks[parent_indices].clone())
        else:
            # 不使用上下文（消融实验：w/o 上下文）
            print("    跳过上下文文本加载（消融实验: w/o 上下文）")
            self.weibo_data = (torch.zeros(n_samples, self.max_length, dtype=torch.long),
                              torch.zeros(n_samples, self.max_length, dtype=torch.float))
            self.root_data = (torch.zeros(n_samples, self.max_length, dtype=torch.long),
                             torch.zeros(n_samples, self.max_length, dtype=torch.float))
            self.parent_data = (torch.zeros(n_samples, self.max_length, dtype=torch.long),
                               torch.zeros(n_samples, self.max_length, dtype=torch.float))

        # 特殊Token提取（优先使用缓存）
        if self._has_special_cache:
            print("    从缓存加载特殊Token ID...")
            self._load_special_ids_from_cache(text_to_special_ids)
        else:
            print("    提取特殊Token ID (VIP用户/表情/关键词)...")
            self._prepare_special_token_ids()

        print("  缓存加载完成。")

    def _load_special_ids_from_cache(self, text_to_special_ids):
        """从缓存加载特殊Token ID"""
        all_special_ids = []
        all_special_masks = []

        for idx in tqdm(range(len(self.df)), desc="    合并特殊Token", leave=False):
            row = self.df.iloc[idx]
            # 从四个文本字段获取特殊token并合并
            special_ids = set()
            for col in ['评论文案', '微博文案', '根评论文案', '父评论文案']:
                text = str(row.get(col, '')) if row.get(col, '') else ''
                ids = text_to_special_ids.get(text, [])
                special_ids.update(ids)

            # 转为列表并截断/填充
            special_ids = list(special_ids)[:self.max_special_tokens]
            pad_len = self.max_special_tokens - len(special_ids)
            mask = [1.0] * len(special_ids) + [0.0] * pad_len
            special_ids = special_ids + [0] * pad_len  # 用0填充

            all_special_ids.append(special_ids)
            all_special_masks.append(mask)

        self.special_ids_tensor = torch.tensor(all_special_ids, dtype=torch.long)
        self.special_mask_tensor = torch.tensor(all_special_masks, dtype=torch.float)

    def _tokenize_texts(self, tokenizer):
        """实时分词（原有逻辑）"""
        print("  正在预处理文本数据 (Tokenization)...")

        def batch_encode(texts):
            """批量编码文本，返回Tensor"""
            clean_texts = [preprocess_text_for_bge(str(t)) if t else "空" for t in texts]

            all_ids = []
            all_masks = []
            for text in tqdm(clean_texts, desc="    Tokenizing", leave=False):
                if not text or text == "":
                    text = "空"
                encoded = tokenizer.encode(text)
                ids = encoded.ids[:self.max_length]
                pad_len = self.max_length - len(ids)
                mask = [1.0] * len(ids) + [0.0] * pad_len
                ids = ids + [0] * pad_len

                all_ids.append(ids)
                all_masks.append(mask)

            return (torch.tensor(all_ids, dtype=torch.long),
                    torch.tensor(all_masks, dtype=torch.float))

        # 始终处理评论文案
        print("    处理评论文案...")
        self.comment_data = batch_encode(self.df['评论文案'].fillna(''))

        # 上下文文本（消融实验可选）
        n_samples = len(self.df)
        if self.use_context:
            print("    处理微博文案...")
            self.weibo_data = batch_encode(self.df['微博文案'].fillna(''))
            print("    处理根评论文案...")
            self.root_data = batch_encode(self.df['根评论文案'].fillna(''))
            print("    处理父评论文案...")
            self.parent_data = batch_encode(self.df['父评论文案'].fillna(''))
        else:
            # 不使用上下文（消融实验：w/o 上下文）
            print("    跳过上下文文本预处理（消融实验: w/o 上下文）")
            self.weibo_data = (torch.zeros(n_samples, self.max_length, dtype=torch.long),
                              torch.zeros(n_samples, self.max_length, dtype=torch.float))
            self.root_data = (torch.zeros(n_samples, self.max_length, dtype=torch.long),
                             torch.zeros(n_samples, self.max_length, dtype=torch.float))
            self.parent_data = (torch.zeros(n_samples, self.max_length, dtype=torch.long),
                               torch.zeros(n_samples, self.max_length, dtype=torch.float))

        # 提取特殊Token ID
        print("    提取特殊Token ID (VIP用户/表情/关键词)...")
        self._prepare_special_token_ids()

        print("  文本预处理完成，已缓存至内存。")

    def _prepare_special_token_ids(self):
        """从所有文本中提取特殊Token ID并创建Tensor"""
        all_special_ids = []
        all_special_masks = []

        for idx in tqdm(range(len(self.df)), desc="    提取特殊Token", leave=False):
            row = self.df.iloc[idx]
            # 从四个文本字段提取特殊token
            special_ids = set()
            for col in ['评论文案', '微博文案', '根评论文案', '父评论文案']:
                text = row.get(col, '')
                ids = extract_special_token_ids(text)
                special_ids.update(ids)

            # 转为列表并截断/填充
            special_ids = list(special_ids)[:self.max_special_tokens]
            pad_len = self.max_special_tokens - len(special_ids)
            mask = [1.0] * len(special_ids) + [0.0] * pad_len
            special_ids = special_ids + [0] * pad_len  # 用0填充

            all_special_ids.append(special_ids)
            all_special_masks.append(mask)

        self.special_ids_tensor = torch.tensor(all_special_ids, dtype=torch.long)
        self.special_mask_tensor = torch.tensor(all_special_masks, dtype=torch.float)

    def _prepare_numeric_features(self):
        """准备数值特征"""
        # 用户总评论数（log变换，确保非负）
        user_comments_raw = self.df['用户总评论数'].fillna(0).values
        user_comments_raw = np.clip(user_comments_raw, 0, None)  # 确保非负
        user_comments = np.log1p(user_comments_raw)

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

        # 基础特征列表
        feature_list = [
            user_comments,
            is_verified,
            is_first_level,
            hour_sin,
            hour_cos,
            weekday_sin,
            weekday_cos,
        ]

        # 时间密度特征（消融实验可选）
        if self.use_density_features:
            # 时间顺序索引（标准化）
            time_idx = self.df.get('时间顺序索引', pd.Series([0] * len(self.df))).values
            time_idx = (time_idx - time_idx.mean()) / (time_idx.std() + 1e-8)

            # 最大相似度
            max_sim = self.df.get('最大相似度', pd.Series([0] * len(self.df))).values

            # 重复次数（log变换，确保非负）
            repeat_raw = self.df.get('重复次数', pd.Series([0] * len(self.df))).values
            repeat_raw = np.clip(repeat_raw, 0, None)  # 确保非负
            repeat_count = np.log1p(repeat_raw)

            feature_list.extend([time_idx, max_sim, repeat_count])

        # 合并所有数值特征
        self.numeric_features = np.column_stack(feature_list).astype(np.float32)

        # 检查并替换NaN和Inf
        self.numeric_features = np.nan_to_num(self.numeric_features, nan=0.0, posinf=0.0, neginf=0.0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 极快的索引操作
        return {
            'comment_ids': self.comment_data[0][idx],
            'comment_mask': self.comment_data[1][idx],
            'weibo_ids': self.weibo_data[0][idx],
            'weibo_mask': self.weibo_data[1][idx],
            'root_ids': self.root_data[0][idx],
            'root_mask': self.root_data[1][idx],
            'parent_ids': self.parent_data[0][idx],
            'parent_mask': self.parent_data[1][idx],
            'numeric_features': self.numeric_features_tensor[idx],
            'special_ids': self.special_ids_tensor[idx],
            'special_mask': self.special_mask_tensor[idx],
            'target': self.targets[idx],
        }
