# 时间顺序与相似度特征提取
# 支持两种模式：
#   --method bge     使用BGE模型计算语义相似度（默认）
#   --method minhash 使用MinHash+N-gram计算Jaccard相似度

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='计算时间密度特征')
    parser.add_argument('--method', type=str, default='bge', choices=['bge', 'minhash'],
                        help='相似度计算方法: bge(语义相似度) 或 minhash(Jaccard相似度)')
    parser.add_argument('--ngram', type=int, default=3, help='N-gram的N值（仅minhash模式）')
    parser.add_argument('--num_perm', type=int, default=128, help='MinHash排列数量（仅minhash模式）')
    parser.add_argument('--threshold', type=float, default=0.4, help='相似度阈值（minhash默认0.5，bge默认0.9）')
    parser.add_argument('--window', type=int, default=10000, help='滑动窗口大小')
    parser.add_argument('--topk_threshold', type=int, default=None,
                        help='全局TopK高频阈值（BGE默认5，MinHash默认3），超过此次数的文本会被记住')
    return parser.parse_args()

# ==================== MinHash实现 ====================
class MinHash:
    """MinHash算法实现，用于快速估计Jaccard相似度"""
    def __init__(self, num_perm=128, seed=42):
        self.num_perm = num_perm
        np.random.seed(seed)
        # 生成哈希函数参数 (a*x + b) mod p
        self.max_hash = (1 << 32) - 1
        self.prime = 4294967311  # 大于2^32的质数
        self.a = np.random.randint(1, self.prime, size=num_perm, dtype=np.uint64)
        self.b = np.random.randint(0, self.prime, size=num_perm, dtype=np.uint64)

    def get_signature(self, tokens):
        """计算一个文本的MinHash签名"""
        if not tokens:
            return np.full(self.num_perm, self.max_hash, dtype=np.uint32)

        signature = np.full(self.num_perm, self.max_hash, dtype=np.uint64)
        for token in tokens:
            # 使用Python内置hash，转为正整数
            h = hash(token) & self.max_hash
            # 计算所有哈希函数的值
            hash_values = (self.a * h + self.b) % self.prime
            signature = np.minimum(signature, hash_values)
        return signature.astype(np.uint32)

    def jaccard_similarity(self, sig1, sig2):
        """通过签名估计Jaccard相似度"""
        return np.mean(sig1 == sig2)

def get_ngrams(text, n=3):
    """将文本转换为N-gram集合"""
    if not isinstance(text, str) or len(text) == 0:
        return set()
    # 移除空白字符
    text = ''.join(text.split())
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def compute_minhash_features(all_df, args):
    """使用MinHash计算相似度特征"""
    print(f"\n使用MinHash算法 (N-gram={args.ngram}, num_perm={args.num_perm})")

    minhash = MinHash(num_perm=args.num_perm)
    texts = all_df['评论文案'].tolist()

    # 计算所有文本的MinHash签名
    print("计算MinHash签名...")
    signatures = []
    for text in tqdm(texts, desc="生成签名"):
        ngrams = get_ngrams(text, args.ngram)
        sig = minhash.get_signature(ngrams)
        signatures.append(sig)
    signatures = np.array(signatures)

    # 计算相似度特征
    print("\n计算相似度特征...")
    threshold = args.threshold if args.threshold != 0.9 else 0.5  # minhash默认阈值0.5
    topk_threshold = args.topk_threshold if args.topk_threshold is not None else 3  # minhash默认3
    max_similar = np.zeros(len(all_df))
    repeat_count = np.zeros(len(all_df), dtype=int)
    window_size = args.window

    # 全局TopK Hash：记录高频文本的签名和出现次数
    # key: 签名的hash值, value: (签名, 累计匹配次数, 首次出现索引)
    global_topk = {}
    print(f"启用全局TopK记忆 (阈值={topk_threshold})")

    for i in tqdm(range(1, len(all_df)), desc="计算相似度"):
        current_sig = signatures[i]
        start_idx = max(0, i - window_size)

        # 1. 在滑动窗口内计算相似度
        prev_sigs = signatures[start_idx:i]
        window_max_sim = 0.0
        window_repeat = 0
        if len(prev_sigs) > 0:
            similarities = np.mean(prev_sigs == current_sig, axis=1)
            window_max_sim = similarities.max()
            window_repeat = (similarities >= threshold).sum()

        # 2. 在全局TopK中查找（窗口外的高频"老梗"）
        global_max_sim = 0.0
        global_repeat = 0
        for sig_hash, (stored_sig, count, first_idx) in global_topk.items():
            # 只检查窗口外的记录
            if first_idx < start_idx:
                sim = np.mean(stored_sig == current_sig)
                if sim > global_max_sim:
                    global_max_sim = sim
                if sim >= threshold:
                    global_repeat += 1

        # 3. 合并结果
        max_similar[i] = max(window_max_sim, global_max_sim)
        repeat_count[i] = window_repeat + global_repeat

        # 4. 更新全局TopK（当窗口内匹配次数达到阈值时）
        if window_repeat >= topk_threshold:
            sig_hash = hash(current_sig.tobytes())
            if sig_hash not in global_topk:
                global_topk[sig_hash] = (current_sig.copy(), window_repeat, i)
            else:
                # 更新累计次数
                old_sig, old_count, old_idx = global_topk[sig_hash]
                global_topk[sig_hash] = (old_sig, old_count + window_repeat, old_idx)

    print(f"全局TopK记录数: {len(global_topk)}")
    return max_similar, repeat_count

# ==================== BGE模型实现 ====================
def compute_bge_features(all_df, args):
    """使用BGE模型计算语义相似度特征"""
    import torch
    import torch.nn as nn
    import json

    # 完全禁用tensorflow相关导入
    sys.modules['tensorflow'] = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载BGE模型
    model_path = r'C:\Users\LJK\.cache\huggingface\hub\models--BAAI--bge-base-zh-v1.5\snapshots\f03589ceff5aac7111bd60cfc7d497ca17ecac65'
    print(f"使用模型路径: {model_path}")

    print("加载tokenizer...")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(model_path, 'tokenizer.json'))

    with open(os.path.join(model_path, 'vocab.txt'), 'r', encoding='utf-8') as f:
        vocab = {line.strip(): idx for idx, line in enumerate(f)}

    pad_token_id = vocab.get('[PAD]', 0)

    print("加载BERT模型...")
    with open(os.path.join(model_path, 'config.json'), 'r') as f:
        config = json.load(f)

    # BERT模型定义（简化版）
    class BertEmbeddings(nn.Module):
        def __init__(self, config):
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

            q, k, v = transpose_for_scores(self.query(hidden_states)), transpose_for_scores(self.key(hidden_states)), transpose_for_scores(self.value(hidden_states))
            scores = torch.matmul(q, k.transpose(-1, -2)) / (self.attention_head_size ** 0.5)
            if attention_mask is not None:
                scores = scores + attention_mask
            context = torch.matmul(self.dropout(nn.functional.softmax(scores, dim=-1)), v)
            return context.permute(0, 2, 1, 3).contiguous().view(context.size(0), -1, self.all_head_size)

    class BertLayer(nn.Module):
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
            intermediate = nn.functional.gelu(self.intermediate(hidden_states))
            return self.output_norm(hidden_states + self.dropout(self.output_dense(intermediate)))

    class BertModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.embeddings = BertEmbeddings(config)
            self.layers = nn.ModuleList([BertLayer(config) for _ in range(config['num_hidden_layers'])])

        def forward(self, input_ids, attention_mask=None):
            if attention_mask is not None:
                attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
            hidden_states = self.embeddings(input_ids)
            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask)
            return hidden_states

    model = BertModel(config)
    state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location='cpu')

    # 映射权重
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

    model_state = model.state_dict()
    matched = {k: v for k, v in new_state_dict.items() if k in model_state and model_state[k].shape == v.shape}
    model_state.update(matched)
    model.load_state_dict(model_state)
    model = model.to(device).eval()
    print(f"模型加载完成，匹配权重: {len(matched)}/{len(model_state)}")

    # 计算embedding
    @torch.no_grad()
    def get_embeddings(texts, batch_size=64):
        embeddings = []
        tokenizer.enable_padding(pad_id=pad_token_id, pad_token='[PAD]')
        tokenizer.enable_truncation(max_length=128)

        for i in tqdm(range(0, len(texts), batch_size), desc="计算embedding"):
            batch_texts = [t if isinstance(t, str) and len(t) > 0 else "空" for t in texts[i:i+batch_size]]
            encoded = tokenizer.encode_batch(batch_texts)
            max_len = max(len(enc.ids) for enc in encoded)
            input_ids = torch.zeros((len(batch_texts), max_len), dtype=torch.long, device=device)
            attention_mask = torch.zeros((len(batch_texts), max_len), dtype=torch.float, device=device)
            for j, enc in enumerate(encoded):
                input_ids[j, :len(enc.ids)] = torch.tensor(enc.ids)
                attention_mask[j, :len(enc.ids)] = 1.0
            output = model(input_ids, attention_mask)
            embeddings.append(output[:, 0, :].cpu().numpy())
        return np.vstack(embeddings)

    print("\n计算所有评论的embedding...")
    texts = all_df['评论文案'].tolist()
    embeddings = get_embeddings(texts, batch_size=64)
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    print("\n计算相似度特征...")
    threshold = args.threshold
    topk_threshold = args.topk_threshold if args.topk_threshold is not None else 5  # BGE默认5
    max_similar = np.zeros(len(all_df))
    repeat_count = np.zeros(len(all_df), dtype=int)
    window_size = args.window

    # 全局TopK Hash：记录高频文本的embedding和出现次数
    # key: 文本hash值, value: (embedding, 累计匹配次数, 首次出现索引)
    global_topk = {}
    print(f"启用全局TopK记忆 (阈值={topk_threshold})")

    for i in tqdm(range(1, len(all_df)), desc="计算相似度"):
        start_idx = max(0, i - window_size)
        current_emb = embeddings_norm[i]

        # 1. 在滑动窗口内计算相似度
        prev_embs = embeddings_norm[start_idx:i]
        window_max_sim = 0.0
        window_repeat = 0
        if len(prev_embs) > 0:
            similarities = np.dot(prev_embs, current_emb)
            window_max_sim = similarities.max()
            window_repeat = (similarities >= threshold).sum()

        # 2. 在全局TopK中查找（窗口外的高频"老梗"）
        global_max_sim = 0.0
        global_repeat = 0
        for emb_hash, (stored_emb, count, first_idx) in global_topk.items():
            # 只检查窗口外的记录
            if first_idx < start_idx:
                sim = np.dot(stored_emb, current_emb)
                if sim > global_max_sim:
                    global_max_sim = sim
                if sim >= threshold:
                    global_repeat += 1

        # 3. 合并结果
        max_similar[i] = max(window_max_sim, global_max_sim)
        repeat_count[i] = window_repeat + global_repeat

        # 4. 更新全局TopK（当窗口内匹配次数达到阈值时）
        if window_repeat >= topk_threshold:
            # 使用文本内容的hash作为key
            text_hash = hash(texts[i]) if isinstance(texts[i], str) else i
            if text_hash not in global_topk:
                global_topk[text_hash] = (current_emb.copy(), window_repeat, i)
            else:
                # 更新累计次数
                old_emb, old_count, old_idx = global_topk[text_hash]
                global_topk[text_hash] = (old_emb, old_count + window_repeat, old_idx)

    print(f"全局TopK记录数: {len(global_topk)}")
    return max_similar, repeat_count

def main():
    args = parse_args()

    # 设置默认TopK阈值
    if args.topk_threshold is None:
        args.topk_threshold = 5 if args.method == 'bge' else 3

    print(f"相似度计算方法: {args.method}")
    print(f"相似度阈值: {args.threshold}")
    print(f"滑动窗口大小: {args.window}")
    print(f"全局TopK阈值: {args.topk_threshold}")

    # 加载数据
    print("\n加载数据...")
    train_df = pd.read_pickle('train.pkl')
    val_df = pd.read_pickle('val.pkl')
    test_df = pd.read_pickle('test.pkl')

    train_df['数据集'] = 'train'
    val_df['数据集'] = 'val'
    test_df['数据集'] = 'test'
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"总数据量: {len(all_df):,} 条")

    # 按时间排序
    print("\n按时间排序...")
    all_df['发布时间'] = pd.to_datetime(all_df['发布时间'])
    all_df = all_df.sort_values('发布时间').reset_index(drop=True)
    all_df['时间顺序索引'] = range(len(all_df))
    print(f"时间范围: {all_df['发布时间'].min()} ~ {all_df['发布时间'].max()}")

    # 根据方法计算特征
    if args.method == 'minhash':
        max_similar, repeat_count = compute_minhash_features(all_df, args)
    else:
        max_similar, repeat_count = compute_bge_features(all_df, args)

    all_df['最大相似度'] = max_similar
    all_df['重复次数'] = repeat_count

    print(f"\n相似度统计:")
    print(f"  平均最大相似度: {max_similar.mean():.4f}")
    print(f"  平均重复次数: {repeat_count.mean():.2f}")
    print(f"  有重复的评论数: {(repeat_count > 0).sum():,}")

    # 保存数据
    print("\n保存数据...")
    suffix = f'_{args.method}' if args.method != 'bge' else ''
    for dataset_name in ['train', 'val', 'test']:
        subset = all_df[all_df['数据集'] == dataset_name][['序号', '时间顺序索引', '最大相似度', '重复次数']]
        filename = f'{dataset_name}_time_density{suffix}.pkl'
        subset.to_pickle(filename)
        print(f"  {filename}: {len(subset):,} 条")

    print("\n完成!")

if __name__ == '__main__':
    main()
