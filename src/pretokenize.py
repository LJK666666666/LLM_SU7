"""
预分词脚本：生成分词缓存以加速训练

将所有唯一文本（评论/微博/根评论/父评论）进行一次性分词，
保存到缓存目录，训练时直接加载避免重复分词。

使用方法：
    python src/pretokenize.py                    # 生成缓存
    python src/pretokenize.py --force            # 强制重新生成
    python src/pretokenize.py --output_dir cache # 指定输出目录
"""

import argparse
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from tokenizers import Tokenizer
from tqdm import tqdm

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import BGE_MODEL_PATH, ROOT_DIR
from src.models.bge_nn import preprocess_text_for_bge, extract_special_token_ids


def parse_args():
    parser = argparse.ArgumentParser(description='预分词缓存生成工具')
    parser.add_argument('--force', action='store_true',
                        help='强制重新生成缓存（即使已存在）')
    parser.add_argument('--max_length', type=int, default=128,
                        help='最大序列长度 (default: 128)')
    parser.add_argument('--output_dir', type=str, default='cache',
                        help='缓存输出目录 (default: cache)')
    return parser.parse_args()


def collect_unique_texts(data_files):
    """收集所有数据文件中的唯一文本

    返回:
        preprocessed_texts: 预处理后的唯一文本集合（用于分词）
        raw_texts: 原始唯一文本集合（用于特殊Token提取）
    """
    text_columns = ['评论文案', '微博文案', '根评论文案', '父评论文案']
    preprocessed_texts = set()
    raw_texts = set()

    for file_path in data_files:
        if not file_path.exists():
            print(f"  警告: 文件不存在 {file_path}")
            continue

        print(f"  加载 {file_path.name}...")
        df = pd.read_pickle(file_path)

        for col in text_columns:
            if col in df.columns:
                # 收集原始文本
                raw = df[col].fillna('')
                raw_texts.update(raw.unique())

                # 收集预处理后的文本（用于分词）
                processed = raw.apply(
                    lambda t: preprocess_text_for_bge(str(t)) if t else "空"
                )
                preprocessed_texts.update(processed.unique())

    # 确保空文本被处理
    preprocessed_texts.add("空")
    raw_texts.add("")

    return preprocessed_texts, raw_texts


def tokenize_texts(texts, tokenizer, max_length):
    """批量分词"""
    all_ids = []
    all_masks = []

    for text in tqdm(texts, desc="分词中"):
        if not text or text == "":
            text = "空"

        encoded = tokenizer.encode(text)
        ids = encoded.ids[:max_length]
        pad_len = max_length - len(ids)
        mask = [1.0] * len(ids) + [0.0] * pad_len
        ids = ids + [0] * pad_len

        all_ids.append(ids)
        all_masks.append(mask)

    return (
        torch.tensor(all_ids, dtype=torch.long),
        torch.tensor(all_masks, dtype=torch.float)
    )


def extract_special_ids_for_texts(raw_texts):
    """为每个原始文本提取特殊Token ID"""
    text_to_special_ids = {}

    for text in tqdm(raw_texts, desc="提取特殊Token ID"):
        if not text:
            text_to_special_ids[""] = []
            continue
        special_ids = extract_special_token_ids(text)
        text_to_special_ids[text] = special_ids

    return text_to_special_ids


def main():
    args = parse_args()

    # 输出目录
    output_dir = ROOT_DIR / args.output_dir
    cache_file = output_dir / 'tokenized_texts.pt'
    mapping_file = output_dir / 'text_mapping.pkl'
    special_ids_file = output_dir / 'special_token_ids.pkl'
    meta_file = output_dir / 'cache_meta.json'

    # 检查是否已存在
    if cache_file.exists() and not args.force:
        print(f"缓存已存在: {output_dir}")
        print("使用 --force 参数强制重新生成")
        return

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("预分词缓存生成")
    print("=" * 60)

    # 1. 收集所有唯一文本
    print("\n【步骤1】收集唯一文本...")
    data_files = [
        ROOT_DIR / 'train.pkl',
        ROOT_DIR / 'val.pkl',
        ROOT_DIR / 'test.pkl',
    ]
    preprocessed_texts, raw_texts = collect_unique_texts(data_files)

    # 排序保证确定性
    preprocessed_texts = sorted(list(preprocessed_texts))
    raw_texts = sorted(list(raw_texts))
    print(f"  预处理后唯一文本数量: {len(preprocessed_texts):,}")
    print(f"  原始唯一文本数量: {len(raw_texts):,}")

    # 2. 创建文本到索引映射（预处理后的文本）
    print("\n【步骤2】创建索引映射...")
    text_to_idx = {text: i for i, text in enumerate(preprocessed_texts)}
    print(f"  映射表大小: {len(text_to_idx):,}")

    # 3. 加载tokenizer
    print("\n【步骤3】加载BGE tokenizer...")
    tokenizer_path = os.path.join(str(BGE_MODEL_PATH), 'tokenizer.json')
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"  Tokenizer加载完成: {tokenizer_path}")

    # 4. 批量分词（预处理后的文本）
    print(f"\n【步骤4】批量分词 (max_length={args.max_length})...")
    input_ids, attention_mask = tokenize_texts(preprocessed_texts, tokenizer, args.max_length)
    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  attention_mask shape: {attention_mask.shape}")

    # 5. 提取特殊Token ID（原始文本）
    print("\n【步骤5】提取特殊Token ID...")
    text_to_special_ids = extract_special_ids_for_texts(raw_texts)
    print(f"  特殊Token映射表大小: {len(text_to_special_ids):,}")

    # 6. 保存缓存
    print("\n【步骤6】保存缓存文件...")

    # 保存分词结果
    torch.save({
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'max_length': args.max_length,
    }, cache_file)
    print(f"  保存: {cache_file}")
    print(f"  文件大小: {cache_file.stat().st_size / 1024 / 1024:.2f} MB")

    # 保存映射
    with open(mapping_file, 'wb') as f:
        pickle.dump({'text_to_idx': text_to_idx}, f)
    print(f"  保存: {mapping_file}")

    # 保存特殊Token ID映射
    with open(special_ids_file, 'wb') as f:
        pickle.dump({'text_to_special_ids': text_to_special_ids}, f)
    print(f"  保存: {special_ids_file}")
    print(f"  文件大小: {special_ids_file.stat().st_size / 1024 / 1024:.2f} MB")

    # 保存元信息
    meta = {
        'version': '2.0',  # 版本升级，包含特殊Token缓存
        'created_at': datetime.now().isoformat(),
        'n_preprocessed_texts': len(preprocessed_texts),
        'n_raw_texts': len(raw_texts),
        'max_length': args.max_length,
        'data_files': [str(f) for f in data_files if f.exists()],
    }
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  保存: {meta_file}")

    print("\n" + "=" * 60)
    print("预分词缓存生成完成!")
    print(f"缓存目录: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
