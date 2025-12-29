"""
点赞数同步脚本
从小米SU7数据同步点赞数到小米SU7数据_2

功能：
1. 热门微博.csv：weibo_url匹配，更新like_count
2. 热门微博评论数据.csv：评论ID匹配，更新点赞数
3. 热门微博转发数据.csv：原创类型与热门微博.csv统一

使用方法：
python src/update_like_counts.py
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path


# ==================== 配置 ====================
SOURCE_ROOT = Path('D:/010_CodePrograms/L/LLM_su7/小米SU7数据')
TARGET_ROOT = Path('D:/010_CodePrograms/L/LLM_su7/小米SU7数据_2')


class LikeCountUpdater:
    def __init__(self, source_root: Path, target_root: Path):
        self.source_root = source_root
        self.target_root = target_root
        self.date_folders = self._get_date_folders()

        # 统计信息
        self.stats = {
            'weibo': {'matched': 0, 'total': 0},
            'comment': {'matched': 0, 'total': 0},
            'repost': {'matched': 0, 'total': 0}
        }

    def _get_date_folders(self) -> list:
        """获取所有日期文件夹"""
        folders = sorted([
            d.name for d in self.target_root.iterdir()
            if d.is_dir() and d.name.startswith('2025-')
        ])
        return folders

    def _read_csv_safe(self, filepath: Path, expected_columns: list = None) -> pd.DataFrame:
        """安全读取CSV文件"""
        if not filepath.exists():
            return None

        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')

            # 检查是否缺少列名（第一行是URL而不是列名）
            if expected_columns and len(df.columns) > 0:
                first_col = str(df.columns[0])
                if first_col.startswith('http'):
                    df = pd.read_csv(filepath, encoding='utf-8-sig',
                                    header=None, names=expected_columns)
            return df
        except Exception as e:
            print(f"  [警告] 读取文件失败 {filepath.name}: {e}")
            return None

    def _normalize_comment_id(self, id_series: pd.Series) -> pd.Series:
        """标准化评论ID，移除.0后缀"""
        return id_series.astype(str).str.replace(r'\.0$', '', regex=True)

    def update_weibo_likes(self, date: str) -> tuple:
        """更新热门微博的点赞数"""
        source_file = self.source_root / date / '热门微博.csv'
        target_file = self.target_root / date / '热门微博.csv'

        if not source_file.exists() or not target_file.exists():
            return 0, 0

        source_df = pd.read_csv(source_file, encoding='utf-8-sig')
        target_df = pd.read_csv(target_file, encoding='utf-8-sig')

        # 构建映射：weibo_url -> like_count
        like_map = dict(zip(source_df['weibo_url'], source_df['like_count']))

        # 更新（未匹配自动为NaN）
        target_df['like_count'] = target_df['weibo_url'].map(like_map)

        target_df.to_csv(target_file, index=False, encoding='utf-8-sig')

        matched = target_df['like_count'].notna().sum()
        total = len(target_df)
        return matched, total

    def update_comment_likes(self, date: str) -> tuple:
        """更新评论的点赞数"""
        source_file = self.source_root / date / '热门微博评论数据.csv'
        target_file = self.target_root / date / '热门微博评论数据.csv'

        if not source_file.exists() or not target_file.exists():
            return 0, 0

        source_df = pd.read_csv(source_file, encoding='utf-8-sig')
        target_df = pd.read_csv(target_file, encoding='utf-8-sig')

        # 标准化评论ID并构建映射
        source_ids = self._normalize_comment_id(source_df['评论ID'])
        like_map = dict(zip(source_ids, source_df['点赞数']))

        # 更新目标数据
        target_ids = self._normalize_comment_id(target_df['评论ID'])
        target_df['点赞数'] = target_ids.map(like_map)

        target_df.to_csv(target_file, index=False, encoding='utf-8-sig')

        matched = target_df['点赞数'].notna().sum()
        total = len(target_df)
        return matched, total

    def update_repost_likes(self, date: str) -> tuple:
        """更新转发数据中原创记录的点赞数"""
        source_weibo = self.source_root / date / '热门微博.csv'
        target_repost = self.target_root / date / '热门微博转发数据.csv'

        if not source_weibo.exists() or not target_repost.exists():
            return 0, 0

        # 读取热门微博（获取正确点赞数）
        weibo_df = pd.read_csv(source_weibo, encoding='utf-8-sig')
        like_map = dict(zip(weibo_df['weibo_url'], weibo_df['like_count']))

        # 读取转发数据（处理可能缺失列名的情况）
        expected_cols = [
            '原文链接', '原创/转发', '标题', '原文作者', '全文内容',
            '根标题', '根微博作者', '原微博内容', '发布时间',
            '转发数', '评论数', '点赞数', '用户认证',
            '用户总评论数', '用户总转发数', '用户总点赞数'
        ]
        repost_df = self._read_csv_safe(target_repost, expected_cols)
        if repost_df is None:
            return 0, 0

        # 只更新"原创"类型
        if '原创/转发' not in repost_df.columns:
            return 0, 0

        original_mask = repost_df['原创/转发'] == '原创'

        # 只更新能匹配到的原创记录，未匹配的保持原值
        matched_count = 0
        for idx in repost_df[original_mask].index:
            url = repost_df.loc[idx, '原文链接']
            if url in like_map:
                repost_df.loc[idx, '点赞数'] = like_map[url]
                matched_count += 1

        repost_df.to_csv(target_repost, index=False, encoding='utf-8-sig')

        total = original_mask.sum()
        return matched_count, total

    def run(self):
        """执行更新"""
        print("=" * 60)
        print("点赞数同步脚本")
        print(f"源数据: {self.source_root}")
        print(f"目标数据: {self.target_root}")
        print(f"日期数量: {len(self.date_folders)}")
        print("=" * 60)

        for date in tqdm(self.date_folders, desc="处理日期"):
            # 更新热门微博
            m, t = self.update_weibo_likes(date)
            self.stats['weibo']['matched'] += m
            self.stats['weibo']['total'] += t

            # 更新评论
            m, t = self.update_comment_likes(date)
            self.stats['comment']['matched'] += m
            self.stats['comment']['total'] += t

            # 更新转发（原创）
            m, t = self.update_repost_likes(date)
            self.stats['repost']['matched'] += m
            self.stats['repost']['total'] += t

        # 输出统计
        print("\n" + "=" * 60)
        print("【更新统计】")
        print("=" * 60)

        for key, name in [('weibo', '热门微博'), ('comment', '评论数据')]:
            data = self.stats[key]
            unmatched = data['total'] - data['matched']
            pct = data['matched'] / data['total'] * 100 if data['total'] > 0 else 0
            print(f"{name}: 匹配 {data['matched']}/{data['total']} ({pct:.1f}%), 未匹配设为NaN: {unmatched}")

        # 转发数据单独处理（未匹配保持原值）
        data = self.stats['repost']
        unmatched = data['total'] - data['matched']
        pct = data['matched'] / data['total'] * 100 if data['total'] > 0 else 0
        print(f"转发数据(原创): 匹配更新 {data['matched']}/{data['total']} ({pct:.1f}%), 未匹配保持原值: {unmatched}")

        print("\n完成!")


def main():
    updater = LikeCountUpdater(SOURCE_ROOT, TARGET_ROOT)
    updater.run()


if __name__ == '__main__':
    main()
