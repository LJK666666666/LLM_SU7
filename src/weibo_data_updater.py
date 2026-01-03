"""
微博数据更新器 - 简化版
功能：
1. 重新爬取所有微博，删除已删除/私密的记录
2. 使用分页API获取所有评论的真实点赞数
3. 同步转发数据中的"原创"记录与微博数据
4. 支持命令行参数设置时间范围

使用方法：
python src/weibo_data_updater.py --start 2025-03-27 --end 2025-04-14
python src/weibo_data_updater.py --date 2025-03-27
python src/weibo_data_updater.py --all
"""

import requests
from curl_cffi import requests as curl_requests
import pandas as pd
import time
import json
import re
import os
import argparse
from glob import glob
from tqdm import tqdm
from datetime import datetime
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ============================================
# 配置区 - 请在这里设置你的Cookie
# ============================================
MAX_WORKERS = 1  # 并行线程数
COOKIE = """
SCF=Au5MMLg4qKaUXllY_iYmixl59g2UMqtKoyamq9pMwGJlrLrO_TxzLl96PUk2VCW6BuiCYNR1Zzzoql3XjtUctBI.; SUB=_2A25ETvwtDeRhGeFH61AT-SvFzT2IHXVnInHlrDV6PUJbktAYLWTakW1NeC_PDhI_9DWZMJHmqT7EfXM2h5e4UKsE; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WFcJoFcRC-5hYfmevTMjv2F5NHD95QN1K5Eeo.f1KqpWs4DqcjMi--NiK.Xi-2Ri--ciKnRi-zNS0.7eoz4SK.ceBtt; ALF=1769085309; _T_WM=20654691948; WEIBOCN_FROM=1110006030; MLOGIN=1; XSRF-TOKEN=9ec205; mweibo_short_token=04a3f91630; M_WEIBOCN_PARAMS=oid%3D5151777810024981%26luicode%3D20000061%26lfid%3D5151777810024981%26uicode%3D20000174
"""

# 数据根目录
DATA_ROOT = 'D:/010_CodePrograms/L/LLM_su7/小米SU7数据'

# 请求配置
REQUEST_DELAY = (0.1, 0.15)  # 请求间隔（秒）
MAX_RETRIES = 3  # 最大重试次数
COMMENT_PAGE_SIZE = 5  # 每页评论数

# 隧道代理配置（更自动切换IP）
TUNNEL_PROXY = {
    'enabled': True,  
    'tunnel': 'd749.kdltps.com:15818',
    'username': 't16682212001679',
    'password': 'mui8c3je',
}

# 随机User-Agent
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
]


class WeiboDataUpdater:
    def __init__(self, cookie=None, max_workers=MAX_WORKERS):
        self.max_workers = max_workers
        self.lock = threading.Lock()
        self.cookie = (cookie or COOKIE).strip()
        self.request_count = 0

        # 隧道代理（每次请求自动切换IP）
        if TUNNEL_PROXY['enabled']:
            tunnel = TUNNEL_PROXY['tunnel']
            username = TUNNEL_PROXY['username']
            password = TUNNEL_PROXY['password']
            self.proxies = {
                'http': f'http://{username}:{password}@{tunnel}/',
                'https': f'http://{username}:{password}@{tunnel}/'
            }
            print(f"[代理] 使用隧道代理: {tunnel}")
        else:
            self.proxies = None
            print("[代理] 未启用代理")

    def _get_headers(self):
        """生成请求头"""
        xsrf_token = ''
        if 'XSRF-TOKEN=' in self.cookie:
            match = re.search(r'XSRF-TOKEN=([^;]+)', self.cookie)
            if match:
                xsrf_token = match.group(1)

        return {
            'User-Agent': random.choice(USER_AGENTS),
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Referer': 'https://m.weibo.cn/',
            'X-Requested-With': 'XMLHttpRequest',
            'X-XSRF-TOKEN': xsrf_token,
            'Origin': 'https://m.weibo.cn',
            'Connection': 'close',  # 隧道代理不复用连接
            'Cookie': self.cookie
        }

    def extract_weibo_id(self, url):
        """从URL提取微博ID"""
        match = re.search(r'weibo\.com/\d+/(\w+)', str(url))
        return match.group(1) if match else None

    def _get_weibo_from_detail_page(self, weibo_id):
        """从 /detail/{id} 页面获取微博信息"""
        detail_url = f'https://m.weibo.cn/detail/{weibo_id}'
        headers = self._get_headers()
        headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'

        try:
            response = curl_requests.get(
                detail_url,
                headers=headers,
                proxies=self.proxies,
                impersonate='chrome120',
                timeout=15
            )

            if response.status_code == 200:
                html = response.text

                # 提取数据（优先检查是否有有效数据）
                reposts_match = re.search(r'"reposts_count"\s*:\s*(\d+)', html)
                comments_match = re.search(r'"comments_count"\s*:\s*(\d+)', html)
                attitudes_match = re.search(r'"attitudes_count"\s*:\s*(\d+)', html)

                if reposts_match:
                    return {
                        'weibo_id': weibo_id,
                        'reposts_count': int(reposts_match.group(1)),
                        'comments_count': int(comments_match.group(1)) if comments_match else 0,
                        'attitudes_count': int(attitudes_match.group(1)) if attitudes_match else 0,
                    }, 'success'

                # 没有数据时，判断页面类型
                # 真正的验证码页面：包含 captchaId 和 initGeetest
                if 'captchaId' in html and 'initGeetest' in html:
                    return None, 'captcha'

                # 微博不存在/被删除：返回错误页面、首页或提示页面
                # 常见情况："微博-出错了"、"页面不存在"、首页（内容很短且无数据）
                if '出错了' in html or '页面不存在' in html or '抱歉，你访问的页面' in html or len(html) < 5000:
                    return None, 'deleted_or_private'

                # 其他情况（可能是页面结构变化）
                return None, 'parse_failed'
            else:
                return None, f'http_{response.status_code}'
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['460', 'connect', 'tunnel', 'timeout', 'timed out']):
                return None, 'proxy_error'
            return None, f'error_{str(e)}'

    def get_weibo_info(self, weibo_url):
        """获取单条微博的详细信息"""
        weibo_id = self.extract_weibo_id(weibo_url)
        if not weibo_id:
            return None, 'invalid_url'

        max_retries = 3
        for attempt in range(max_retries):
            data, status = self._get_weibo_from_detail_page(weibo_id)

            if status == 'success' and data:
                return {
                    'weibo_id': weibo_id,
                    'weibo_url': weibo_url,
                    'reposts_count': data.get('reposts_count', 0),
                    'comments_count': data.get('comments_count', 0),
                    'attitudes_count': data.get('attitudes_count', 0),
                }, 'success'
            elif status == 'deleted_or_private':
                return None, status
            elif status in ['captcha', 'proxy_error']:
                # 隧道代理自动切换IP，重试即可
                time.sleep(0.5)
                continue
            else:
                # 其他错误，重试
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                return None, status

        return None, 'max_retries_exceeded'

    def get_all_comments(self, weibo_url, max_pages=50):
        """获取微博的所有评论（分页遍历）"""
        weibo_id = self.extract_weibo_id(weibo_url)
        if not weibo_id:
            return []

        all_comments = []
        max_id = 0
        max_id_type = 0

        for page in range(max_pages):
            if page == 0:
                api_url = f'https://m.weibo.cn/comments/hotflow?id={weibo_id}&mid={weibo_id}&max_id_type=0'
            else:
                api_url = f'https://m.weibo.cn/comments/hotflow?id={weibo_id}&mid={weibo_id}&max_id={max_id}&max_id_type={max_id_type}'

            try:
                response = curl_requests.get(
                    api_url,
                    headers=self._get_headers(),
                    proxies=self.proxies,
                    impersonate='chrome120',
                    timeout=15
                )
                if response.status_code != 200:
                    break
                data = response.json()
            except:
                break

            if data is None or data.get('ok') != 1:
                break

            comments_data = data.get('data', {})
            comments = comments_data.get('data', [])

            if not comments:
                break

            for comment in comments:
                comment_info = {
                    'comment_id': comment.get('id'),
                    'user_id': comment.get('user', {}).get('id'),
                    'user_name': comment.get('user', {}).get('screen_name'),
                    'content': self._clean_html(comment.get('text', '')),
                    'like_count': comment.get('like_count', 0),
                    'total_number': comment.get('total_number', 0),
                    'created_at': comment.get('created_at'),
                    'source': comment.get('source', ''),
                }
                all_comments.append(comment_info)

            max_id = comments_data.get('max_id', 0)
            max_id_type = comments_data.get('max_id_type', 0)

            if max_id == 0:
                break

            time.sleep(random.uniform(0.5, 1.0))

        return all_comments

    def _clean_html(self, text):
        """清理HTML标签"""
        if not text:
            return ''
        clean = re.sub(r'<[^>]+>', '', str(text))
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean

    def test_cookie(self):
        """测试Cookie是否有效"""
        print("测试Cookie有效性...")
        test_url = 'https://m.weibo.cn/api/config'

        for attempt in range(3):
            try:
                resp = curl_requests.get(
                    test_url,
                    headers=self._get_headers(),
                    proxies=self.proxies,
                    impersonate='chrome120',
                    timeout=15
                )
                if resp.status_code == 200:
                    data = resp.json()
                    login_status = data.get('data', {}).get('login', False)
                    if login_status:
                        user = data.get('data', {}).get('user', {})
                        screen_name = user.get('screen_name', '未知')
                        print(f"[OK] Cookie有效！登录用户: {screen_name}")
                        return True
                    else:
                        print("[FAIL] Cookie无效或未登录")
                        return False
            except Exception as e:
                time.sleep(0.5)
                continue

        print("[FAIL] 多次尝试后仍无法连接")
        return False

    def _load_checkpoint(self, filepath):
        """加载检查点"""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_checkpoint(self, filepath, data):
        """保存检查点"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def update_date_folder(self, date_folder, clean_checkpoint=False):
        """更新单个日期文件夹的数据"""
        date_name = os.path.basename(date_folder)
        checkpoint_file = f'{date_folder}/.update_checkpoint.json'

        if clean_checkpoint and os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        checkpoint = self._load_checkpoint(checkpoint_file)

        if checkpoint.get('all_done'):
            print(f"\n[跳过] {date_name} 已完成处理")
            return checkpoint.get('result')

        print(f"\n{'='*60}")
        print(f"处理日期: {date_name}")
        print(f"{'='*60}")

        weibo_file = f'{date_folder}/热门微博.csv'
        comment_file = f'{date_folder}/热门微博评论数据.csv'
        repost_file = f'{date_folder}/热门微博转发数据.csv'

        if not os.path.exists(weibo_file):
            print(f"[跳过] 未找到热门微博文件: {weibo_file}")
            return

        # ==================== 1. 更新热门微博数据 ====================
        if not checkpoint.get('step1_done'):
            print(f"\n--- 1. 更新热门微博数据 ---")
            weibo_df = pd.read_csv(weibo_file, encoding='utf-8-sig')
            original_count = len(weibo_df)
            print(f"原始微博数: {original_count}")

            url_col = 'weibo_url' if 'weibo_url' in weibo_df.columns else '微博链接'

            valid_weibos = []
            deleted_urls = []

            # 定义处理单条微博的函数
            def process_weibo(row):
                weibo_url = row[url_col]
                info, status = self.get_weibo_info(weibo_url)
                time.sleep(random.uniform(*REQUEST_DELAY))

                if status == 'success' and info:
                    row_dict = row.to_dict()
                    row_dict['repost_count'] = info['reposts_count']
                    row_dict['comment_count'] = info['comments_count']
                    row_dict['like_count'] = info['attitudes_count']
                    return ('success', row_dict, weibo_url)
                else:
                    return ('failed', None, weibo_url)

            # 并行处理
            rows = [row for _, row in weibo_df.iterrows()]
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(process_weibo, row): row for row in rows}
                for future in tqdm(as_completed(futures), total=len(futures), desc="爬取微博"):
                    result_type, row_dict, url = future.result()
                    if result_type == 'success':
                        valid_weibos.append(row_dict)
                    else:
                        deleted_urls.append(url)

            if not valid_weibos and original_count > 0:
                print(f"[ERROR] 所有 {original_count} 条微博请求都失败！")
                return None

            if valid_weibos:
                weibo_df_updated = pd.DataFrame(valid_weibos)
                weibo_df_updated.to_csv(weibo_file, index=False, encoding='utf-8-sig')
                print(f"[OK] 微博数据更新完成: {len(valid_weibos)}/{original_count} (删除 {len(deleted_urls)} 条)")

            checkpoint['step1_done'] = True
            checkpoint['url_col'] = url_col
            checkpoint['valid_weibos'] = valid_weibos
            checkpoint['deleted_count'] = len(deleted_urls)
            checkpoint['original_count'] = original_count
            self._save_checkpoint(checkpoint_file, checkpoint)
        else:
            print(f"\n--- 1. 更新热门微博数据 [已完成，跳过] ---")
            url_col = checkpoint.get('url_col', '微博链接')
            valid_weibos = checkpoint.get('valid_weibos', [])

        original_count = checkpoint.get('original_count', len(valid_weibos))
        deleted_count = checkpoint.get('deleted_count', 0)

        if not valid_weibos and original_count > 0:
            print(f"[ERROR] 检查点数据异常")
            return None

        # ==================== 2. 更新评论数据 ====================
        if os.path.exists(comment_file) and not checkpoint.get('step2_done'):
            print(f"\n--- 2. 更新评论数据 ---")
            comment_df = pd.read_csv(comment_file, encoding='utf-8-sig')
            original_comment_count = len(comment_df)
            print(f"原始评论数: {original_comment_count}")

            valid_urls = [w[url_col] for w in valid_weibos]
            url_col_comment = '原文链接'
            comment_df_filtered = comment_df[comment_df[url_col_comment].isin(valid_urls)].copy()
            print(f"过滤后评论数: {len(comment_df_filtered)}")

            comment_likes_map = {}
            unique_urls = comment_df_filtered[url_col_comment].unique()

            # 定义处理单个微博评论的函数
            def fetch_comments(weibo_url):
                comments = self.get_all_comments(weibo_url)
                time.sleep(random.uniform(0.5, 1.0))
                result = {}
                for c in comments:
                    result[str(c['comment_id'])] = c['like_count']
                return result

            # 并行处理
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(fetch_comments, url): url for url in unique_urls}
                for future in tqdm(as_completed(futures), total=len(futures), desc="爬取评论"):
                    result = future.result()
                    comment_likes_map.update(result)

            comment_df_filtered['点赞数'] = comment_df_filtered['评论ID'].astype(str).map(
                lambda x: comment_likes_map.get(x, 0)
            )

            comment_df_filtered.to_csv(comment_file, index=False, encoding='utf-8-sig')
            print(f"[OK] 评论数据已保存")

            checkpoint['step2_done'] = True
            self._save_checkpoint(checkpoint_file, checkpoint)

        # ==================== 3. 更新转发数据 ====================
        if os.path.exists(repost_file) and not checkpoint.get('step3_done'):
            print(f"\n--- 3. 更新转发数据 ---")
            repost_df = pd.read_csv(repost_file, encoding='utf-8-sig')
            original_repost_count = len(repost_df)
            print(f"原始转发数: {original_repost_count}")

            type_col = '原创/转发' if '原创/转发' in repost_df.columns else 'type'
            url_col_repost = '原文链接'

            weibo_stats_map = {
                w[url_col]: {
                    'repost_count': w['repost_count'],
                    'comment_count': w['comment_count'],
                    'like_count': w['like_count']
                }
                for w in valid_weibos
            }

            valid_urls = [w[url_col] for w in valid_weibos]

            current_original_url = None
            group_urls = []
            for idx, row in repost_df.iterrows():
                if row[type_col] == '原创':
                    current_original_url = row[url_col_repost]
                group_urls.append(current_original_url)

            repost_df['_group_url'] = group_urls
            repost_df_filtered = repost_df[repost_df['_group_url'].isin(valid_urls)].copy()

            original_mask = repost_df_filtered[type_col] == '原创'
            for idx in repost_df_filtered[original_mask].index:
                url = repost_df_filtered.loc[idx, url_col_repost]
                if url in weibo_stats_map:
                    stats = weibo_stats_map[url]
                    repost_df_filtered.loc[idx, '转发数'] = stats['repost_count']
                    repost_df_filtered.loc[idx, '评论数'] = stats['comment_count']
                    repost_df_filtered.loc[idx, '点赞数'] = stats['like_count']

            repost_df_filtered = repost_df_filtered.drop(columns=['_group_url'])
            repost_df_filtered.to_csv(repost_file, index=False, encoding='utf-8-sig')
            print(f"[OK] 转发数据已保存")

            checkpoint['step3_done'] = True
            self._save_checkpoint(checkpoint_file, checkpoint)

        result = {
            'date': date_name,
            'weibo': {'original': original_count, 'valid': len(valid_weibos), 'deleted': deleted_count},
        }
        checkpoint['all_done'] = True
        checkpoint['result'] = result
        checkpoint.pop('valid_weibos', None)
        self._save_checkpoint(checkpoint_file, checkpoint)

        print(f"\n[OK] {date_name} 处理完成!")
        return result


def get_date_folders(data_root, start_date=None, end_date=None, single_date=None):
    """获取指定日期范围的文件夹"""
    all_folders = sorted(glob(f'{data_root}/2025-*'))

    if single_date:
        folder = f'{data_root}/{single_date}'
        return [folder] if os.path.exists(folder) else []

    if start_date and end_date:
        filtered = []
        for folder in all_folders:
            date_str = os.path.basename(folder)
            if start_date <= date_str <= end_date:
                filtered.append(folder)
        return filtered

    return all_folders


def main():
    parser = argparse.ArgumentParser(description='微博数据更新器')
    parser.add_argument('--start', type=str, help='开始日期 (格式: 2025-03-27)')
    parser.add_argument('--end', type=str, help='结束日期 (格式: 2025-04-14)')
    parser.add_argument('--date', type=str, help='单个日期 (格式: 2025-03-27)')
    parser.add_argument('--all', action='store_true', help='处理所有日期')
    parser.add_argument('--cookie', type=str, help='微博Cookie')
    parser.add_argument('--clean', action='store_true', help='清除检查点，强制重新处理')

    args = parser.parse_args()

    if not args.all and not args.date and not (args.start and args.end):
        args.all = True
        print("[默认] 未指定范围，将处理所有日期")

    if args.all:
        folders = get_date_folders(DATA_ROOT)
    elif args.date:
        folders = get_date_folders(DATA_ROOT, single_date=args.date)
    else:
        folders = get_date_folders(DATA_ROOT, start_date=args.start, end_date=args.end)

    if not folders:
        print("未找到符合条件的日期文件夹")
        return

    print(f"找到 {len(folders)} 个日期文件夹:")
    for f in folders:
        print(f"  - {os.path.basename(f)}")

    updater = WeiboDataUpdater(cookie=args.cookie)

    if not updater.test_cookie():
        print("\n请更新Cookie后重试")
        return

    results = []
    for folder in folders:
        result = updater.update_date_folder(folder, clean_checkpoint=args.clean)
        if result:
            results.append(result)

    print(f"\n{'='*60}")
    print("【处理总结】")
    print(f"{'='*60}")
    for r in results:
        print(f"{r['date']}: 微博 {r['weibo']['valid']}/{r['weibo']['original']} (删除 {r['weibo']['deleted']})")

    print(f"\n全部完成!")


if __name__ == '__main__':
    main()
