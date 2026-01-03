"""
评论数预测模型训练脚本 (重构版)

代码已重构为模块化结构:
- src/config.py: 配置文件
- src/models/: 模型定义
- src/data/: 数据加载和特征提取
- src/evaluation/: 评估指标
- src/utils/: 工具函数
- src/main.py: 主训练逻辑

使用方法：
    python -m src --model rf
    python -m src --model rf --features base,text
    python -m src --model rf --features base,text,lda,density
    python -m src --model xgboost --features all
    python -m src --model ngboost --features base,text,density
    python -m src --model bge_nn --epochs 30 --batch_size 32

旧的使用方法（兼容）：
    python src/train_comment_predictor.py --model rf --features base,text
"""

import sys
import os

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# 从新模块导入并运行
from src.main import parse_args, train

if __name__ == '__main__':
    args = parse_args()
    train(args)
