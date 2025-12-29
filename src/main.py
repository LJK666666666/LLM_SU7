"""
评论数预测模型训练脚本
预测目标：评论的子评论数

使用方法：
    python src/main.py --model rf
    python src/main.py --model rf --features base,text
    python src/main.py --model rf --features base,text,lda,density
    python src/main.py --model xgboost --features all
    python src/main.py --model ngboost --features base,text,density
    python src/main.py --model bge_nn --epochs 30 --batch_size 32
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入配置
from .config import ROOT_DIR, RESULT_DIR, TARGET_COL, LOG_OFFSET, FEATURE_GROUPS

# 导入模型
from .models import MODEL_REGISTRY

# 导入数据处理
from .data import (
    load_data, load_lda_features, load_density_features,
    extract_time_features, extract_text_features,
    merge_external_features, get_feature_cols, prepare_features
)

# 导入评估
from .evaluation import evaluate

# 导入工具
from .utils import get_result_dir, save_results


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
        return train_bge_nn(args, model_cls, train_df, val_df, test_df,
                           train_density, val_density, test_density)

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

    # 打印评估结果
    print_metrics(train_metrics, val_metrics, test_metrics, all_metrics,
                  y_test_std if hasattr(model, 'supports_uncertainty') else None)

    # 8. 保存结果
    result_dir = get_result_dir(args.model)
    save_results(result_dir, model, all_metrics, args, X_test, y_test, y_test_pred,
                 y_std=y_test_std, feature_cols=feature_cols)

    print("\n" + "=" * 60)
    print("【训练完成】")
    print("=" * 60)


def train_bge_nn(args, model_cls, train_df, val_df, test_df,
                 train_density, val_density, test_density):
    """BGE神经网络模型训练流程"""
    model = model_cls(
        freeze_bert=not args.finetune_bge,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.nn_learning_rate,
        patience=args.patience
    )
    print(f"\n模型: {model.name}")
    print(f"  冻结BGE: {model.freeze_bert}")
    print(f"  隐藏层大小: {model.hidden_size}")
    print(f"  Dropout: {model.dropout}")
    print(f"  Epochs: {model.epochs}")
    print(f"  Batch Size: {model.batch_size}")
    print(f"  学习率: {model.learning_rate}")

    # 训练
    print("\n【训练中...】")
    model.fit(train_df, val_df, train_density, val_density)
    print("训练完成!")

    # 评估
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

    # 打印评估结果
    print_metrics(train_metrics, val_metrics, test_metrics, all_metrics, y_test_std)

    # 保存结果
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
        json.dump(all_metrics, f, indent=2, ensure_ascii=False, default=float)
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


def print_metrics(train_metrics, val_metrics, test_metrics, all_metrics, y_test_std=None):
    """打印评估指标"""
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

    # 打印不确定性指标（如果有）
    if 'train_NLL' in all_metrics:
        print("\n【不确定性估计】")
        print(f"{'数据集':<8} {'NLL':<12} {'LogNLL':<12} {'PICP@95%':<12} {'MPIW':<10}")
        print("-" * 55)
        print(f"{'训练集':<8} {all_metrics['train_NLL']:<12.4f} {train_metrics['train_LogNLL']:<12.4f} {train_metrics['train_PICP@95%']*100:<12.2f}% {train_metrics['train_MPIW@95%']:<10.4f}")
        print(f"{'验证集':<8} {all_metrics['val_NLL']:<12.4f} {val_metrics['val_LogNLL']:<12.4f} {val_metrics['val_PICP@95%']*100:<12.2f}% {val_metrics['val_MPIW@95%']:<10.4f}")
        print(f"{'测试集':<8} {all_metrics['test_NLL']:<12.4f} {test_metrics['test_LogNLL']:<12.4f} {test_metrics['test_PICP@95%']*100:<12.2f}% {test_metrics['test_MPIW@95%']:<10.4f}")

        if y_test_std is not None:
            print(f"\n预测标准差统计 (log空间):")
            print(f"  均值: {y_test_std.mean():.4f}")
            print(f"  中位数: {np.median(y_test_std):.4f}")
            print(f"  最小值: {y_test_std.min():.4f}")
            print(f"  最大值: {y_test_std.max():.4f}")


def parse_args():
    """解析命令行参数"""
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
    parser.add_argument('--nn_learning_rate', type=float, default=1e-4,
                        help='神经网络学习率，用于bge_nn (default: 1e-4)')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (default: 42)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
