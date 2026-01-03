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

消融实验：
    python src/main.py --model bge_nn --ablation no_cross_attn    # w/o Cross-Attention
    python src/main.py --model bge_nn --ablation no_density       # w/o 重复特征
    python src/main.py --model bge_nn --ablation no_context       # w/o 上下文（只使用评论文本）
    python src/main.py --model bge_nn --ablation std_nll          # w/o Log NLL (Standard NLL)

    # 或使用单独参数组合多个消融
    python src/main.py --model bge_nn --no_cross_attention --no_density

指定输出目录：
    python src/main.py --model bge_nn --output_dir results/my_experiment

使用预分词缓存（加速训练）：
    python src/pretokenize.py                    # 首次：生成缓存
    python src/main.py --model bge_nn --use_cache  # 使用缓存训练
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

import pickle


def find_latest_checkpoint(model_name):
    """查找最新的模型checkpoint目录

    参数:
        model_name: 模型名称

    返回:
        最新的checkpoint目录路径，如果不存在则返回None
    """
    base_name = f'comment_pred_{model_name}'
    existing = list(RESULT_DIR.glob(f'{base_name}_*'))

    if not existing:
        return None

    # 找到最大的编号
    max_num = 0
    latest_dir = None
    for p in existing:
        try:
            num = int(p.name.split('_')[-1])
            if num > max_num:
                max_num = num
                latest_dir = p
        except ValueError:
            pass

    return latest_dir


def load_checkpoint(checkpoint_path, model_name):
    """加载模型checkpoint

    参数:
        checkpoint_path: checkpoint目录路径
        model_name: 模型名称

    返回:
        (model, config) 元组
    """
    checkpoint_path = Path(checkpoint_path)

    # 加载传统模型
    if model_name in ['rf', 'ridge', 'lasso', 'gbdt', 'xgboost', 'lightgbm', 'ngboost']:
        model_file = checkpoint_path / 'model.pkl'
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_file}")

        with open(model_file, 'rb') as f:
            model = pickle.load(f)

    # 加载BGE神经网络模型
    elif model_name in ['bge_nn', 'bge_mini']:
        model_file = checkpoint_path / 'model.pt'
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_file}")

        # 创建模型实例
        model_cls = MODEL_REGISTRY[model_name]

        # 加载配置
        config_file = checkpoint_path / 'config.json' if (checkpoint_path / 'config.json').exists() else checkpoint_path / 'args.json'
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        model = model_cls(
            freeze_bert=config.get('freeze_bert', True) if 'freeze_bert' in config else not config.get('finetune_bge', False),
            hidden_size=config.get('hidden_size', 256),
            dropout=config.get('dropout', 0.1),
            epochs=config.get('epochs', 30),
            batch_size=config.get('batch_size', 32),
            learning_rate=config.get('nn_learning_rate', 1e-4),
            patience=config.get('patience', 5)
        )

        # 加载权重
        checkpoint = torch.load(model_file, map_location='cpu')
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.model.eval()

    else:
        raise ValueError(f"未知模型类型: {model_name}")

    # 加载配置
    config_file = checkpoint_path / 'config.json' if (checkpoint_path / 'config.json').exists() else checkpoint_path / 'args.json'
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return model, config


def test_only(args):
    """仅测试模式：加载已保存的模型并在测试集上评估

    参数:
        args: 命令行参数
    """
    # 确定checkpoint路径
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = RESULT_DIR / args.checkpoint
    else:
        checkpoint_path = find_latest_checkpoint(args.model)

    if checkpoint_path is None or not checkpoint_path.exists():
        print(f"错误: 找不到模型checkpoint")
        print(f"请指定 --checkpoint 参数，或确保 results/comment_pred_{args.model}_* 目录存在")
        return

    print("=" * 60)
    print(f"【测试模式 - 模型: {args.model}】")
    print("=" * 60)
    print(f"加载checkpoint: {checkpoint_path}")

    # 加载模型和配置
    model, config = load_checkpoint(checkpoint_path, args.model)
    print(f"模型加载成功: {model.name if hasattr(model, 'name') else args.model}")

    # 解析特征组
    feature_groups_str = config.get('features', args.features)
    if feature_groups_str == 'all':
        feature_groups = ['base', 'text', 'lda', 'density']
    else:
        feature_groups = [g.strip() for g in feature_groups_str.split(',')]

    # 获取特征列
    feature_cols = get_feature_cols(feature_groups)
    print(f"特征组: {feature_groups}")
    print(f"特征列 ({len(feature_cols)}个): {feature_cols}")

    # 加载数据
    print("\n【加载数据】")
    train_df, val_df, test_df = load_data(use_pkl=True)

    # 加载外部特征（如需要）
    test_lda, test_density = None, None

    if 'lda' in feature_groups:
        _, _, test_lda = load_lda_features()
        if test_lda is None:
            feature_groups.remove('lda')
            feature_cols = get_feature_cols(feature_groups)

    if 'density' in feature_groups:
        density_method = config.get('density_method', args.density_method)
        _, _, test_density = load_density_features(density_method)
        if test_density is None:
            feature_groups.remove('density')
            feature_cols = get_feature_cols(feature_groups)

    # 特征工程
    print("\n【特征工程】")
    if 'base' in feature_groups:
        print("  提取时间特征...")
        test_df = extract_time_features(test_df)

    if 'text' in feature_groups:
        print("  提取文本特征...")
        test_df = extract_text_features(test_df)

    if 'lda' in feature_groups or 'density' in feature_groups:
        print("  合并外部特征...")
        test_df = merge_external_features(test_df, test_lda if 'lda' in feature_groups else None, test_density)

    # BGE神经网络模型测试
    if args.model in ['bge_nn', 'bge_mini']:
        return test_bge_nn(args, model, test_df, test_density, checkpoint_path)

    # 传统模型测试
    print("\n【准备特征】")
    X_test, y_test = prepare_features(test_df, feature_cols)
    print(f"测试集特征形状: {X_test.shape}")

    # 预测
    print("\n【评估结果】")
    y_test_pred = model.predict(X_test)

    # 获取不确定性估计（如果支持）
    y_test_std = None
    if hasattr(model, 'supports_uncertainty') and model.supports_uncertainty:
        _, y_test_std = model.predict_dist(X_test)

    # 计算评估指标
    test_metrics = evaluate(y_test, y_test_pred, prefix='test_', y_std=y_test_std)

    # 打印结果
    print(f"\n{'指标':<15} {'值':<12}")
    print("-" * 30)
    # print(f"{'RMSE':<15} {test_metrics['test_RMSE']:<12.4f}") # 该指标意义不大
    print(f"{'MAE':<15} {test_metrics['test_MAE']:<12.4f}")
    print(f"{'MSLE':<15} {test_metrics['test_MSLE']:<12.4f}")
    # print(f"{'R2':<15} {test_metrics['test_R2']:<12.4f}") # 该指标无意义
    print(f"{'ACP@20%':<15} {test_metrics['test_ACP@20%']*100:<12.2f}%")
    print(f"{'ACP@50%':<15} {test_metrics['test_ACP@50%']*100:<12.2f}%")

    if y_test_std is not None:
        print(f"\n【不确定性指标】")
        print(f"{'LogNLL':<15} {test_metrics['test_LogNLL']:<12.4f}")
        print(f"{'PICP@95%':<15} {test_metrics['test_PICP@95%']*100:<12.2f}%")
        print(f"{'MPIW@95%':<15} {test_metrics['test_MPIW@95%']:<12.4f}")

    print("\n" + "=" * 60)
    print("【测试完成】")
    print("=" * 60)


def test_bge_nn(args, model, test_df, test_density, checkpoint_path):
    """BGE神经网络模型测试"""
    print("\n【评估结果】")

    y_test_pred = model.predict(test_df, test_density)
    _, y_test_std = model.predict_dist(test_df, test_density)
    y_test = test_df['子评论数'].values

    # 计算评估指标
    test_metrics = evaluate(y_test, y_test_pred, prefix='test_', y_std=y_test_std)

    # 计算NLL
    test_nll = model.compute_nll(test_df, test_density)
    test_metrics['test_NLL'] = test_nll

    # 打印结果
    print(f"\n{'指标':<15} {'值':<12}")
    print("-" * 30)
    print(f"{'RMSE':<15} {test_metrics['test_RMSE']:<12.4f}")
    print(f"{'MAE':<15} {test_metrics['test_MAE']:<12.4f}")
    print(f"{'MSLE':<15} {test_metrics['test_MSLE']:<12.4f}")
    print(f"{'R2':<15} {test_metrics['test_R2']:<12.4f}")
    print(f"{'ACP@20%':<15} {test_metrics['test_ACP@20%']*100:<12.2f}%")
    print(f"{'ACP@50%':<15} {test_metrics['test_ACP@50%']*100:<12.2f}%")

    print(f"\n【不确定性指标】")
    print(f"{'NLL':<15} {test_nll:<12.4f}")
    print(f"{'LogNLL':<15} {test_metrics['test_LogNLL']:<12.4f}")
    print(f"{'PICP@95%':<15} {test_metrics['test_PICP@95%']*100:<12.2f}%")
    print(f"{'MPIW@95%':<15} {test_metrics['test_MPIW@95%']:<12.4f}")

    print("\n" + "=" * 60)
    print("【测试完成】")
    print("=" * 60)


def train(args):
    """主训练流程"""
    # 解析特征组
    if args.features == 'all':
        feature_groups = ['base', 'text', 'lda', 'density']
    else:
        feature_groups = [g.strip() for g in args.features.split(',')]

    # 获取特征列
    feature_cols = get_feature_cols(feature_groups)

    mode_str = {'train': '仅训练', 'test': '仅测试', 'full': '训练+测试'}[args.mode]
    print("=" * 60)
    print(f"【{mode_str} - 模型: {args.model}】")
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

    # BGE神经网络模型需要特殊处理
    if args.model in ['bge_nn', 'bge_mini']:
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

    # NGBoost: 获取不确定性估计
    y_train_std, y_val_std, y_test_std = None, None, None
    if hasattr(model, 'supports_uncertainty') and model.supports_uncertainty:
        _, y_train_std = model.predict_dist(X_train)
        _, y_val_std = model.predict_dist(X_val)

    # 计算评估指标（包含不确定性指标）
    train_metrics = evaluate(y_train, y_train_pred, prefix='train_', y_std=y_train_std)
    val_metrics = evaluate(y_val, y_val_pred, prefix='val_', y_std=y_val_std)

    all_metrics = {**train_metrics, **val_metrics}
    all_metrics['feature_groups'] = ','.join(feature_groups)
    all_metrics['feature_cols'] = feature_cols

    # NGBoost: 添加NLL指标
    if hasattr(model, 'supports_uncertainty') and model.supports_uncertainty:
        train_nll = model.compute_nll(X_train, y_train)
        val_nll = model.compute_nll(X_val, y_val)
        all_metrics['train_NLL'] = train_nll
        all_metrics['val_NLL'] = val_nll

    # full模式：评估测试集
    y_test_pred = None
    test_metrics = {}
    if args.mode == 'full':
        y_test_pred = model.predict(X_test)
        if hasattr(model, 'supports_uncertainty') and model.supports_uncertainty:
            _, y_test_std = model.predict_dist(X_test)
        test_metrics = evaluate(y_test, y_test_pred, prefix='test_', y_std=y_test_std)
        all_metrics.update(test_metrics)

        if hasattr(model, 'supports_uncertainty') and model.supports_uncertainty:
            test_nll = model.compute_nll(X_test, y_test)
            all_metrics['test_NLL'] = test_nll

    # 打印评估结果
    print_metrics(train_metrics, val_metrics, test_metrics, all_metrics,
                  y_test_std if hasattr(model, 'supports_uncertainty') and args.mode == 'full' else None)

    # 8. 保存结果
    if args.output_dir:
        result_dir = Path(args.output_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
    else:
        result_dir = get_result_dir(args.model)
    save_results(result_dir, model, all_metrics, args, X_test, y_test,
                 y_test_pred if y_test_pred is not None else np.zeros_like(y_test),
                 y_std=y_test_std, feature_cols=feature_cols)

    print("\n" + "=" * 60)
    print("【训练完成】")
    print("=" * 60)


def train_bge_nn(args, model_cls, train_df, val_df, test_df,
                 train_density, val_density, test_density):
    """BGE神经网络模型训练流程"""
    # 处理消融实验参数
    # 快捷方式：--ablation 参数
    use_cross_attention = True
    use_context = True
    use_density_features = True
    loss_type = args.loss_type

    if args.ablation:
        if args.ablation == 'no_cross_attn':
            use_cross_attention = False
        elif args.ablation == 'no_density':
            use_density_features = False
        elif args.ablation == 'no_context':
            use_context = False
        elif args.ablation == 'std_nll':
            loss_type = 'standard_nll'

    # 单独参数也可以覆盖
    if args.no_cross_attention:
        use_cross_attention = False
    if args.no_context:
        use_context = False
    if args.no_density:
        use_density_features = False

    # 解析缓存目录
    cache_dir = None
    if args.use_cache:
        from .config import ROOT_DIR
        cache_dir = ROOT_DIR / 'cache'
    elif args.cache_dir:
        cache_dir = Path(args.cache_dir)

    if cache_dir and not (cache_dir / 'tokenized_texts.pt').exists():
        print(f"警告: 缓存目录 {cache_dir} 不存在或不完整，将使用实时分词")
        print("提示: 运行 python src/pretokenize.py 生成缓存")
        cache_dir = None

    model = model_cls(
        freeze_bert=not args.finetune_bge,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.nn_learning_rate,
        patience=args.patience,
        # 消融实验参数
        loss_type=loss_type,
        use_cross_attention=use_cross_attention,
        use_context=use_context,
        use_density_features=use_density_features
    )
    print(f"\n模型: {model.name}")
    print(f"  冻结BGE: {model.freeze_bert}")
    print(f"  隐藏层大小: {model.hidden_size}")
    print(f"  Dropout: {model.dropout}")
    print(f"  Epochs: {model.epochs}")
    print(f"  Batch Size: {model.batch_size}")
    print(f"  学习率: {model.learning_rate}")
    if cache_dir:
        print(f"  预分词缓存: {cache_dir}")

    # 创建结果目录（用于保存训练过程中的权重）
    if args.output_dir:
        result_dir = Path(args.output_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
    else:
        result_dir = get_result_dir(args.model)
    print(f"\n结果保存目录: {result_dir}")

    # 训练（传入save_dir以便每个epoch保存权重）
    # full模式下同时传入测试集，提前分词避免评估时重复分词
    print("\n【训练中...】")
    if args.mode == 'full':
        model.fit(train_df, val_df, train_density, val_density, save_dir=result_dir,
                  test_df=test_df, test_density=test_density, cache_dir=cache_dir)
    else:
        model.fit(train_df, val_df, train_density, val_density, save_dir=result_dir,
                  cache_dir=cache_dir)
    print("训练完成!")

    # 评估（使用缓存的数据集，无需重新分词）
    print("\n【评估结果】")

    # 使用 evaluate_all 方法一次性获取所有评估指标
    y_train_pred, y_train_std, train_nll = model.evaluate_all(use_cached='train')
    y_val_pred, y_val_std, val_nll = model.evaluate_all(use_cached='val')

    # 获取真实值
    y_train = train_df['子评论数'].values
    y_val = val_df['子评论数'].values
    y_test = test_df['子评论数'].values

    # 根据 loss_type 确定 sigma_space
    sigma_space = 'original' if model.loss_type == 'standard_nll' else 'log'

    # 计算评估指标
    train_metrics = evaluate(y_train, y_train_pred, prefix='train_', y_std=y_train_std, sigma_space=sigma_space)
    val_metrics = evaluate(y_val, y_val_pred, prefix='val_', y_std=y_val_std, sigma_space=sigma_space)

    all_metrics = {**train_metrics, **val_metrics}
    all_metrics['model'] = args.model
    all_metrics['freeze_bert'] = model.freeze_bert

    # 添加NLL指标
    all_metrics['train_NLL'] = train_nll
    all_metrics['val_NLL'] = val_nll

    # full模式：评估测试集（使用缓存的数据集）
    y_test_pred = None
    y_test_std = None
    test_metrics = {}
    if args.mode == 'full':
        y_test_pred, y_test_std, test_nll = model.evaluate_all(use_cached='test')
        test_metrics = evaluate(y_test, y_test_pred, prefix='test_', y_std=y_test_std, sigma_space=sigma_space)
        all_metrics.update(test_metrics)
        all_metrics['test_NLL'] = test_nll

    # 打印评估结果
    print_metrics(train_metrics, val_metrics, test_metrics, all_metrics,
                  y_test_std if args.mode == 'full' else None)

    # 保存结果（result_dir 已在训练前创建）

    # 保存最终模型（覆盖训练过程中的 model_best.pt）
    model_path = result_dir / 'model.pt'
    torch.save({
        'model_state_dict': model.model.state_dict(),
        'train_losses': model.train_losses,
        'val_losses': model.val_losses,
    }, model_path)
    print(f"\n最终模型已保存: {model_path}")

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

    # 预测效果图（full模式用测试集，train模式用验证集）
    ax = axes[1]
    if args.mode == 'full' and y_test_pred is not None:
        plot_y_true, plot_y_pred, plot_y_std = y_test, y_test_pred, y_test_std
        ax_label = '测试集'
    else:
        plot_y_true, plot_y_pred, plot_y_std = y_val, y_val_pred, y_val_std
        ax_label = '验证集'

    sample_size = min(1000, len(plot_y_true))
    idx = np.random.choice(len(plot_y_true), sample_size, replace=False)
    ax.errorbar(plot_y_true[idx], plot_y_pred[idx], yerr=plot_y_std[idx], fmt='o', alpha=0.3,
                markersize=2, elinewidth=0.5, capsize=0)
    max_val = min(plot_y_true.max(), 100)
    ax.plot([0, max_val], [0, max_val], 'r--', label='完美预测', linewidth=2)
    ax.set_xlabel(f'实际子评论数 ({ax_label})')
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
    has_test = len(test_metrics) > 0

    # 打印基础指标
    print(f"\n{'数据集':<8} {'RMSE':<10} {'MAE':<10} {'MSLE':<10} {'R2':<10}")
    print("-" * 50)
    print(f"{'训练集':<8} {train_metrics['train_RMSE']:<10.4f} {train_metrics['train_MAE']:<10.4f} {train_metrics['train_MSLE']:<10.4f} {train_metrics['train_R2']:<10.4f}")
    print(f"{'验证集':<8} {val_metrics['val_RMSE']:<10.4f} {val_metrics['val_MAE']:<10.4f} {val_metrics['val_MSLE']:<10.4f} {val_metrics['val_R2']:<10.4f}")
    if has_test:
        print(f"{'测试集':<8} {test_metrics['test_RMSE']:<10.4f} {test_metrics['test_MAE']:<10.4f} {test_metrics['test_MSLE']:<10.4f} {test_metrics['test_R2']:<10.4f}")

    # 打印ACP指标
    print(f"\n{'数据集':<8} {'ACP@20%':<12} {'ACP@50%':<12}")
    print("-" * 35)
    print(f"{'训练集':<8} {train_metrics['train_ACP@20%']*100:<12.2f}% {train_metrics['train_ACP@50%']*100:<12.2f}%")
    print(f"{'验证集':<8} {val_metrics['val_ACP@20%']*100:<12.2f}% {val_metrics['val_ACP@50%']*100:<12.2f}%")
    if has_test:
        print(f"{'测试集':<8} {test_metrics['test_ACP@20%']*100:<12.2f}% {test_metrics['test_ACP@50%']*100:<12.2f}%")

    # 打印不确定性指标（如果有）
    if 'train_NLL' in all_metrics:
        print("\n【不确定性估计】")
        print(f"{'数据集':<8} {'NLL':<12} {'LogNLL':<12} {'PICP@95%':<12} {'MPIW':<10}")
        print("-" * 55)
        print(f"{'训练集':<8} {all_metrics['train_NLL']:<12.4f} {train_metrics['train_LogNLL']:<12.4f} {train_metrics['train_PICP@95%']*100:<12.2f}% {train_metrics['train_MPIW@95%']:<10.4f}")
        print(f"{'验证集':<8} {all_metrics['val_NLL']:<12.4f} {val_metrics['val_LogNLL']:<12.4f} {val_metrics['val_PICP@95%']*100:<12.2f}% {val_metrics['val_MPIW@95%']:<10.4f}")
        if has_test and 'test_NLL' in all_metrics:
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

    # 运行模式
    parser.add_argument('--mode', type=str, default='full',
                        choices=['train', 'test', 'full'],
                        help='运行模式: train(仅训练), test(仅测试), full(训练+测试) (default: full)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='测试模式时加载的模型路径（目录），如 results/rf_1 (default: None，自动查找最新)')

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

    # 消融实验参数
    parser.add_argument('--ablation', type=str, default=None,
                        choices=['no_cross_attn', 'no_density', 'no_context', 'std_nll'],
                        help='消融实验类型: no_cross_attn(w/o Cross-Attention), '
                             'no_density(w/o 重复特征), no_context(w/o 上下文), '
                             'std_nll(w/o Log NLL) (default: None)')
    parser.add_argument('--loss_type', type=str, default='log_nll',
                        choices=['log_nll', 'standard_nll'],
                        help='损失函数类型 (default: log_nll)')
    parser.add_argument('--no_cross_attention', action='store_true',
                        help='禁用Cross-Attention (消融实验)')
    parser.add_argument('--no_context', action='store_true',
                        help='禁用上下文文本，只使用评论文本 (消融实验)')
    parser.add_argument('--no_density', action='store_true',
                        help='禁用时间密度特征 (消融实验)')

    # 其他参数
    parser.add_argument('--output_dir', type=str, default=None,
                        help='指定结果保存目录路径（如不指定则自动生成）')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='预分词缓存目录路径')
    parser.add_argument('--use_cache', action='store_true',
                        help='使用默认缓存目录 cache/（等效于 --cache_dir cache）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (default: 42)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.mode == 'test':
        test_only(args)
    else:
        # train 或 full 模式都执行训练流程
        train(args)
