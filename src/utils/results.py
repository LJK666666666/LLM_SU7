"""
结果保存模块：保存模型、指标、可视化等
"""
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ..config import RESULT_DIR


def get_result_dir(model_name):
    """获取结果保存目录

    自动创建 results/{model_name}_{num} 目录，num自动递增避免覆盖

    参数:
        model_name: 模型名称

    返回:
        结果目录路径
    """
    base_name = f'comment_pred_{model_name}'

    # 查找已存在的目录
    existing = list(RESULT_DIR.glob(f'{base_name}_*'))
    if existing:
        nums = []
        for p in existing:
            try:
                num = int(p.name.split('_')[-1])
                nums.append(num)
            except ValueError:
                pass
        next_num = max(nums) + 1 if nums else 1
    else:
        next_num = 1

    result_dir = RESULT_DIR / f'{base_name}_{next_num}'
    result_dir.mkdir(parents=True, exist_ok=True)

    return result_dir


def save_results(result_dir, model, metrics, args, X_test, y_test, y_pred, y_std=None, feature_cols=None):
    """保存所有训练结果

    保存内容:
    - model.pkl: 模型文件
    - metrics.json: 评估指标
    - config.json: 训练配置
    - prediction_plot.png: 预测可视化
    - predictions.npz: 预测结果数组
    - feature_importance.png: 特征重要性（如果模型支持）

    参数:
        result_dir: 结果目录
        model: 训练好的模型
        metrics: 评估指标字典
        args: 命令行参数
        X_test: 测试集特征
        y_test: 测试集目标
        y_pred: 预测值
        y_std: 预测标准差（可选）
        feature_cols: 特征列名（可选）
    """
    result_dir = Path(result_dir)

    # 1. 保存模型
    model_path = result_dir / 'model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"模型已保存: {model_path}")

    # 2. 保存指标
    metrics_path = result_dir / 'metrics.json'
    metrics_serializable = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            metrics_serializable[k] = float(v)
        elif isinstance(v, np.ndarray):
            metrics_serializable[k] = v.tolist()
        else:
            metrics_serializable[k] = v

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_serializable, f, ensure_ascii=False, indent=2)
    print(f"指标已保存: {metrics_path}")

    # 3. 保存配置
    config_path = result_dir / 'config.json'
    config = vars(args) if hasattr(args, '__dict__') else dict(args)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"配置已保存: {config_path}")

    # 4. 保存预测结果
    pred_path = result_dir / 'predictions.npz'
    save_dict = {'y_test': y_test, 'y_pred': y_pred}
    if y_std is not None:
        save_dict['y_std'] = y_std
    np.savez(pred_path, **save_dict)
    print(f"预测结果已保存: {pred_path}")

    # 5. 绘制预测可视化
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 散点图
    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.3, s=10)
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='理想预测')
    ax.set_xlabel('真实值')
    ax.set_ylabel('预测值')
    ax.legend()
    ax.set_xlim(0, min(100, max_val))
    ax.set_ylim(0, min(100, max_val))

    # 残差分布
    ax = axes[1]
    residuals = y_pred - y_test
    ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--')
    ax.set_xlabel('残差 (预测值 - 真实值)')
    ax.set_ylabel('频数')

    plt.tight_layout()
    plot_path = result_dir / 'prediction_plot.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"预测图已保存: {plot_path}")

    # 6. 特征重要性（如果模型支持）
    has_importance = hasattr(model, 'get_feature_importance') or hasattr(model, 'feature_importances_')
    if has_importance and feature_cols is not None:
        # 获取特征重要性
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
        else:
            importance = model.feature_importances_

        importance = np.array(importance)
        if importance.ndim > 1:
            importance = importance.mean(axis=0)  # 多维时取平均

        feature_cols_list = list(feature_cols) if not isinstance(feature_cols, list) else feature_cols

        fig, ax = plt.subplots(figsize=(10, max(6, len(feature_cols_list) * 0.3)))

        # 排序
        indices = np.argsort(importance)
        sorted_cols = [feature_cols_list[i] for i in indices]
        sorted_importance = importance[indices]

        ax.barh(range(len(sorted_cols)), sorted_importance, align='center')
        ax.set_yticks(range(len(sorted_cols)))
        ax.set_yticklabels(sorted_cols)
        ax.set_xlabel('重要性')

        plt.tight_layout()
        importance_path = result_dir / 'feature_importance.png'
        plt.savefig(importance_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"特征重要性图已保存: {importance_path}")

        # 保存特征重要性数值
        importance_dict = dict(zip(feature_cols_list, importance.tolist()))
        with open(result_dir / 'feature_importance.json', 'w', encoding='utf-8') as f:
            json.dump(importance_dict, f, ensure_ascii=False, indent=2)

    print(f"\n所有结果已保存到: {result_dir}")
