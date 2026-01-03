
  消融实验设置

  | 消融类型            | 命令行参数               | 说明                                     |
  |---------------------|--------------------------|------------------------------------------|
  | w/o Cross-Attention | --ablation no_cross_attn | 用加权平均替代Cross-Attention            |
  | w/o 重复特征        | --ablation no_density    | 禁用MinHash时间密度特征                  |
  | w/o 上下文          | --ablation no_context    | 只使用评论文本，不使用微博/根评论/父评论 |
  | w/o Log NLL         | --ablation std_nll       | 使用原始空间标准NLL损失                  |

  使用方法

  # w/o 上下文消融实验（只使用评论文本本身）
  python src/main.py --model bge_nn --ablation no_context

  # 或使用单独参数
  python src/main.py --model bge_nn --no_context