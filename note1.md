• 我检查了 train.pkl/val.pkl/test.pkl（共 271,452 行）。按你说的条件：

  - 条件：子评论数 > 10 且 子评论数 / 用户总评论数 > 0.8
  - 排除用户总评论数 = 0 的情况（避免除零）：
      - 总数：951
      - train：745
      - val：97
      - test：109

  如果不排除用户总评论数为 0 的行（会导致比值为无穷大，从而被统计进去），总数是 1400（train 1103 / val 139 / test 158）。

  你希望哪种口径作为最终答案？如果你只关心训练集或只关心某个 split，也可以告诉我。



• 默认跑 python src/main.py --model bge_nn 时用到的特征/输入如下（BGE 模型不走 --features 那套手工特征组）：                                                                 
                                                                                                                                                                             
  - 文本输入（四路，默认启用上下文）：评论文本、微博文本、根评论文本、父评论文本，都会做 BGE tokenizer 编码。src/data/dataset.py                                             
  - 特殊 token 嵌入（默认启用）：@用户（VIP/普通）、微博方括号表情、Unicode emoji、小米关键词等，从四路文本抽取 ID。src/models/bge_nn.py、src/data/dataset.py                
  - 数值特征（基础）：                                                                                                                                                       
    用户总评论数(log1p)、用户是否认证、是否一级评论、发布时间小时(sin/cos)、发布时间星期(sin/cos)。src/data/dataset.py                                                       
  - 时间密度特征（默认启用）：                                                                                                                                               
    时间顺序索引(标准化)、最大相似度、重复次数(log1p)。src/data/dataset.py                                                                                                   
                                                                                                                                                                             
  不包含的：LDA 主题特征、手工文本特征（长度/标点/emoji计数等）、是否工作日、微博评论数 这些都属于传统模型的特征组，不被 bge_nn 使用。src/data/features.py、src/config.py    
                                                                                                                                                                             
  如果你想确认某个字段是否实际参与 bge_nn，我可以按字段逐个对照 CommentPredictorNN 的输入拼接逻辑再帮你核。     