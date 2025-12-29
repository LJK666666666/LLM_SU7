"""
配置文件：定义项目路径、特征列、常量等
"""
from pathlib import Path

# ==================== 路径配置 ====================
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR
RESULT_DIR = ROOT_DIR / 'results'
BGE_MODEL_PATH = ROOT_DIR / 'bge-base-zh-v1.5'

# ==================== 数据配置 ====================
TRAIN_FILE = 'train.csv'
VAL_FILE = 'val.csv'
TEST_FILE = 'test.csv'
TRAIN_PKL = 'train.pkl'
VAL_PKL = 'val.pkl'
TEST_PKL = 'test.pkl'

# ==================== 目标变量 ====================
TARGET_COL = '子评论数'
LOG_OFFSET = 10  # log变换偏移量

# ==================== 特征定义 ====================
# 基础特征（来自原始数据）
BASE_FEATURE_COLS = [
    '用户总评论数',      # 用户历史评论数（log变换后）
    '用户是否认证',      # 是否认证用户
    '是否一级评论',      # 是否为一级评论
    '微博评论数',        # 所属微博的评论数
    '发布小时',          # 发布时间的小时
    '发布星期',          # 发布时间的星期几
    '是否工作日',        # 是否为工作日
]

# 文本特征（从评论文案提取）
TEXT_FEATURE_COLS = [
    '评论长度',          # 评论字符数
    '感叹号数',          # 感叹号数量
    '问号数',            # 问号数量
    '表情数',            # 表情符号数量
    '话题标签有无',      # 是否包含#话题#
    '小米相关词数',      # 小米相关关键词出现次数
]

# LDA主题特征
LDA_FEATURE_COLS = [
    '主题',              # LDA主题编号
]

# 时间密度特征（来自compute_time_density.py）
DENSITY_FEATURE_COLS = [
    '时间顺序索引',      # 评论在微博下的时间顺序
    '最大相似度',        # 与历史评论的最大相似度
    '重复次数',          # 相似评论出现次数
]

# 特征组映射
FEATURE_GROUPS = {
    'base': BASE_FEATURE_COLS,
    'text': TEXT_FEATURE_COLS,
    'lda': LDA_FEATURE_COLS,
    'density': DENSITY_FEATURE_COLS,
}

# ==================== BGE神经网络模型配置 ====================
# VIP用户白名单（来自vip_users.txt，出现>=20次的用户）
VIP_USERS = [
    '小米法务部',        # 328次
    '雷军',              # 292次
    '小米汽车',          # 69次
    '王化',              # 66次
    '鸿蒙智行法务',      # 49次
    '薛定谔的英短咕咕咕', # 33次
    '余承东',            # 32次
    '小米公司发言人',    # 29次
    '小蒜苗长',          # 29次
    '万能的大熊',        # 27次
    '我是大彬同学',      # 27次
    '科技新一',          # 26次
    '美国驻华大使馆',    # 26次
    'AI逃逸',            # 24次
    '小米公司',          # 24次
    '羊驼的睡衣',        # 24次
    '卢伟冰',            # 22次
    '诗雨370491153',     # 20次
    '不会武功的武功李云飞',  # 20次
]

# VIP用户到ID的映射
VIP_USER_TO_ID = {user: i for i, user in enumerate(VIP_USERS)}

# 特殊token
SPECIAL_TOKEN_USER = '_USER_'

# ==================== 特殊嵌入配置 ====================
# 特殊token类型（用于可训练嵌入）

# 1. VIP用户嵌入（每个VIP用户一个独立嵌入）
# VIP_USERS 已在上面定义，共19个用户

# 2. 微博方括号表情符号嵌入（[xxx]格式）
WEIBO_EMOJI_LIST = [
    'doge', '哈哈', '笑cry', '允悲', '二哈', '吃瓜', '微笑', '跪了',
    '赞', '心', '爱你', '抱抱', '泪', '怒', '吐', '污',
    '挖鼻', '思考', '疑问', '费解', '黑线', '汗', '拜拜', 'good',
    '酷', '鼓掌', '可怜', '失望', '悲伤', '委屈', '生病', '抓狂',
    '笑哭', '捂脸', '偷笑', '坏笑', '嘻嘻', '哼', '怒骂', '打脸',
]
WEIBO_EMOJI_TO_ID = {emoji: i for i, emoji in enumerate(WEIBO_EMOJI_LIST)}

# 3. Unicode Emoji表情（真实emoji字符，如🌿😄等）
UNICODE_EMOJI_LIST = [
    '😂', '😭', '🤣', '😍', '😊', '🥺', '😢', '😡',
    '👍', '👎', '👏', '🙏', '💪', '❤️', '💔', '🔥',
    '✨', '🌟', '⭐', '🎉', '🎊', '💯', '🆘', '⚠️',
    '🚗', '🚙', '🏎️', '🔋', '⚡', '💨', '🌿', '🍃',
    '😏', '🤔', '😅', '😆', '😎', '🤡', '💀', '🤮',
]
UNICODE_EMOJI_TO_ID = {emoji: i for i, emoji in enumerate(UNICODE_EMOJI_LIST)}

# 4. 小米相关关键词嵌入（来自xiaomi_word.txt）
XIAOMI_EMBED_KEYWORDS = [
    # 品牌与产品
    '小米汽车', '小米SU7', 'SU7', 'su7', '雷军',
    # 竞品
    '保时捷', 'Taycan', '比亚迪', '特斯拉', 'Model3',
    '蔚来', '小鹏', '理想', '问界', '华为',
    # 智能驾驶
    '智能驾驶', '自动驾驶', '辅助驾驶', '智驾',
    # 续航与充电
    '续航', '电池', '充电', '快充', '超充',
    # 座舱与系统
    '智能座舱', '车机', '大屏', '澎湃OS',
    # 购买相关
    '性价比', '质价比', '定价', '预售', '交付', '锁单', '大定', '小定',
    # 热门词汇
    '遥遥领先', '真香', '割韭菜', '智商税',
]
XIAOMI_KEYWORD_TO_ID = {kw: i for i, kw in enumerate(XIAOMI_EMBED_KEYWORDS)}

# 特殊嵌入总数
NUM_VIP_EMBEDDINGS = len(VIP_USERS)                    # 19
NUM_WEIBO_EMOJI_EMBEDDINGS = len(WEIBO_EMOJI_LIST)     # 40
NUM_UNICODE_EMOJI_EMBEDDINGS = len(UNICODE_EMOJI_LIST) # 40
NUM_XIAOMI_EMBEDDINGS = len(XIAOMI_EMBED_KEYWORDS)     # 39
NUM_OTHER_EMBEDDINGS = 1  # [USER] 非VIP用户统一嵌入

TOTAL_SPECIAL_EMBEDDINGS = (
    NUM_VIP_EMBEDDINGS +
    NUM_WEIBO_EMOJI_EMBEDDINGS +
    NUM_UNICODE_EMOJI_EMBEDDINGS +
    NUM_XIAOMI_EMBEDDINGS +
    NUM_OTHER_EMBEDDINGS
)  # 139

# 特殊嵌入ID偏移量
VIP_EMBED_OFFSET = 0
WEIBO_EMOJI_EMBED_OFFSET = NUM_VIP_EMBEDDINGS
UNICODE_EMOJI_EMBED_OFFSET = WEIBO_EMOJI_EMBED_OFFSET + NUM_WEIBO_EMOJI_EMBEDDINGS
XIAOMI_EMBED_OFFSET = UNICODE_EMOJI_EMBED_OFFSET + NUM_UNICODE_EMOJI_EMBEDDINGS
OTHER_EMBED_OFFSET = XIAOMI_EMBED_OFFSET + NUM_XIAOMI_EMBEDDINGS
USER_EMBED_ID = OTHER_EMBED_OFFSET  # 非VIP用户的嵌入ID

# 小米相关词汇（用于特征提取，更广泛的列表）
XIAOMI_KEYWORDS = [
    '小米', 'xiaomi', 'SU7', 'su7', '雷军', '卢伟冰',
    '智能驾驶', '智驾', '自动驾驶', '辅助驾驶',
    '电动车', '电车', '新能源', '纯电',
    '续航', '充电', '快充', '超充',
    '电池', '三元锂', '磷酸铁锂',
    '底盘', '悬架', '空气悬架',
    '性能', '加速', '百公里', '零百',
    '内饰', '中控', '大屏', '车机',
    '外观', '设计', '颜值',
]
