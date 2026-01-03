"""
传统机器学习模型：Ridge, Lasso, RF, GBDT, XGBoost, LightGBM, NGBoost
"""
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from ..config import LOG_OFFSET


def log_transform(y):
    """对数变换"""
    return np.log(y + LOG_OFFSET)


def inverse_log_transform(y_log):
    """逆对数变换"""
    return np.exp(y_log) - LOG_OFFSET


class RidgeModel:
    """Ridge回归模型"""
    def __init__(self, alpha=1.0, **kwargs):
        self.name = 'Ridge'
        self.model = Ridge(alpha=alpha)
        self.supports_uncertainty = False
        self.use_log_target = True

    def fit(self, X, y):
        y_log = log_transform(y)
        self.model.fit(X, y_log)

    def predict(self, X):
        y_log = self.model.predict(X)
        return inverse_log_transform(y_log)


class LassoModel:
    """Lasso回归模型"""
    def __init__(self, alpha=1.0, **kwargs):
        self.name = 'Lasso'
        self.model = Lasso(alpha=alpha, max_iter=10000)
        self.supports_uncertainty = False
        self.use_log_target = True

    def fit(self, X, y):
        y_log = log_transform(y)
        self.model.fit(X, y_log)

    def predict(self, X):
        y_log = self.model.predict(X)
        return inverse_log_transform(y_log)


class RandomForestModel:
    """随机森林模型"""
    def __init__(self, n_estimators=100, max_depth=10, **kwargs):
        self.name = 'RandomForest'
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42
        )
        self.supports_uncertainty = False
        self.use_log_target = True

    def fit(self, X, y):
        y_log = log_transform(y)
        self.model.fit(X, y_log)

    def predict(self, X):
        y_log = self.model.predict(X)
        return inverse_log_transform(y_log)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class GBDTModel:
    """GBDT梯度提升模型"""
    def __init__(self, n_estimators=100, max_depth=10, learning_rate=0.1, **kwargs):
        self.name = 'GBDT'
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )
        self.supports_uncertainty = False
        self.use_log_target = True

    def fit(self, X, y):
        y_log = log_transform(y)
        self.model.fit(X, y_log)

    def predict(self, X):
        y_log = self.model.predict(X)
        return inverse_log_transform(y_log)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


# 尝试导入XGBoost
try:
    import xgboost as xgb

    class XGBoostModel:
        """XGBoost模型"""
        def __init__(self, n_estimators=100, max_depth=10, learning_rate=0.1, **kwargs):
            self.name = 'XGBoost'
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_jobs=-1,
                random_state=42
            )
            self.supports_uncertainty = False
            self.use_log_target = True

        def fit(self, X, y):
            y_log = log_transform(y)
            self.model.fit(X, y_log)

        def predict(self, X):
            y_log = self.model.predict(X)
            return inverse_log_transform(y_log)

        @property
        def feature_importances_(self):
            return self.model.feature_importances_

except ImportError:
    print("[提示] XGBoost未安装，请使用命令安装: pip install xgboost")


# 尝试导入LightGBM
try:
    import lightgbm as lgb

    class LightGBMModel:
        """LightGBM模型"""
        def __init__(self, n_estimators=100, max_depth=10, learning_rate=0.1, **kwargs):
            self.name = 'LightGBM'
            self.model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_jobs=-1,
                random_state=42,
                verbose=-1
            )
            self.supports_uncertainty = False
            self.use_log_target = True

        def fit(self, X, y):
            y_log = log_transform(y)
            self.model.fit(X, y_log)

        def predict(self, X):
            y_log = self.model.predict(X)
            return inverse_log_transform(y_log)

        @property
        def feature_importances_(self):
            return self.model.feature_importances_

except ImportError:
    print("[提示] LightGBM未安装，请使用命令安装: pip install lightgbm")


# 尝试导入NGBoost
try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from ngboost.scores import LogScore

    class NGBoostModel:
        """NGBoost自然梯度提升模型

        支持概率预测，输出均值和方差
        使用对数尺度的NLL损失函数
        """
        def __init__(self, n_estimators=100, max_depth=10, learning_rate=0.1, **kwargs):
            self.name = 'NGBoost'
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.learning_rate = learning_rate
            self.model = None
            self.supports_uncertainty = True
            self.use_log_target = True

        def fit(self, X, y):
            from sklearn.tree import DecisionTreeRegressor

            # 使用对数变换的目标
            y_log = log_transform(y)

            # 创建基学习器
            base_learner = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=42
            )

            # 创建NGBoost模型
            self.model = NGBRegressor(
                Dist=Normal,
                Score=LogScore,
                Base=base_learner,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                verbose=False,
                random_state=42
            )

            self.model.fit(X, y_log)

        def predict(self, X):
            """预测均值（原始空间）"""
            y_log_pred = self.model.predict(X)
            return inverse_log_transform(y_log_pred)

        def predict_dist(self, X):
            """预测分布参数

            返回:
                mu: 预测均值（原始空间）
                sigma: 预测标准差（log空间）
            """
            dist = self.model.pred_dist(X)
            mu_log = dist.loc  # log空间的均值
            sigma_log = dist.scale  # log空间的标准差

            # 均值转回原始空间
            mu = inverse_log_transform(mu_log)

            return mu, sigma_log

        def compute_nll(self, X, y):
            """计算NLL损失"""
            y_log = log_transform(y)
            dist = self.model.pred_dist(X)
            nll = -dist.logpdf(y_log).mean()
            return nll

        @property
        def feature_importances_(self):
            return self.model.feature_importances_

except ImportError:
    print("[提示] NGBoost未安装，请使用命令安装: pip install ngboost")
