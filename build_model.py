from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
import numpy as np

import math
def get_mse(records_real, records_predict):
    ## 均方误差 估计值与真值 偏差
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None

def get_rmse(records_real, records_predict):
    ## 均方根误差：是均方误差的算术平方根
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None

#定义交叉验证的策略，以及评估函数
def rmse_cv(model,X,y):
    ## 针对各折数据集的测试结果的均方根误差
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))   # cv 代表数据划分的KFold折数
    return rmse





class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    ##=============== 参数说明 ================##
    # mod --- 堆叠过程的第一层中的算法
    # meta_model --- 堆叠过程的第二层中的算法，也称次学习器

    def __init__(self, mod, meta_model):
        self.mod = mod  # 首层学习器模型
        self.meta_model = meta_model  # 次学习器模型
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)  # 这就是堆叠的最大特征进行了几折的划分

    ## 训练函数
    def fit(self, X, y):
        self.saved_model = [list() for i in self.mod]  # self.saved_model包含所有第一层学习器
        oof_train = np.zeros((X.shape[0], len(self.mod)))  # 维度：训练样本数量*模型数量，训练集的首层预测值

        for i, model in enumerate(self.mod):  # 返回的是索引和模型本身
            for train_index, val_index in self.kf.split(X, y):  # 返回的是数据分割成分（训练集和验证集对应元素）的索引
                renew_model = clone(model)  # 模型的复制
                renew_model.fit(X[train_index], y[train_index])  # 对分割出来的训练集数据进行训练
                self.saved_model[i].append(renew_model)  # 把模型添加进去
                # oof_train[val_index,i] = renew_model.predict(X[val_index]).reshape(-1,1) #用来预测验证集数据

                val_prediction = renew_model.predict(X[val_index]).reshape(-1, 1)  # 验证集的预测结果，注：结果是没有索引的

                for temp_index in range(val_prediction.shape[0]):
                    oof_train[val_index[temp_index], i] = val_prediction[temp_index]  # 用来预测验证集数据的目标值

        self.meta_model.fit(oof_train, y)  # 次学习器模型训练，这里只是用到了首层预测值作为特征
        return self
    ## 预测函数
    def predict(self,X):
        temp = []
        # for single_model in self.saved_model:
        #     singles = []
        #     for model in single_model:
        #         single_test = np.array(model.predict(X))
        #         singles.append(single_test)
        #     part_test = np.column_stack(singles).mean(axis=1)
        #     temp.append(part_test)
        # whole_test = np.column_stack(temp)
        whole_test = np.column_stack([np.column_stack([model.predict(X) for model in single_model]).mean(axis=1)
                                      for single_model in self.saved_model])        #得到的是整个测试集的首层预测值

        return self.meta_model.predict(whole_test)

    ## 获取首层学习结果的堆叠特征
    def get_oof(self, X, y, test_X):
        oof = np.zeros((X.shape[0], len(self.mod)))  # 初始化为0
        test_single = np.zeros((test_X.shape[0], 5))  # 初始化为0
        # display(test_single.shape)
        test_mean = np.zeros((test_X.shape[0], len(self.mod)))
        for i, model in enumerate(self.mod):  # i是模型
            for j, (train_index, val_index) in enumerate(self.kf.split(X, y)):  # j是所有划分好的的数据
                clone_model = clone(model)  # 克隆模块，相当于把模型复制一下
                clone_model.fit(X[train_index], y[train_index])  # 把分割好的数据进行训练

                val_prediction = clone_model.predict(X[val_index]).reshape(-1, 1)  # 验证集的预测结果，注：结果是没有索引的
                for temp_index in range(val_prediction.shape[0]):
                    oof[val_index[temp_index], i] = val_prediction[temp_index]  # 用来预测验证集数据

                # oof[val_index,i] = clone_model.predict(X[val_index]).reshape(-1,1)    #对验证集进行预测
                # test_single[:,j] = clone_model.predict(test_X).reshape(-1,1)           #对测试集进行预测

                test_prediction = clone_model.predict(test_X).reshape(-1, 1)  # 对测试集进行预测

                # display(test_prediction.shape)
                test_single[:, j] = test_prediction[:, 0]
            test_mean[:, i] = test_single.mean(axis=1)  # 测试集算好均值
        return oof, test_mean






