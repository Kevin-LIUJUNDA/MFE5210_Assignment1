import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm


# read the raw data
def get_raw_data():
    df_train = pd.read_csv('./data/train.csv')
    df_test = pd.read_csv('./data/test.csv')
    return df_train, df_test

# delete outliers of train data
def delete_outlier(df_train):
    ## 删除异常值
    df_train.drop(df_train[(df_train['OverallQual'] < 5) & (df_train['SalePrice'] > 200000)].index, inplace=True)
    df_train.drop(df_train[(df_train['YearBuilt'] < 1900) & (df_train['SalePrice'] > 400000)].index, inplace=True)
    df_train.drop(df_train[(df_train['YearBuilt'] > 1980) & (df_train['SalePrice'] > 700000)].index, inplace=True)
    df_train.drop(df_train[(df_train['TotalBsmtSF'] > 6000) & (df_train['SalePrice'] < 200000)].index, inplace=True)
    df_train.drop(df_train[(df_train['GrLivArea'] > 4000) & (df_train['SalePrice'] < 200000)].index, inplace=True)

    ## 重置索引，使得索引值连续
    df_train.reset_index(drop=True, inplace=True)


# id is not relevant, delete it and store it
def delete_and_store_id(df_train,df_test):
    train_id = df_train['Id']
    test_id = df_test['Id']
    df_train.drop(['Id'],axis=1,inplace=True)
    df_test.drop(['Id'], axis=1, inplace=True)
    return train_id, test_id

# transform_normal
def transform_normal(df_train):
    df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
    #print(df_train['SalePrice'])

# merge train and test data
def merge(df_train,df_test):
    all_data = pd.concat([df_train,df_test],axis=0)
    all_data.reset_index(drop=True,inplace=True)
    return all_data

# fill null value
def fill_null(all_data):
    # char feature
    str_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",
                "GarageCond", \
                "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType", "MSSubClass"]
    for col in str_cols:
        all_data[col].fillna("None", inplace=True)

    # num feature (features that some house may not have)
    num_cols = ["BsmtUnfSF", "TotalBsmtSF", "BsmtFinSF2", "BsmtFinSF1", "BsmtFullBath", "BsmtHalfBath", \
                "MasVnrArea", "GarageCars", "GarageArea", "GarageYrBlt"]
    for col in num_cols:
        all_data[col].fillna(0, inplace=True)

    # using mode to fill (some common feature that every house should have)
    other_cols = ["MSZoning", "Electrical", "KitchenQual", "Exterior1st", "Exterior2nd", "SaleType"]
    for col in other_cols:
        all_data[col].fillna(all_data[col].mode()[0], inplace=True)

    # neighbor median
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

    all_data = all_data.drop(["Utilities"], axis=1)

    all_data["Functional"] = all_data["Functional"].fillna("Typ")

    return all_data


def data_cleansing():
    df_train, df_test = get_raw_data()
    delete_outlier(df_train)
    train_id, test_id = delete_and_store_id(df_train,df_test)
    transform_normal(df_train)
    all_data = merge(df_train,df_test)
    all_data = fill_null(all_data)

    return all_data,train_id,test_id




all_data,train_id,test_id = data_cleansing()

#
# def plt_distribution(data, obj_col):
#     sns.distplot(data[obj_col] , fit=norm);
#
#     # 获取训练集数据分布曲线的拟合均值和标准差
#     (mu, sigma) = norm.fit(data[obj_col])
#     print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#
#     # 绘制分布曲线
#     plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#                 loc='best')
#     plt.ylabel('Frequency')
#     plt.title('SalePrice distribution')
#
#     # 绘制图像查看数据的分布状态
#     fig = plt.figure()
#     res = stats.probplot(data[obj_col], plot=plt)
#     plt.show()
# plt_distribution(df_train, 'SalePrice')


# count=all_data.isnull().sum().sort_values(ascending=False)
# ratio=count/len(all_data)
# nulldata=pd.concat([count,ratio],axis=1,keys=['count','ratio'])
# del count, ratio
# print(nulldata[nulldata.ratio>0] )
