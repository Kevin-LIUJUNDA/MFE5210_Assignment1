from feature_engineering import *

def split_data():
    all_data, train_id, test_id = feature_engineer()

    cols = list(all_data.columns)
    for col in cols:  # 可能特征工程的过程中会产生极个别异常值（正负无穷大），这里用众数填充
        all_data[col].values[np.isinf(all_data[col].values)] = all_data[col].median()
    del cols, col

    index_train = all_data[all_data['SalePrice'].notnull()].index
    index_test = all_data[all_data['SalePrice'].isnull()].index

    train_data = all_data.loc[index_train,:]  # 注意索引值对应关系
    test_data = all_data.loc[index_test,:]

    y_train = train_data["SalePrice"]
    x_train = train_data.copy()
    x_train.drop(["SalePrice"], axis=1, inplace=True)
    x_test = test_data.copy()
    x_test.drop(["SalePrice"], axis=1, inplace=True)

    # del train_data,test_data
    return y_train, x_train, x_test, all_data


def dim_reduction():
    y_train, x_train, x_test, all_data = split_data()
    # 归一化
    scaler = RobustScaler()
    x_train = scaler.fit(x_train).transform(x_train)  # 训练样本特征归一化
    x_test = scaler.transform(x_test)

    from sklearn.linear_model import Lasso  ##运用算法来进行训练集的得到特征的重要性，特征选择的一个作用是，wrapper基础模型
    lasso_model = Lasso(alpha=0.001)
    lasso_model.fit(x_train, y_train)


    ## 索引和重要性做成dataframe形式
    FI_lasso = pd.DataFrame({"Feature Importance": lasso_model.coef_},
                            all_data.drop(["SalePrice"],axis=1).columns)
    ## 由高到低进行排序
    FI_lasso.sort_values("Feature Importance", ascending=False,inplace=True)

    # ## 获取重要程度大于0的系数指标
    # FI_lasso[FI_lasso["Feature Importance"] != 0].sort_values("Feature Importance").plot(kind="barh", figsize=(12, 40),
    #                                                                                      color='g')
    # plt.xticks(rotation=90)
    # plt.show()  ##画图显示

    # FI_index = FI_lasso.index
    # FI_val = FI_lasso["Feature Importance"].values
    # FI_lasso = pd.DataFrame(FI_val, columns=['Feature Importance'], index=FI_index)
    #print(FI_lasso)


    # can select features here, [:200] select top 200 important features
    choose_cols = FI_lasso.index.tolist()
    choose_cols.append("SalePrice")
    choose_data = all_data[choose_cols].copy()

    return choose_data






dim_reduction()



