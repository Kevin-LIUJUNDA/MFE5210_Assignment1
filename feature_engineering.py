from data_cleansing import *
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder                    #标签编码
from sklearn.preprocessing import RobustScaler, StandardScaler    #去除异常值与数据标准化
from sklearn.pipeline import Pipeline, make_pipeline              #构建管道
from scipy.stats import skew,norm                                 #偏度
from scipy.special import boxcox1p                           # box-cox变换
from sklearn.decomposition import PCA



def custom_coding(x):
    if(x=='Ex'): r = 0
    elif(x=='Gd'): r = 1
    elif(x=='TA'): r = 2
    elif(x=='Fa'): r = 3
    elif(x=='None'): r = 4
    else: r = 5
    return r

# 顺序变量  手动赋予数字顺序
def ordinal_feature(all_data):
    cols = ['BsmtCond', 'BsmtQual', 'ExterCond', 'ExterQual', 'FireplaceQu', 'GarageCond', 'GarageQual', 'HeatingQC',
            'KitchenQual', 'PoolQC']
    for col in cols:
        all_data[col] = all_data[col].apply(custom_coding)

# 类别型  转化为str，然后LabelEncode
def to_str(all_data):
    cols = ['MSSubClass', 'YrSold', 'MoSold', 'OverallCond', "MSZoning", "BsmtFullBath", "BsmtHalfBath", "HalfBath",
            "Functional", "Electrical", "KitchenQual", "KitchenAbvGr", "SaleType", "Exterior1st", "Exterior2nd",
            "YearBuilt",  "YearRemodAdd", "GarageYrBlt", "BedroomAbvGr", "LowQualFinSF"]
    for col in cols:
        all_data[col] = all_data[col].astype(str)

# categorical  LabelEncoder
def categorical(all_data):
    ## 年份等特征的标签编码
    str_cols = ["YearBuilt", "YearRemodAdd", 'GarageYrBlt', "YrSold", 'MoSold']
    for col in str_cols:
        all_data[col] = LabelEncoder().fit_transform(all_data[col])

    lab_cols = ['Heating', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish',
                'LandSlope', 'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 'YrSold',
                'MoSold', 'MSZoning', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'Exterior1st',
                'MasVnrType', 'Foundation', 'GarageType', 'SaleType', 'SaleCondition']

    for col in lab_cols:
        new_col = "labfit_" + col
        #new_col = col
        all_data[new_col] = LabelEncoder().fit_transform(all_data[col])

# 构建新特征
def add_features(all_data):
    all_data['TotalHouseArea'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    all_data['YearsSinceRemodel'] = all_data['YrSold'].astype(int) - all_data['YearRemodAdd'].astype(int)
    all_data['Total_Home_Quality'] = all_data['OverallQual'].astype(int) + all_data['OverallCond'].astype(int)

    all_data['HasWoodDeck'] = (all_data['WoodDeckSF'] == 0) * 1
    all_data['HasOpenPorch'] = (all_data['OpenPorchSF'] == 0) * 1
    all_data['HasEnclosedPorch'] = (all_data['EnclosedPorch'] == 0) * 1
    all_data['Has3SsnPorch'] = (all_data['3SsnPorch'] == 0) * 1
    all_data['HasScreenPorch'] = (all_data['ScreenPorch'] == 0) * 1

    all_data["TotalAllArea"] = all_data["TotalHouseArea"] + all_data["GarageArea"]  # 房屋总面积加车库面积
    all_data["TotalHouse_and_OverallQual"] = all_data["TotalHouseArea"] * all_data["OverallQual"]  # 房屋总面积和房屋材质指标乘积
    all_data["GrLivArea_and_OverallQual"] = all_data["GrLivArea"] * all_data["OverallQual"]  # 地面上居住总面积和房屋材质指标乘积
    all_data["LotArea_and_OverallQual"] = all_data["LotArea"] * all_data["OverallQual"]  # 地段总面积和房屋材质指标乘积
    all_data["MSZoning_and_TotalHouse"] = all_data["labfit_MSZoning"] * all_data["TotalHouseArea"]  # 一般区域分类与房屋总面积的乘积
    all_data["MSZoning_and_OverallQual"] = all_data["labfit_MSZoning"] + all_data["OverallQual"]  # 一般区域分类指标与房屋材质指标之和
    all_data["MSZoning_and_YearBuilt"] = all_data["labfit_MSZoning"] + all_data["YearBuilt"]  # 一般区域分类指标与初始建设年份之和
    ## 地理邻近环境位置指标与总房屋面积之积
    all_data["Neighborhood_and_TotalHouse"] = all_data["labfit_Neighborhood"] * all_data["TotalHouseArea"]
    all_data["Neighborhood_and_OverallQual"] = all_data["labfit_Neighborhood"] + all_data["OverallQual"]
    all_data["Neighborhood_and_YearBuilt"] = all_data["labfit_Neighborhood"] + all_data["YearBuilt"]
    all_data["BsmtFinSF1_and_OverallQual"] = all_data["BsmtFinSF1"] * all_data["OverallQual"]  # 1型成品的面积和房屋材质指标乘积
    ## 家庭功能评级指标与房屋总面积的乘积
    all_data["Functional_and_TotalHouse"] = all_data["labfit_Functional"] * all_data["TotalHouseArea"]
    all_data["Functional_and_OverallQual"] = all_data["labfit_Functional"] + all_data["OverallQual"]
    all_data["TotalHouse_and_LotArea"] = all_data["TotalHouseArea"] + all_data["LotArea"]
    ## 房屋与靠近公路或铁路指标乘积系数
    all_data["Condition1_and_TotalHouse"] = all_data["labfit_Condition1"] * all_data["TotalHouseArea"]
    all_data["Condition1_and_OverallQual"] = all_data["labfit_Condition1"] + all_data["OverallQual"]
    all_data["Bsmt"] = all_data["BsmtFinSF1"] + all_data["BsmtFinSF2"] + all_data["BsmtUnfSF"]  # 地下室相关面积总和指标
    all_data["Rooms"] = all_data["FullBath"] + all_data["TotRmsAbvGrd"]  # 地面上全浴室和地面上房间总数量之和
    ## 开放式门廊、围廊、三季门廊、屏风玄关总面积
    all_data["PorchArea"] = all_data["OpenPorchSF"] + all_data["EnclosedPorch"] + \
                            all_data["3SsnPorch"] + all_data["ScreenPorch"]
    ## 全部功能区总面积（房屋、地下室、车库、门廊等）
    all_data["TotalPlace"] = all_data["TotalAllArea"] + all_data["PorchArea"]

# 改变数据分布
def change_distribution(all_data):
    num_features = all_data.select_dtypes(include=['int64', 'float64', 'int32']).copy()
    num_features.drop(['SalePrice'], axis=1, inplace=True)  # 去掉目标值房价列

    num_feature_names = list(num_features.columns)

    num_features_data = pd.melt(all_data, value_vars=num_feature_names)

    skewed_feats = all_data[num_feature_names].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewed_feats})
    #print(skewness[skewness["Skew"].abs() > 0.75])

    skew_cols = list(skewness[skewness["Skew"].abs() > 1].index)
    for col in skew_cols:
        # all_data[col] = boxcox1p(all_data[col], 0.15)                                  # 偏度超过阈值的特征做box-cox变换
        all_data[col] = np.log1p(all_data[col])



def feature_engineer():
    all_data, train_id, test_id = data_cleansing()
    #print(all_data['SalePrice'])

    ordinal_feature(all_data)
    to_str(all_data)
    categorical(all_data)
    add_features(all_data)
    change_distribution(all_data)

    all_data = pd.get_dummies(all_data)

    return all_data, train_id, test_id


all_data = feature_engineer()



