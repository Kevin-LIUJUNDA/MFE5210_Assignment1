from dimension_reduction import *
from build_model import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer


all_data = dim_reduction()
index_train = all_data[all_data['SalePrice'].notnull()].index
index_test = all_data[all_data['SalePrice'].isnull()].index

train_data = all_data.loc[index_train,:]  # 注意索引值对应关系
test_data = all_data.loc[index_test,:]

y_train = train_data["SalePrice"]
x_train = train_data.copy()
x_train.drop(["SalePrice"], axis=1, inplace=True)
x_test = test_data.copy()
x_test.drop(["SalePrice"], axis=1, inplace=True)

scaler = RobustScaler()
x_train = scaler.fit(x_train).transform(x_train)  #训练样本特征归一化
x_test = scaler.transform(x_test)                 #测试集样本特征归一化
y_train = y_train.values.reshape(-1,1)

# PCA
pca_model = PCA(n_components=375)
x_train = pca_model.fit_transform(x_train)
x_test = pca_model.transform(x_test)




## 指定每一个算法的参数
lasso = Lasso(alpha=0.0004,max_iter=10000)
ridge = Ridge(alpha=35)
svr = SVR(gamma= 0.0004,kernel='rbf',C=15,epsilon=0.009)
ker = KernelRidge(alpha=0.2 ,kernel='polynomial',degree=3 , coef0=1.2)
ela = ElasticNet(alpha=0.0005,l1_ratio=0.08,max_iter=10000)
bay = BayesianRidge()
xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,learning_rate=0.05, max_depth=3,
                   min_child_weight=1.7817, n_estimators=2200,reg_alpha=0.4640,
                   reg_lambda=0.8571,subsample=0.5213, silent=1,random_state =7, nthread = -1)
lgbm = LGBMRegressor(objective='regression',num_leaves=5,learning_rate=0.05, n_estimators=700,max_bin = 55,
                     bagging_fraction = 0.8,bagging_freq = 5, feature_fraction = 0.25,feature_fraction_seed=9,
                     bagging_seed=9,min_data_in_leaf = 6, min_sum_hessian_in_leaf = 11)

GBR = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =5)

x_train = SimpleImputer().fit_transform(x_train)
y_train = SimpleImputer().fit_transform(y_train.reshape(-1,1)).ravel()

## stack_model定义
stack_model = stacking(mod=[ela,svr,bay,lasso,ridge,ker], meta_model=ker)


x_train_stack, x_test_stack = stack_model.get_oof(x_train,y_train,x_test)
x_train_add = np.hstack((x_train,x_train_stack))
x_test_add = np.hstack((x_test,x_test_stack))
last_x_train_stack, last_x_test_stack = stack_model.get_oof(x_train_add,y_train,x_test_add)


ker = KernelRidge(alpha=0.2 ,kernel='polynomial',degree=3 , coef0=1.2)
my_model = ker.fit(last_x_train_stack, y_train)
y_pred_stack = np.expm1(my_model.predict(last_x_test_stack))
print(rmse_cv(my_model,last_x_train_stack,y_train).mean())

y_train_stack = my_model.predict(last_x_train_stack)       # 查看训练集的拟合误差
print(get_rmse(y_train, y_train_stack))

#y_pred = np.exp(stack_model.predict(x_test_add))
# ResultData=pd.DataFrame(np.hstack((test_id.values.reshape(-1,1),y_pred.reshape(-1,1))), index=range(len(y_pred)),
#                         columns=['Id', 'SalePrice'])
# ResultData['Id'] = ResultData['Id'].astype('int')
# # ResultData.to_csv("submission.csv",index=False)
# ResultData.to_csv('1st.csv',index=False)

# xgb
xgb.fit(last_x_train_stack, y_train)
y_pred_xgb = np.expm1(xgb.predict(last_x_test_stack))
print(rmse_cv(xgb,last_x_train_stack,y_train).mean())

y_train_xgb = xgb.predict(last_x_train_stack)
print(get_rmse(y_train, y_train_xgb))



# lightgbm
lgbm.fit(last_x_train_stack, y_train)
y_pred_lgbm = np.expm1(lgbm.predict(last_x_test_stack))
print(rmse_cv(lgbm,last_x_train_stack,y_train).mean())

y_train_lgbm = lgbm.predict(last_x_train_stack)
print(get_rmse(y_train, y_train_lgbm))

y_pred = y_pred_lgbm

ResultData=pd.DataFrame(np.hstack((test_id.values.reshape(-1,1),y_pred.reshape(-1,1))), index=range(len(y_pred)), \
                        columns=['Id', 'SalePrice'])
ResultData['Id'] = ResultData['Id'].astype('int')
ResultData.to_csv('lgbm.csv',index=False)




