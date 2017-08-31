#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 21:05:48 2017

@author: zhuoqinyu
"""
#Kaggle link: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
df_train=pd.read_csv("train.csv")
df_test=pd.read_csv("test.csv")
testid=df_test.Id
#all_data = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'],
#                      df_test.loc[:,'MSSubClass':'SaleCondition']))
print(df_train.columns)
print(df_train['SalePrice'].describe())

## correlations of numerical variables & SalePrice
#for var in num_vars:
#    df_train.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
##Relationship with categorical features
#var='OverallQual'
#f,ax=plt.subplots(figsize=(8,6))
#fig=sns.boxplot(x=var,y="SalePrice",data=df_train)
#fig.axis(ymin=0,ymax=800000)
#var = 'YearBuilt'
#f,ax=plt.subplots(figsize=(8,6))
#fig=sns.boxplot(x=var,y="SalePrice",data=df_train)
#fig.axis(ymin=0,ymax=800000)
### correlation heatmap
#plt.figure(figsize=(45,43))
#foo=sns.heatmap(df_train.corr(),vmax=0.8,square=True)
#plt.xticks(rotation=60)
#plt.yticks(rotation=30)
#plt.tight_layout()

## zoomed heatmap
#plt.figure(figsize=(45,43))
#k=10
#cols=df_train.corr().nlargest(k,'SalePrice')['SalePrice'].index
#cm=np.corrcoef(df_train[cols].values.T)
#sns.set(font_scale=1.25)
#hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
#plt.xticks(rotation=60)
#plt.yticks(rotation=30)
#plt.tight_layout()
#plt.show()

## set of scatterplots
#sns.set()
#cols=['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
#sns.pairplot(df_train[cols],size=2.5)
#plt.show()

## missing data
total=df_train.isnull().sum().sort_values(ascending=False)
percent=(df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([total,percent],axis=1,keys=['Total', 'Percent'])
#               Total   Percent
#PoolQC          1453  0.995205
#MiscFeature     1406  0.963014
#Alley           1369  0.937671
#Fence           1179  0.807534
#FireplaceQu      690  0.472603
#LotFrontage      259  0.177397
#GarageCond        81  0.055479
#GarageType        81  0.055479
#GarageYrBlt       81  0.055479
#GarageFinish      81  0.055479
#GarageQual        81  0.055479
#BsmtExposure      38  0.026027
#BsmtFinType2      38  0.026027
#BsmtFinType1      37  0.025342
#BsmtCond          37  0.025342
#BsmtQual          37  0.025342
#MasVnrArea         8  0.005479
#MasVnrType         8  0.005479
#Electrical         1  0.000685
#Delete all the variables with missing data, except the variable 'Electrical'. In Electrical just 
#delete the observation with missing data.
df_train=df_train.drop((missing_data[missing_data['Total']>1]).index,1)
df_test=df_test.drop((missing_data[missing_data['Total']>1]).index,1)
df_train=df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max()#checking there is no missing data
##Univariate analysis Standardizing data
## Standardize
#scaler=StandardScaler()
#df_train['SalePrice'] = scaler.fit_transform(df_train['SalePrice'][:,np.newaxis])
## log transformation
df_train["SalePrice"] = np.log1p(df_train["SalePrice"])
#df_test['SalePrice']=StandardScaler().transform(df_test['SalePrice'][:,np.newaxis])#test set doesn't have target values
#low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
#high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
##Bivariate analysis
#var = 'GrLivArea'
#df_train.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
#df_train.sort_values(by=var,ascending=False)

df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

# Check normality 
#histogram and normal probability plot
#sns.distplot(df_train['SalePrice'], fit=norm)
#fig = plt.figure()
#res = stats.probplot(df_train['SalePrice'], plot=plt)
#print("Skewness: %f" % df_train['SalePrice'].skew())
#print("Kurtosis: %f" % df_train['SalePrice'].kurt())

##applying log transformation
#df_train['SalePrice'] = np.log(df_train['SalePrice'])

##transformed histogram and normal probability plot
#sns.distplot(df_train['SalePrice'], fit=norm);
#fig = plt.figure()
#res = stats.probplot(df_train['SalePrice'], plot=plt)

## for GrLivArea
#sns.distplot(df_train['GrLivArea'], fit=norm);
#fig = plt.figure()
#res = stats.probplot(df_train['GrLivArea'], plot=plt)
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
df_test['GrLivArea'] = np.log(df_test['GrLivArea'])
#sns.distplot(df_train['GrLivArea'], fit=norm)
#fig = plt.figure()
#res = stats.probplot(df_train['GrLivArea'], plot=plt)
#plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])# after log transformation the distribution is normal

###log transform skewed numeric features:
#numeric_feats = df_train.dtypes[df_train.dtypes != "object"].index
#skewed_feats = df_train[numeric_feats].apply(lambda x: x.dropna().skew()) #compute skewness
#skewed_feats = skewed_feats[skewed_feats > 0.75]
#skewed_feats = skewed_feats.index.drop('SalePrice')
#df_train[skewed_feats] = np.log(df_train[skewed_feats])
#df_test[skewed_feats]=np.log(df_test[skewed_feats])

##histogram and normal probability plot
#sns.distplot(df_train['TotalBsmtSF'], fit=norm);
#fig = plt.figure()
#res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)

##create column for new variable (one is enough because it's a binary categorical feature)
##if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
df_test['HasBsmt'] = pd.Series(len(df_test['TotalBsmtSF']), index=df_test.index)
df_test['HasBsmt'] = 0 
df_test.loc[df_test['TotalBsmtSF']>0,'HasBsmt'] = 1
##transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
df_test.loc[df_test['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_test['TotalBsmtSF'])

##convert categorical variable into dummy
target=df_train.SalePrice
all_data = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'],
                      df_test.loc[:,'MSSubClass':'SaleCondition']))
all_data=pd.get_dummies(all_data)
df_train = all_data[:df_train.shape[0]]
df_test = all_data[df_train.shape[0]:]

## filling NA with the mean of the column:
df_train = df_train.fillna(df_train.mean())
df_test=df_test.fillna(df_test.mean())

#creating matrices for sklearn:
X_train = df_train
X_test = df_test
y =target
#==============================================================================
# Models regulized linear regression
#==============================================================================
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse=np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return rmse
##I2_Ridge
#model_ridge=Ridge()
#alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
#cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
#cv_ridge=pd.Series(cv_ridge,index=alphas)
#cv_ridge.plot(title = "Validation")
#plt.xlabel("alpha")
#plt.ylabel("rmse")
#cv_ridge.min()#0.11334765604051882

## Lasso 
model_lasso=LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
rmse_cv(model_lasso).mean() #0.11205709530490301
coef = pd.Series(model_lasso.coef_, index = X_train.columns) # Lasso will select features, unimportant feature has coef of zero
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
#Lasso picked 88 variables and eliminated the other 131 variables
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
plt.figure(figsize=(8,10))
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.tight_layout()

preds=pd.DataFrame({"predictions":model_lasso.predict(X_train),"target":y})
preds["residuals"]=preds["target"]-preds["predictions"]
preds.plot(x="predictions",y="target",kind="scatter")

###xgboost model
import xgboost as xgb
dtrain=xgb.DMatrix(X_train,label=y)
dtest=xgb.DMatrix(X_test)
params={"max_depth":2,"eta":0.1}
#model=xgb.cv(params,dtrain,num_boost_round=500, early_stopping_rounds=100)
#model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)
model_xgb.fit(X_train,y)
xgb_preds=model_xgb.predict(X_test)
lasso_preds=model_lasso.predict(X_test)
predictions=pd.DataFrame({"xgb":xgb_preds,"lasso":lasso_preds})
predictions.plot(x="xgb",y="lasso",kind="scatter")
# a weighted average of uncorrelated results for test set prediction
#preds=0.3*lasso_preds+0.7*xgb_preds
preds=np.expm1(xgb_preds)
solution=pd.DataFrame({"id":testid,"SalePrice":preds})
solution.to_csv("ridge_sol.csv",index=False)
