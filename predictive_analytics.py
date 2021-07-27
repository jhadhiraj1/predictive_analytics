# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 14:36:01 2021

@author: RUDRA
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

data=pd.read_csv("insurance.csv")

count_nan=data.isnull().sum()
print(count_nan[count_nan>0])

data['bmi'].fillna(data['bmi'].mean(),inplace=True)

count_nan=data.isnull().sum()

print(count_nan)
print(data['smoker'])


print(data)
sex=data.iloc[:,1:2].values
smoker=data.iloc[:,4:5].values

le=LabelEncoder()
sex[:,0]=le.fit_transform(sex[:,0])
sex=pd.DataFrame(sex)
sex.columns=['sex']
le_sex_mapping=dict(zip(le.classes_,le.transform(le.classes_)))
print(le_sex_mapping)

le=LabelEncoder()
smoker[:,0]=le.fit_transform(smoker[:,0])
smoker=pd.DataFrame(smoker)
smoker.coulumn=['smoker']
le_smoker_mapping=dict(zip(le.classes_,le.transform(le.classes_)))
print(le_smoker_mapping)
print(data['smoker'])

region=data.iloc[:,5:6].values
ohe=OneHotEncoder()
region=ohe.fit_transform(region).toarray()
region=pd.DataFrame(region)
region.columns=['northeast','northwest','southeast','southwest']
print(region[:10])

X_num=data[['age','bmi','children']]
X_final=pd.concat([X_num,sex,smoker,region],axis=1)

y_final=data[['expenses']].copy()
X_train,X_test,y_train,y_test=train_test_split(X_final,y_final,test_size=0.33,random_state=0)


##NOrmalization using MinMax
n_scaler=MinMaxScaler()
X_train=n_scaler.fit_transform(X_train.astype(np.float))
X_test=n_scaler.transform(X_test.astype(np.float))
##Normalization using Standardization

s_scaler=StandardScaler()
X_train=s_scaler.fit_transform(X_train.astype(np.float))

X_test=s_scaler.transform(X_test.astype(np.float))

lr=LinearRegression().fit(X_train,y_train)
y_train_pred=lr.predict(X_train)
y_test_pred=lr.predict(X_test)

print("lr co-efficient is {}".format(lr.coef_))
print("Intercep {}".format(lr.intercept_))
print("y_train Score: %.3f and y_test score: %.3f" % (lr.score(X_train,y_train),lr.score(X_test,y_test)))

##Applying Polynomial Features to the datas
poly_f=PolynomialFeatures(degree=2)
poly_X=poly_f.fit_transform(X_final)

X_train,X_test,y_train,y_test=train_test_split(poly_X,y_final,test_size=0.33,random_state=0)
    

s_scaler=StandardScaler()
X_train=s_scaler.fit_transform(X_train.astype(np.float))
X_test=s_scaler.transform(X_test.astype(np.float))



poly_lr=LinearRegression().fit(X_train,y_train)
poly_y_train_pred=poly_lr.predict(X_train)
poly_y_test_pred=poly_lr.predict(X_test)

print("Polynomoial lr Co-efficient:{}".format(poly_lr.coef_))
print("Y-intercept is :{}".format(poly_lr.intercept_))
print("y_train score: %.3f and y_test score:%.3f"
      % (poly_lr.score(X_train,y_train),poly_lr.score(X_test,y_test)))

## SVR Modelling

svr=SVR(kernel='linear',C=300)

X_train,X_test,y_train,y_test=train_test_split(X_final,y_final,test_size=0.33,random_state=0)

s_scaler=StandardScaler()
X_train=s_scaler.fit_transform(X_train.astype(np.float))
X_test=s_scaler.transform(X_test.astype(np.float))

svr=svr.fit(X_train,y_train.values.ravel())
y_train_pred=svr.predict(X_train)
y_test_pred=svr.predict(X_test)


print("y_train score: %.3f and y_test score: %.3f" %(svr.score(X_train,y_train),svr.score(X_test,y_test)))

dt=DecisionTreeRegressor(random_state=0);
dt=dt.fit(X_train,y_train.values.ravel())
y_train_pred=dt.predict(X_train)
y_test_pred=dt.predict(X_test)

print("y_train Score : %.3f and y_test score :%.3f" %(dt.score(X_train,y_train),dt.score(X_test,y_test)))

## Random Forest Regressor
rf=RandomForestRegressor(n_estimators=100,
                         criterion='mse',
                         random_state=1,
                         n_jobs=-1)
X_train,X_test,y_train,y_test=train_test_split(X_final,y_final,test_size=0.33,random_state=0)

n_scaler=StandardScaler()
X_train=n_scaler.fit_transform(X_train.astype(np.float))
X_test=n_scaler.transform(X_test.astype(np.float))


rf=rf.fit(X_train,y_train.values.ravel())
y_train_pred=rf.predict(X_train)
y_test_pred=rf.predict(X_test)

print("y_train score :%.3f and y_test score: %.3f"%(rf.score(X_train,y_train),rf.score(X_test,y_test)))