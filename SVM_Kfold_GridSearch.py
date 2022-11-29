from pyexpat import model
import numpy as np
import pandas as pd
from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import metrics

df = pd.read_excel(r"E:\Fudan_Luoqiang_MDDProject\848MDD_794NC\数据分析\数据整理统计分析\健康人中的分类模型\Global_Dosen_NetworkClassification.xlsx")
#print(df)
X = df.iloc[:,2:8]
#print(X)
Y = df.iloc[:,8]
#print(Y)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
#print(X_std)
#X_train, X_test, Y_train, Y_test = train_test_split(X_std, Y, test_size= 0.3)
parameters = {'kernel':['rbf'],'C':[.001,0.01,0.1,1,10,100,1000],'gamma':[.001,0.01,0.1,1,10,100,1000]}
#model = SVC()
#参考 https://www.cnblogs.com/ysugyl/p/8711205.html
Grid_model = GridSearchCV(SVC(), parameters, cv = 10) #实例化一个GridSearchCV类
X_train, X_test, Y_train, Y_test = train_test_split(X_std, Y) 
Grid_model.fit(X_train, Y_train) #训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator
print("Test set score:{:.2f}".format(Grid_model.score(X_test, Y_test)))
print("Best parameters:{}".format(Grid_model.best_params_))
print("Best score on train set:{:.2f}".format(Grid_model.best_score_))




