from pyexpat import model
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from sklearn import metrics
# 对比决策树和随机森林
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
'''
# 对比决策树和随机森林
clf = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=0)

clf = clf.fit(X_train,Y_train)
rfc = rfc.fit(X_train,Y_train)

score_c = clf.score(X_test,Y_test)
score_r = rfc.score(X_test,Y_test)
print("Sigle Tree:{}".format(score_c)
      ,"Random Forest:{}".format(score_r)
     )
'''
parameters = {'n_estimators':range(1,100)}
rf = RandomForestClassifier()
clf = GridSearchCV(estimator=rf,param_grid=parameters, cv = 5)
X_train, X_test, Y_train, Y_test = train_test_split(X_std, Y)
clf.fit(X_train, Y_train)

print("Test set score:{:.2f}".format(clf.score(X_test, Y_test)))
print("Best parameters:{}".format(clf.best_params_))
print("Best score on train set:{:.2f}".format(clf.best_score_))
