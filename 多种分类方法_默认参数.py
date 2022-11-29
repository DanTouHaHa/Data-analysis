from numpy import corrcoef
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA


df = pd.read_excel(r"E:\Fudan_Luoqiang_MDDProject\848MDD_794NC\数据分析\数据整理统计分析\健康人中的分类模型\Global_Power_significantNodeClassification.xlsx")
#print(df)
X = df.iloc[:,2:79]
print(X)
#corrcoef(X)
Y = df.iloc[:,79]
print(Y)
'''
#PCA降维取主成分
pca = PCA(n_components = 1)
X_r = pca.fit(X).transform(X)
#print(X_r)
'''
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
#print(X_std)
#划分训练集、测试集
#X_train, X_test, Y_train, Y_test = train_test_split(X_std, Y, test_size= 0.30)

#定义5种模型
svm = SVC()
lr = LogisticRegression()
tree = DecisionTreeClassifier()
forest = RandomForestClassifier()
Gbdt = GradientBoostingClassifier()
#svm.fit()
#lr.fit()
#tree.fit()
#forest.fit()
#Gbdt.fit()
def muti_score(model):
    #warnings.filterwarnings('ignore')
    accuracy = cross_val_score(model, X_std, Y, scoring='accuracy', cv=5)
    precision = cross_val_score(model,  X_std, Y, scoring='precision', cv=5)
    recall = cross_val_score(model,  X_std, Y, scoring='recall', cv=5)
    f1_score = cross_val_score(model,  X_std, Y, scoring='f1', cv=5)
    auc = cross_val_score(model,  X_std, Y, scoring='roc_auc', cv=5)
    print("准确率:",accuracy.mean())
    print("精确率:",precision.mean())
    print("召回率:",recall.mean())
    print("F1_score:",f1_score.mean())
    print("AUC:",auc.mean())

model_name = ["svm", "lr", "tree","forest","Gbdt"]
for name in model_name:
    model = eval(name)
    print(name)
    muti_score(model)
