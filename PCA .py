from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_excel(r"E:\Fudan_Luoqiang_MDDProject\848MDD_794NC\数据分析\数据整理统计分析\健康人中的分类模型\NonGlobal_Dosen_NetworkClassification.xlsx")
#print(df)
X = df.iloc[:,2:8]
#print(X)
#corrcoef(X)
Y = df.iloc[:,8]
#print(Y)
#scaler = StandardScaler()
#X_std = scaler.fit_transform(X)
pca = PCA(n_components = 2)
X_r = pca.fit(X).transform(X)
print(X_r)
# Percentage of variance explained for each components
print(
    "explained variance ratio (first two components): %s"
    % str(pca.explained_variance_ratio_)
)