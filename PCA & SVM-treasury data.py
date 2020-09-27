import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn import linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import *
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import svm
import time
from sklearn.linear_model import LinearRegression
import seaborn as sns

df = pd.read_csv('hw5_treasury yield curve data.csv')
df=df.dropna(how='any',axis=0)

# Splitting the data into 85% training and 15% test subsets.
X, y = df.iloc[:, 1:31].values, df.iloc[:,31].values

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.15, random_state=42)

scalerX = StandardScaler().fit(X_train)
scalery = StandardScaler()

X_train_std = scalerX.transform(X_train)
X_test_std = scalerX.transform(X_test)

y_train_std = scalery.fit_transform(y_train.reshape(-1,1))
y_test_std = scalery.transform(y_test.reshape(-1,1))

l=[]
for m in range(0,6860):
    for i in y_train_std[m]:
        l.append(i)
y_train_std=np.array(l)
l=[]
for m in range(0,1211):
    for i in y_test_std[m]:
        l.append(i)
y_test_std=np.array(l)

plt.figure(figsize=(12,10))
cm=np.corrcoef(df.iloc[:, 1:32].values.T)
hm = sns.heatmap(cm,cbar=True,square=True,fmt='.2f',annot=False,yticklabels=df.columns[1:32],xticklabels=df.columns[1:32],
                 annot_kws={'size': 15})
plt.tight_layout()
plt.title('Heatmap for all features')
plt.show()

# Eigendecomposition of the covariance matrix.
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
#print('\nEigenvalues \n%s' % eigen_vals)

# Total and explained variance
tot = sum(eigen_vals)
var_exp =[ (i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
var_exp =np.array( [(i / tot) for i in sorted(eigen_vals, reverse=True)])

plt.bar(range(1, 31), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, 31), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('images/05_02.png', dpi=300)
plt.show()

#print the cumulative explained variance
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print("The cumulative explained variance of all features:")
print(pca.explained_variance_ratio_)
print("")

pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
print("The cumulative explained variance of 3 pca features:")
print(pca.explained_variance_ratio_)

#figure out pc 1 to3
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1],X_train_pca[:, 2],s=5)
#plt.xlabel('PC 1')
#plt.ylabel('PC 2')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
plt.show()

def train_and_evaluate(clf,X_train_std,X_test_std,y_train_std,y_test_std,linear_or_not):
    clf.fit(X_train_std,y_train_std)
    y_train_pred = clf.predict(X_train_std)
    y_test_pred = clf.predict(X_test_std)
    plt.scatter(y_train_pred,  y_train_pred - y_train_std,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test_std,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.title('Residual errors')
    plt.hlines(y=0, xmin=-1, xmax=1, color='black', lw=2)
    plt.xlim([-1, 1])
    plt.tight_layout()
    plt.show()
    if(linear_or_not):
        print('Slope: ' ,end=' ')
        clf_coef=clf.coef_
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print(clf_coef)
        print('Intercept: %.3f' % clf.intercept_)
    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train_std, y_train_pred),
        mean_squared_error(y_test_std, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train_std, y_train_pred),
        r2_score(y_test_std, y_test_pred)))

#Applying different models
print("")
clf=SGDRegressor(penalty=None,random_state=42)
slr=LinearRegression()
print("Linear regression: #original datasets without transformation")
start = time.perf_counter()
train_and_evaluate(slr,X_train,X_test,y_train,y_test,True)
end = time.perf_counter()
print ("Using time:",end-start,"s")
print("")

print("Linear regression: #original datasets")
start = time.perf_counter()
train_and_evaluate(clf,X_train_std,X_test_std,y_train,y_test,True)
end = time.perf_counter()
print ("Using time:",end-start,"s")
print("")
print("Linear regression: #pca datasets")
start = time.perf_counter()
train_and_evaluate(clf,X_train_pca,X_test_pca,y_train,y_test,True)
end = time.perf_counter()
print ("Using time:",end-start,"s")
print("")

clf_svr=svm.SVR(kernel="linear")
print("SVM linear regression: #original datasets")
start = time.perf_counter()
train_and_evaluate(clf_svr,X_train_std,X_test_std,y_train,y_test,True)
end = time.perf_counter()
print ("Using time:",end-start,"s")
print("")
print("SVM linear regression: #pca datasets")
start = time.perf_counter()
train_and_evaluate(clf_svr,X_train_pca,X_test_pca,y_train_std,y_test_std,True)
end = time.perf_counter()
print ("Using time:",end-start,"s")
print("")

clf_svr=svm.SVR(kernel="rbf")
print("SVM RBF regression: #original datasets")
start = time.perf_counter()
train_and_evaluate(clf_svr,X_train_std,X_test_std,y_train_std,y_test_std,False)
end = time.perf_counter()
print ("Using time:",end-start,"s")
print("")
print("SVM RBF regression: #pca datasets")
start = time.perf_counter()
train_and_evaluate(clf_svr,X_train_pca,X_test_pca,y_train_std,y_test_std,False)
end = time.perf_counter()
print ("Using time:",end-start,"s")
print("")



print("My name is Xin Zhang")
print("My NetID is: xzhan81")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")














