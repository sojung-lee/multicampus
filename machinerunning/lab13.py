import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir('E:/TEACHING/TEACHING KOREA/MULTICAMPUS/LAB/LAB13/')

from sklearn import datasets
data = datasets.load_iris()
print(data.DESCR)
data.keys()
data.data
data.feature_names
data.target
data.target_names

iris = pd.DataFrame(data.data, columns=data.feature_names)
iris['species'] = data.target
iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris.species.replace(0,'setosa', inplace=True)
iris.species.replace(1,'versicolor', inplace=True)
iris.species.replace(2,'virginica', inplace=True)

#Summarize the data frame. 데이터프레임 요약
#Shape, head(20), describe(), groupby(‘class’).size(), dtypes
iris.shape
iris.head(20)
iris.describe()
iris.groupby('species').size()
iris.dtypes

#Data visualization. 데이터시각화
#Univariate plot – boxplot, histogram. 박스플랏과 히스토그램을 이용한 일변량분석
iris.boxplot()
plt.show()

iris.hist()
plt.show()

#Multivariate plot – scatter_matrix. 산점도를 이용한 다변량 분석
from pandas.plotting import scatter_matrix
scatter_matrix(iris)
sns.pairplot(iris)

#Multiple graphs
#fig, axs = plt.subplot(nrows=2, ncols=2)
#sns.histplot(x='sepal_length', data=iris, ax=axs[0,0])
#sns.histplot(x='sepal_length', data=iris, ax=axs[0,1])

#species analysis
#종에 따른 통계 및 그래프 countplot
tab = iris.groupby('species').size()
pct = (tab / tab.sum()) *100
freq_tab = pd.concat([tab, pct], axis=1)
freq_tab.columns = ['freq', 'percentage']

sns.countplot(x='species', data=iris)
sns.barplot(tab.index, tab)

#속성에 따른 박스플랏 boxplot
iris.boxplot()
col_names = list(iris.columns[:4])
for i in range(len(col_names)):
    plt.subplot(1,4,i+1)
    iris.boxplot(column=col_names[i])
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(ncols=4)
for i in range(len(col_names)):
    sns.boxplot(y=col_names[i], data=iris, ax=axes[i])

#속성끼리의 관계 pairplot
from pandas.plotting import scatter_matrix
scatter_matrix(iris)
sns.pairplot(iris, hue='species')

#종에 따른 속성분석 (박스플랏) boxplot
fig, axes = plt.subplots(2,2)
sns.boxplot(x='species', y='sepal_length', data=iris, ax=axes[0,0])
sns.boxplot(x='species', y='sepal_width', data=iris, ax=axes[0,1])
sns.boxplot(x='species', y='petal_length', data=iris, ax=axes[1,0])
sns.boxplot(x='species', y='petal_width', data=iris, ax=axes[1,1])

#종에 따른 속성분석 (밀도그래프) distplot, kdeplot
fig, axes = plt.subplots(2,2)
sns.kdeplot(x='sepal_length', hue='species', data=iris, ax=axes[0,0], legend=None)
sns.kdeplot(x='sepal_width', hue='species',data=iris, ax=axes[0,1], legend=None)
sns.kdeplot(x='petal_length', hue='species',data=iris, ax=axes[1,0], legend=None)
sns.kdeplot(x='petal_width', hue='species',data=iris, ax=axes[1,1], legend=None)

#x and y split
y = iris.species
x = iris.drop('species', axis=1)

#y encoding
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
le = LabelEncoder()
y = le.fit_transform(y)
data.target

#x scaling
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
plt.boxplot(x)
plt.show()

#train and test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2, random_state=1)

#linear regression model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)
lm.coef_
lm.intercept_

#evaluation
from sklearn.metrics import r2_score, accuracy_score
y_pred = lm.predict(x_test)
lm_score = r2_score(y_test, y_pred)

#cross validation
from sklearn.model_selection import cross_val_score
lm_cv = cross_val_score(lm, x_train, y_train, cv=10, scoring='r2')
print(lm_cv.mean(), lm_cv.std())

#decision tree model
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=1, max_depth=3)
dtc.fit(x_train, y_train)

#evaluation
y_pred = dtc.predict(x_test)
dtc_score = accuracy_score(y_test, y_pred)

#cross validation
dtc_cv = cross_val_score(dtc, x_train, y_train, cv=10, scoring='accuracy')
print(dtc_cv.mean(), dtc_cv.std())

#decision tree visualization
from IPython.display import Image
from sklearn import tree
import pydotplus

dot_data = tree.export_graphviz(dtc, feature_names=iris.columns[:4], class_names=data.target_names, filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

graph.write_png('iris.png')
graph.write_pdf('iris.pdf')

#random forest model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)
rfc.fit(x_train, y_train)

#evaluation
y_pred = rfc.predict(x_test)
rfc_score = accuracy_score(y_test, y_pred)
rfc_cv = cross_val_score(rfc, x_train, y_train, cv=10, scoring='accuracy')
print(rfc_cv.mean(), rfc_cv.std())

#comparison
#print('Comparison')
#print('\t\t\tMean \t\tSTD')
#print('Linear Regression:\t {:.4f} \t {:.4f}'.format(lm_cv.mean(), lm_cv.std()))
#print('Decision Tree:\t\t {:.4f} \t {:.4f}'.format(dtc_cv.mean(), dtc_cv.std()))
#print('Random Forest:\t\t {:.4f} \t {:.4f}'.format(rfc_cv.mean(), rfc_cv.std()))

print('Comparison')
mean = [lm_cv.mean(), dtc_cv.mean(), rfc_cv.mean()]
std = [lm_cv.std(), dtc_cv.std(), rfc_cv.std()]
table = pd.DataFrame({'Mean': mean,
                      'STD': std}, index=['lm', 'dt', 'rf'])
print(table)

#comparision with boxplot
df = pd.DataFrame({'lm': lm_cv,
                   'dt': dtc_cv,
                   'rf': rfc_cv})
df.boxplot()
plt.show()

df.hist()
plt.show()

test_data = x_train[:1]
scaler.inverse_transform(test_data)
rfc.predict(test_data)