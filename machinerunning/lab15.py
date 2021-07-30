import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir('E:/TEACHING/TEACHING KOREA/MULTICAMPUS/LAB/LAB15/')

pima = pd.read_csv('pima-indians-diabetes.csv', header=None)
pima.columns=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

#기본탐색
pima.head()
pima.shape
pima.dtypes
a = pima.describe()
pima.isna().sum()

#시각화
pima.hist()
plt.show()

#클래스에 따른 속성분석 (밀도그래프) distplot, kdeplot
fig, axes = plt.subplots(2,2)
sns.kdeplot(x='preg', hue='class', data=pima, ax=axes[0,0], legend=None)
sns.kdeplot(x='plas', hue='class',data=pima, ax=axes[0,1], legend=None)
sns.kdeplot(x='pres', hue='class',data=pima, ax=axes[1,0], legend=None)
sns.kdeplot(x='skin', hue='class',data=pima, ax=axes[1,1], legend=None)
fig, axes = plt.subplots(2,2)
sns.kdeplot(x='test', hue='class', data=pima, ax=axes[0,0], legend=None)
sns.kdeplot(x='mass', hue='class',data=pima, ax=axes[0,1], legend=None)
sns.kdeplot(x='pedi', hue='class',data=pima, ax=axes[1,0], legend=None)
sns.kdeplot(x='age', hue='class',data=pima, ax=axes[1,1], legend=None)

pima.boxplot()
plt.show()

#클래스에 따른 속성분석 (박스플랏) boxplot
fig, axes = plt.subplots(2,2)
sns.boxplot(x='class', y='preg', data=pima, ax=axes[0,0])
sns.boxplot(x='class', y='plas', data=pima, ax=axes[0,1])
sns.boxplot(x='class', y='pres', data=pima, ax=axes[1,0])
sns.boxplot(x='class', y='skin', data=pima, ax=axes[1,1])
fig, axes = plt.subplots(2,2)
sns.boxplot(x='class', y='test', data=pima, ax=axes[0,0])
sns.boxplot(x='class', y='mass', data=pima, ax=axes[0,1])
sns.boxplot(x='class', y='pedi', data=pima, ax=axes[1,0])
sns.boxplot(x='class', y='age', data=pima, ax=axes[1,1])

pd.plotting.scatter_matrix(pima)
sns.pairplot(pima, hue='class')

tab = pima.groupby('class').size()
sns.barplot(x=tab.index, y=tab)

#널값처리, 인코딩
#x and y split
y = pima['class']
x = pima.drop('class', axis=1)
x.boxplot()
plt.show()

#SCALING
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
plt.boxplot(x)
plt.show()

#train and test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2, random_state=1)

#model building
#DT, RF, LDA, KNN, SVC, MLP, LOGISTIC REGRESSION
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

#evaluation
y_pred = dt.predict(x_test)
dt_score = accuracy_score(y_test, y_pred)
dt_cv = cross_val_score(dt, x_test, y_test, cv=10, scoring='accuracy')
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.xlabel('predicted')
plt.ylabel('acutual')
plt.show()
print(classification_report(y_test, y_pred))

#tuning - 파마미터 조정

#prediction with test data
pima[:1]
test_data = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
dt.predict(test_data)

#logistic regression
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

import statsmodels.api as sm
log_reg = sm.Logit(y_train, x_train).fit()
log_reg.summary()

#변수선택
#univariate selection
from sklearn.feature_selection import SelectKBest, RFE
skb = SelectKBest(k=4)
fit = skb.fit(x,y)
df = pd.DataFrame({'var':pima.columns[:-1],
                   'score':fit.scores_})
df.sort_values('score', ascending=False)

#recursive feature elimination (rfe)
log_reg = LogisticRegression()
rfe = RFE(log_reg)
fit = rfe.fit(x,y)
df = pd.DataFrame({'var':pima.columns[:-1],
                   'rank':fit.ranking_})
df.sort_values('rank')

#random forest selection
rf = RandomForestClassifier()
fit = rf.fit(x,y)
df = pd.DataFrame({'var':pima.columns[:-1],
                   'importance':fit.feature_importances_})
df.sort_values('importance', ascending=False)



