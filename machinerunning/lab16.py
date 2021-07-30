import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir('E:/TEACHING/TEACHING KOREA/MULTICAMPUS/LAB/LAB16/')

# Load a South African Heart Disease dataset and create 5 machine learning models, pick the best and build confidence that the accuracy is reliable. 아프리카 심장병 데이터를 가지고 머신러닝 모델을 개발한 후 성능비교
heart = pd.read_csv('SAheart.csv')

# Basic exploration (head, shape, dtypes, describe). 기본 탐색
heart.head()
heart.shape  # (462, 10)
heart.dtypes
a = heart.describe()
heart.columns

# Visualization (boxplot, hist, pairplot, countplot for chd, boxplot by chd, kdeplot by chd). 시각화
heart.boxplot()
plt.show()
heart.hist()
plt.show()
sns.pairplot(heart, hue='chd')
plt.show()
pd.plotting.scatter_matrix(heart)
plt.show()
sns.countplot(heart.chd)
plt.show()
sns.countplot(heart.famhist)
plt.show()

# encoding famhist
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
heart_copy = heart.copy()
heart.famhist = le.fit_transform(heart.famhist)

# 병유무에 따른 박스플랏
len(heart.columns)
cols = list(heart.columns[:-1])
cnt = 0
fig, axs = plt.subplots(3, 3)
for i in range(3):
    for j in range(3):
        sns.boxplot(x='chd', y=cols[cnt], data=heart, ax=axs[i, j])
        cnt += 1

# 병유무에 따른 밀도그래프
cnt = 0
fig, axs = plt.subplots(3, 3)
for i in range(3):
    for j in range(3):
        sns.kdeplot(x=cols[cnt], data=heart, hue='chd', ax=axs[i, j])
        cnt += 1

# 상관관계
sns.heatmap(heart.corr(), annot=True)

# Check null values (isna), encoding famhist, chd (LabelEncoder). 널값체크, 인코딩
heart.isna().sum()  # 널값체크
heart.dtypes  # 데이터타입 체크
heart.chd = le.fit_transform(heart.chd)  # yes 1 no 0
print(heart.head())
print(heart_copy.head())  # famhist present 1 absent 0

# X and y split. 독립변수, 종속변수 분리
heart.columns
y = heart.chd
x = heart.drop('chd', axis=1)

# Scaling (MinMaxScaler, boxplot). 스케일링
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x = scaler.fit_transform(x)
plt.boxplot(x)
plt.show()

# Train and test split (test_size=.2). 훈련, 테스트 데이터 분리
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1)
x.shape
x_train.shape  # 80%
x_test.shape  # 20%
sns.countplot(y_train)
plt.show()

# Resample (RandomUnderSampler, RandomOverSampler, SMOTE). 재생플링
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler()
x_rus, y_rus = rus.fit_sample(x_train, y_train)
sns.countplot(y_rus)
plt.show()

from imblearn.over_sampling import RandomOverSampler, SMOTE

ros = RandomOverSampler()
x_ros, y_ros = ros.fit_sample(x_train, y_train
sns.countplot(y_ros)
plt.show()

smote = SMOTE()
x_train, y_train = smote.fit_sample(x_train, y_train)

# Model building (logistic regression, SVM, random forest, mlp). 모델빌딩
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(random_state=0)
log_model.fit(x_train, y_train)  # 훈련데이터로 모델생성

# Evaluation 평가
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import cross_val_score

y_pred = log_model.predict(x_test)  # 테스트데이터로 모델평가
log_score = accuracy_score(y_test, y_pred)  # 0.7419354838709677
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print(classification_report(y_test, y_pred))

log_cv = cross_val_score(log_model, x_test, y_test, cv=10, scoring='accuracy')
log_cv.mean(), log_cv.std()  # (0.6977777777777778, 0.08488190779904124)
plot_roc_curve(log_model, x_test, y_test)

# Cross validation comparison (cross_val_score with c=10, summary, boxplot) 교처검증 비교
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

names = ['log_model', 'svm', 'rf', 'mlp']
models = [LogisticRegression(), SVC(), RandomForestClassifier(), MLPClassifier()]
cv_scores = []
for i in range(4):
    models[i].fit(x_train, y_train)
cv_scores.append(cross_val_score(models[i], x_test, y_test, cv=10, scoring='accuracy'))
# 교차검증 요약 그래프
plt.boxplot(cv_scores)  # random forest가 가장 정확함
plt.xticks(range(1, 5), names)
plt.show()
# 교차검증 요약 테이블
df = pd.DataFrame({'mean': np.mean(cv_scores, axis=1),
                   'std': np.std(cv_scores, axis=1)}, index=names)  # mlp

# Confusion matrix comparison (heatmap using subplots). 혼동행렬 비교
cnt = 0
fig, axs = plt.subplots(2, 2)
for i in range(2):
    for
j in range(2):
models[cnt].fit(x_train, y_train)
y_pred = models[cnt].predict(x_test)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, ax=axs[i, j]).set_title(names[cnt])
cnt += 1
plt.show()

# ROC curve comparison (plot_roc_curve using subplots). ROC커브 비교
cnt = 0
fig, axs = plt.subplots(2, 2)
for i in range(2):
    for
j in range(2):
models[cnt].fit(x_train, y_train)
plot_roc_curve(models[cnt], x_test, y_test, ax=axs[i, j])
cnt += 1
plt.show()

#################################################################
# CLUSTERING
from sklearn.metrics.pairwise import pairwise_distances

# 거리 구하기
data = np.array([[1, 1],
                 [2, 2],
                 [10, 10]])
D = pairwise_distances(data)
sns.heatmap(D, annot=True)
plt.show()
# np.sqrt((2-1)**2 + (2-1)**2)
# kmeans
plt.style.available
plt.style.use('ggplot')
x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]
sns.scatterplot(x, y)
plt.show()
# kmeans
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
X = np.array([x, y]).T
kmeans.fit(X)
center = kmeans.cluster_centers_  # 각 그룹의 중앙값
label = kmeans.labels_  # 각 데이터의 그룹
sns.scatterplot(x, y, hue=label)  # 원래 데이터
sns.scatterplot(center[:, 0], center[:, 1], color='black', marker='x', s=150)  # 중앙값
plt.show()
# 덴드로그램
from scipy.cluster.hierarchy import dendrogram, linkage

X = np.array([[5, 3],
              [10, 15],
              [15, 12],
              [24, 10],
              [30, 30],
              [85, 70],
              [71, 80],
              [60, 78],
              [70, 55],
              [80, 91]])
len(X)
dendrogram(linkage(X, 'ward'))

# 계층군집
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=2)
cluster.fit_predict(X)
label = cluster.labels_
# 클러스터에 따른 산점도
sns.scatterplot(X[:, 0], X[:, 1], hue=label)
# , cmap='viridis')
plt.show()
# 클러스터 갯수 정하는 방법
# 1. elbow method
ss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
kmeans.fit(X)
ss.append(kmeans.inertia_)
plt.plot(range(1, 11), ss, marker='x')
plt.xlabel('number of cluster')
plt.ylabel('total within sum of squares')
plt.show()  # 적정 군집수는 2개임

# import matplotlib as mpl
# mpl.rcdefaults() #스타일 없앨때
# 2. silhouette method
from sklearn.metrics import silhouette_score

silhouette = []
for i in range(2, 9):
    kmeans = KMeans(n_clusters=i)
kmeans.fit(X)
silhouette.append(silhouette_score(X, labels=kmeans.labels_))
plt.plot(range(2, 9), silhouette, marker='x', c='blue')
plt.plot([1, 2], [0, silhouette[0]], marker='x', c='blue')
# plt.xlabel('number of cluster')
# plt.ylabel('average silhouette width')
plt.show()  # 적정 군집수는 2개임