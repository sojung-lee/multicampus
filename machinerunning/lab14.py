import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale, normalize, Binarizer
import os

os.chdir('E:/TEACHING/TEACHING KOREA/MULTICAMPUS/LAB/LAB14/')

data = {'score': [234, 24, 14, 27, -74, 46, 73, -18, 59, 160]}
df = pd.DataFrame(data)
type(df)

df.score.plot(kind='bar')
plt.show()

scaler = MinMaxScaler()
score_scaled = scaler.fit_transform(df.score.values.reshape(-1, 1))

len(df)
plt.bar(np.arange(1, 11), score_scaled.flatten())
plt.show()

# %matplotlib inline
input_data = np.array([[3, -1.5, 3, -6.4], [0, 3, -1.3, 4.1], [1, 2.3, -2.9, -4.3]])
scaled_data = scaler.fit_transform(input_data)
scaler.inverse_transform(scaled_data)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(input_data)
scaled_data.mean().round()
scaled_data.std()
scaler.inverse_transform(scaled_data)

scale(input_data)
normalize(input_data, norm='l2')
Binarizer().transform(input_data)

# Open pima indian data and scale data (between 0 and 1).
pima = pd.read_csv('pima-indians-diabetes.csv', header=None)
pima.columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

y = pima['class']
x = pima.drop('class', axis=1)

x.boxplot()
plt.show()

scaler = MinMaxScaler()
x1 = scaler.fit_transform(x)
plt.boxplot(x1)
plt.show()

# Standardize data (0 mean, 1 stdev).
scaler = StandardScaler()
x2 = scaler.fit_transform(x)
plt.boxplot(x2)
plt.show()

# Binarize data.
x3 = Binarizer().transform(x)
plt.boxplot(x3)
plt.show()

# Use the USArrests.csv file to reduce the dimension.
arrest = pd.read_csv('USArrests.csv')
arrest.set_index('Unnamed: 0', inplace=True)
arrest.boxplot()
plt.show()

# Scale the measures before MDS. 정규화
scaler = StandardScaler()
arrest = scaler.fit_transform(arrest)
plt.boxplot(arrest)
plt.show()

# 2D projection using Metric MDS. 계량형 MDS사용해서 차원축소
from sklearn.manifold import MDS, Isomap

mds = MDS(n_components=2, random_state=0)
result = mds.fit(arrest)
df = pd.DataFrame(result.embedding_)
df.columns = ['c1', 'c2']
plt.scatter(df.c1, df.c2)

plt.show()

# Determining the number of components. 변수의 갯수를 결정
k = range(1, 50)
stress = []
for i in k:
    stress.append(MDS(n_components=i).fit(arrest).stress_)
plt.plot(k, stress)
plt.show()

# 2D projection using Nonmetric MDS (Isomap). 비계량형 MDS사용해서 차원축소
isomap = Isomap(n_components=2, n_neighbors=10)
result1 = isomap.fit(arrest)
df1 = pd.DataFrame(result1.embedding_)
df1.columns = ['c1', 'c2']
plt.scatter(df1.c1, df1.c2)
plt.show()

# Use the eurpean_city_distances.csv for dimension reduction.
distance = pd.read_csv('eurpean_city_distances.csv', delimiter=';', header=None)
distance
distance.set_index(0, inplace=True)
# distance.set_index(distance.columns[0], inplace=True)
distance.isna().sum()
distance.dropna(axis=1, inplace=True)
distance.columns = distance.index

# 2D projection using Metric MDS (write the city names on the map using annotate). 계량형 MDS사용해서 차원축소
plt.style.use('ggplot')
mds = MDS(n_components=2, random_state=0)
result = mds.fit(distance)
df = pd.DataFrame(result.embedding_)
df.columns = ['c1', 'c2']
plt.scatter(df.c1, df.c2)
for i, j, city in zip(df.c1, df.c2, distance.index):
    plt.annotate(xy=[i, j], s=city, ha='center', xytext=[0, 10], textcoords='offset points')
plt.title('Metric MDS')
plt.show()

# %matplotlib auto
# 2D projection using Nonmetric MDS (Isomap). 비계량형 MDS사용해서 차원축소
isomap = Isomap(n_components=2, n_neighbors=10)
result1 = isomap.fit(distance)
df1 = pd.DataFrame(result1.embedding_)
df1.columns = ['c1', 'c2']
plt.scatter(df1.c1, df1.c2)
for i, j, city in zip(df1.c1, df1.c2, distance.index):
    plt.annotate(xy=[i, j], s=city, ha='center', xytext=[0, 10], textcoords='offset points')
plt.title('Nonmetric MDS')
plt.show()
distance.shape

# Determining the number of components. 변수의 갯수를 결정
k = range(1, 50)
stress = [MDS(n_components=i).fit(distance).stress_ for i in k]
plt.plot(k, stress)
plt.show()
np.argmin(stress)

# PCA
# Use the USArrests.csv file to reduce the dimension.
arrest = pd.read_csv('USArrests.csv')
arrest.set_index('Unnamed: 0', inplace=True)

# 2D projection using PCA. PCA사용해서 차원축소
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
df = pca.fit_transform(arrest)
df = pd.DataFrame(df)
df.columns = ['c1', 'c2']
plt.scatter(df.c1, df.c2)
for i, j, city in zip(df.c1, df.c2, arrest.index):
    plt.annotate(xy=[i, j], s=city, ha='center', xytext=[0, 10], textcoords='offset points')
plt.show()

# Print rotation matrix, explained ratio, accumulated ratio.
pca.components_.T
pca.explained_variance_ratio_
pca.explained_variance_ratio_.sum()

# Plot accumulated ratio
k = range(1, 5)
ratio = []
for i in k:
    pca = PCA(n_components=i).fit(arrest)
    ratio.append(pca.explained_variance_ratio_.sum())
plt.plot(k, ratio)
plt.xlabel('k components')
plt.ylabel('cumulative ev ratio')
plt.show()

# Use the eurpean_city_distances.csv to reduce dimension.
distance = pd.read_csv('eurpean_city_distances.csv', delimiter=';', header=None)
distance.set_index(0, inplace=True)
distance.dropna(axis=1, inplace=True)
distance.columns = distance.index

# 2D projection using PCA. PCA사용해서 차원축소
pca = PCA(n_components=2)
result = pca.fit_transform(distance)
df1 = pd.DataFrame(result)
df1.columns = ['c1', 'c2']

plt.scatter(df1.c1, df1.c2)
for i, j, city in zip(df1.c1, df1.c2, distance.index):
    plt.annotate(xy=[i, j], s=city, ha='center', xytext=[0, 10], textcoords='offset points')
plt.show()

# Print rotation matrix, explained ratio, accumulated ratio.
pca.components_.T
pca.explained_variance_ratio_
pca.explained_variance_ratio_.sum()

# Plot accumulated ratio
k = range(1, len(distance) + 1)
ratio = []
for i in k:
    pca = PCA(n_components=i).fit(distance)
    ratio.append(pca.explained_variance_ratio_.sum())
plt.plot(k, ratio)
plt.xlabel('k components')
plt.ylabel('variance')
plt.show()

# kNN TUTORIAL
cancer = pd.read_csv('breast-cancer-wisconsin.data.txt')
cancer.head()
cancer.dtypes
a = cancer.describe()

# preprocessing
cancer.replace('?', -99999, inplace=True)
cancer.bare_nuclei = cancer.bare_nuclei.astype('int64')
cancer.set_index('id', inplace=True)

# x and y split
cancer.columns
y = cancer['class']
x = cancer.drop('class', axis=1)

x.boxplot()
plt.show()

# train and test split
from sklearn.model_selection import train_test_split, cross_val_score

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=3)

# knn model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
knn.n_neighbors

# evaluation
from sklearn.metrics import accuracy_score

y_pred = knn.predict(x_test)
accuracy_score(y_train, knn.predict(x_train))  # 훈련데이터 정확도
knn_score = accuracy_score(y_test, y_pred)  # 테스트데이터 정확도
knn.score(x_test, y_test)  # 테스트데이터 정확도
knn_cv = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
knn_cv.mean(), knn_cv.std()

# prediction with test data
x_train.iloc[0, :]
test_data = np.array([1, 1, 1, 1, 2, 1, 2, 3, 1])
knn.predict(test_data.reshape(1, -1))  # benign

# decide the optimal number of neighbors
k = list(range(1, 50))
cv_list = []
for i in k:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    cv_list.append(cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy').mean())

error = [1 - x for x in cv_list]
plt.plot(k, error)
plt.show()
np.min(error)
print('optimal number of neighbors - %d' % (np.argmin(error) + 1))

# SVM
from sklearn.svm import SVC

svm = SVC()
svm.fit(x_train, y_train)

# evaluation
from sklearn.metrics import accuracy_score

y_pred = svm.predict(x_test)
svm_score = accuracy_score(y_test, y_pred)  # 테스트데이터 정확도
print('accuracy score {}'.format(svm_score))
svm.score(x_test, y_test)  # 테스트데이터 정확도
svm_cv = cross_val_score(svm, x_train, y_train, cv=10, scoring='accuracy')
print('cv mean: {}, cv std: {}'.format(svm_cv.mean(), svm_cv.std()))

# prediction with test data
x_train.iloc[0, :]
test_data = np.array([1, 1, 1, 1, 2, 1, 2, 3, 1])
print(svm.predict(test_data.reshape(1, -1)))  # benign