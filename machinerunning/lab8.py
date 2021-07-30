import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os

os.chdir('E:/TEACHING/TEACHING KOREA/MULTICAMPUS/LAB/LAB8/')

# Import the train and test data of Big Mart Sales and do some basic exploration tasks. 훈련, 테스트용 데이터를 임포트한 후에 기본 데이터탐색을 수행하시오.
# read_csv() 파일읽기
train = pd.read_csv('bigmart_train.csv')
test = pd.read_csv('bigmart_test.csv')

# shape, dtypes, describe() 데이터검색
train.shape
test.shape
train.dtypes
test.dtypes
a = train.describe(include='all')
a = test.describe(include='all')

# df.isnull().sum() 널값검색
train.isna().sum()
test.isna().sum()

# Distribution and frequency of sales with item visibility. 아이템노출도에 따른 아웃렛세일
train.columns
plt.style.available
plt.style.use('ggplot')
plt.scatter(train.Item_Visibility, train.Item_Outlet_Sales, marker='s', color='navy', s=1)
plt.title('Item Visibility vs Item Outlet Sales')
plt.xlabel('Item Visibility')
plt.ylabel('Item Outlet Sales')
plt.show()

sns.scatterplot(x='Item_Visibility', y='Item_Outlet_Sales', data=train, color='navy', marker='s')
plt.show()

# Sales by outlet identifier. 아웃렛에 따른 총판매량
table = train.groupby('Outlet_Identifier').sum()
sns.barplot(table.index, table.Item_Outlet_Sales, color='purple')
plt.xticks(rotation=90)
plt.show()

# Sales by item type. 아이템종류에 따른 총판매량
train.columns
table = train.groupby('Item_Type').sum()
sns.barplot(table.index, table.Item_Outlet_Sales, color='grey')
plt.xticks(rotation=90)
plt.show()

# Outliers and mean deviation of price by item type. 아이템종류에 따른 상품가격
sns.boxplot(train.Item_Type, train.Item_MRP, color='white')
plt.show()

# Count of Outlet Identifiers. 아웃렛ID에 따른 갯수
table = train.Outlet_Identifier.value_counts()
sns.barplot(table.index, table)
plt.xticks(rotation=90)
plt.show()

# Count of Item Identifiers. 상품 ID에 따른 갯수
train.value_counts('Item_Type')
table = train.Item_Type.value_counts().head(10)
sns.barplot(table.index, table)
plt.xticks(rotation=90)
plt.show()

# Outlet Establishment Year. 아웃렛의 설립년도
table = train.Outlet_Establishment_Year.value_counts()
sns.barplot(table.index, table)
plt.xticks(rotation=90)
plt.show()

# Item Type 상품종류
table = train.Outlet_Establishment_Year.value_counts()
sns.barplot(table.index, table)
plt.xticks(rotation=90)
plt.show()

# %matplotlib auto
# %matplotlib inline

# Set a value 1 for outlet sales in the test dataset. 테스트테이터에 아웃렛세일 열을 만들고 1을 할당
train.shape
train.columns
test.shape
test['Item_Outlet_Sales'] = 1
test.shape

# Combine train and test data. 훈련과 테스트데이터를 합침
df = pd.concat([train, test])
df.shape

# Impute the missing values and assign a value to mis-matched levels. 분실값 및 매칭되지 않는 값 대체
# Impute missing values by median for item weight. 아이템무게의 분실값을 중위수로 대체
df.isna().sum()
df.Item_Weight.fillna(df.Item_Weight.median(), inplace=True)

# Impute missing values by median for item visibility. 노출도의 분실값을 중위수로 대체
df.Item_Visibility.describe()
df.Item_Visibility[df.Item_Visibility == 0] = df.Item_Visibility.median()

# Assign the  name “Other” to unnamed level in outlet size variable. 아웃렛사이즈의 분실값을 Other로 대체
df.isna().sum()
df.Outlet_Size.unique()
df.Outlet_Size.value_counts(dropna=False)
df.Outlet_Size.fillna('Other', inplace=True)

# Rename the various levels of item fat content. 지방함유율의 값 대체
# LF to Low Fat, reg to Regular, low fat to Low Fat
df.Item_Fat_Content.value_counts()
df.Item_Fat_Content.replace({'LF': 'Low Fat',
                             'low fat': 'Low Fat'}, inplace=True)
df.Item_Fat_Content.replace('reg', 'Regular', inplace=True)

# Create a new column, Year =  2021 – outlet establishment year. 새로운 열 만들기
df.columns
df['Year'] = 2021 - df.Outlet_Establishment_Year

# Drop the variables not required in prediction model (exclude item identifier, outlet identifier, item fat content, outlet establishment year, and item type). 필요없는 열을 삭제
df = df.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Fat_Content', 'Outlet_Establishment_Year', 'Item_Type'],
             axis=1)
df.columns

# Divide the bigmart data frame into new_train and new_test. 훈련데이터와 테스트데이터로 나눔
train.shape
test.shape
new_train = df[:8523]
new_test = df[8523:]
new_train.shape
new_test.shape

# Encode the categorical variables. 범주형 변수로 인코딩
new_train.head()
new_train.dtypes
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
new_train.Outlet_Size = le.fit_transform(new_train.Outlet_Size)
new_train.Outlet_Location_Type = le.fit_transform(new_train.Outlet_Location_Type)
new_train.Outlet_Type = le.fit_transform(new_train.Outlet_Type)

# Regression and residual plot. 회귀분석과 잔차그래프
new_train.columns
y = new_train.Item_Outlet_Sales
x = new_train.drop('Item_Outlet_Sales', axis=1)
import statsmodels.api as sm

x = sm.add_constant(x)
model1 = sm.OLS(y, x).fit()
model1.summary()

y_pred = model1.predict(x)
residual = y - y_pred
std_residual = residual / np.std(residual)
sns.residplot(y_pred, std_residual)
plt.show()

# After log transformation, regression and residual plot. 로그변환 후 회귀과 잔차그래프
y = np.log(new_train.Item_Outlet_Sales)
x = new_train.drop('Item_Outlet_Sales', axis=1)
x = sm.add_constant(x)
model2 = sm.OLS(y, x).fit()
model2.summary()

y_pred = model2.predict(x)
residual = y - y_pred
std_residual = residual / np.std(residual)
sns.residplot(y_pred, std_residual)
plt.show()

# Check the r2 score and do some predictions. 성능평가 및 예측
model1.rsquared
model2.rsquared

new_train.columns
x.columns
new_train.iloc[3]
test_data = pd.DataFrame({'const': 1,
                          'Item_Weight': 20,
                          'Item_Visibility': .01,
                          'Item_MRP': 200,
                          'Outlet_Size': 3,
                          'Outlet_Location_Type': 1,
                          'Outlet_Type': 2,
                          'Year': 20}, index=[0])
# test_data = sm.add_constant(test_data)
model2.predict(test_data)
np.exp(model2.predict(test_data))

# NUMPY
a = np.array([1, 2, 3])
a.shape
a.size
a.dtype
a.itemsize
b = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3
b.shape
b.size
b.dtype
b.itemsize

# auto generation
np.arange(10)
np.arange(2, 10)
np.arange(2, 10, 3)
np.linspace(0, 100, 6)
np.zeros(4)
np.zeros((2, 3))
np.ones(3)
np.ones((4, 5))
np.eye(4)

# random numbers
np.random.rand(5)
np.random.rand(3, 4)
np.random.randint(10)
np.random.randint(4, 15)
np.random.randint(10, size=4)
np.random.randint(10, size=(3, 4))

# indexing
a = np.array([[10, 20, 30], [47, 47, 67], [70, 80, 95]])
a[0]
a[1]
a[2]
a[:, 0]
a[:, 1]
a[:, 2]
a[[0, 2]]
a[:, [0, 1]]
a[[0, 2], [0, 1]]

# slicing
a = np.array([0, 10, 20, 30, 40, 50])
# np.arange(0,60,10)
# np.linspace(0,50,6)
a[[1, 3, 4]]

b = np.array([[10, 20, 30],
              [40, 50, 60],
              [70, 80, 90]])
# np.arange(10,100,10).reshape(3,3)
b[:3]
b[:, 1:]
b[:3, 1:]

# math operations
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
a + b
a - b
a * b
a / b
a.dot(b)

x = np.ones((2, 3))
x.shape
y = np.array([[10], [20]])
y.shape
x * y

# iteration
a = np.array([[6, 7, 8], [1, 2, 3], [9, 3, 2]])
for row in a:
    print(row)
a.flatten()
a.flat
for cell in a.flat:
    print(cell)

# stacking
a = np.arange(6).reshape(3, 2)
b = np.arange(6, 12).reshape(3, 2)
np.vstack((a, b))
np.hstack((a, b))

# splitting
a = np.arange(30).reshape(2, 15)
np.vsplit(a, 2)[1]
np.hsplit(a, 3)[2]

# searching and replacing
a = np.arange(12).reshape(3, 4)
a[a > 4] = -1

# exporting and importing
os.getcwd()
np.savetxt('a.txt', a, delimiter=' ')
np.savetxt('a.csv', a, delimiter=',')
np.loadtxt('a.txt')
np.genfromtxt('a.csv', delimiter=',')

# Load data_file.txt using genfromtxt and create a 1d array for time (1 column) and a 2d array for sensor data (2-5 columns). 센서데이터를 연 후 1번 컬럼은 시간을 저장하고, 2-5번 컬럼은 센서데이터로 저장
data = np.loadtxt('data_file.txt', delimiter=',')
time = data[:, 0]
sensor = data[:, 1:5]

# Display the first 6 sensor rows. 센서데이터의 처음 6줄
sensor[:6, :]

# Adjust time to start at zero by subtracting the first element in the time vector (index = 0). 시간을 시간의 0번 인덱스값으로 빼서 0으로 시작하는 시간으로 조정
time = time - time[0]

# Calculate the average of the sensor readings. 각 센서레코드에 대한 평균
sensor_avg = sensor.mean(axis=1)

# Stack time, sensor data, and avg using reshape or transpose. 시간, 센서데이터, 평균데이터를 합치기
time.shape
sensor.shape
sensor_avg.shape
data_adj = np.vstack((time, sensor.T, sensor_avg)).T
data_adj.shape

# Save text file with comma delimiter. 쉼표로 나눠진 텍스트파일로 저장
np.savetxt('data_adj.txt', data_adj, delimiter=' ')

# NUMPY TUTORIAL
x = [0, 1, 2, 3, 4, 5]
a = np.array(x)
a[2]
a[1:4:2]
a[3:]
a[:3]
a.shape
a.size
a.itemsize
a.dtype

b = np.array([[1, 2, 3], [4, 5, 6]])
b.swapaxes(0, 1)
b.T

a = np.arange(6)
a = np.arange(6).reshape(2, 3)
np.array([2, 3, 4])
a = np.arange(1, 12, 2)
a = a.reshape(3, 2)
a.size
a.shape
a.dtype
a.itemsize

b = np.array([(1.5, 2.3), (4, 5, 6)])
b = np.array(1, 2, 3)
a < 4
a * 3
a *= 3
np.zeros((3, 4))
np.ones((2, 3))
np.array([2, 3, 4], dtype=np.int16)
a = np.random.random((2, 3))
np.set_printoptions(precision=2, suppress=True)
a
a.sum()
a.min()
a.max()
a.mean()
a.var()
a.std()
a.sum(axis=1)
a.min(axis=0)
a.argmin()
a.argmax()

a = np.random.randint(0, 10, 5)
a.argsort()
a.sort()

a = np.arange(10) ** 2
a[2]
a[2:5]
for i in a:
    print(i ** 2)
a[::-1]
a = np.random.random((2, 3))
for i in a.flat:
    print(i)
a.transpose()