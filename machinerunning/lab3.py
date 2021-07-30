import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
os.chdir('E:/TEACHING/TEACHING KOREA/MULTICAMPUS/LAB/')

#3D SURFACE PLOT
x = np.arange(-5,5,.25)
y = np.arange(-5,5,.25)
x,y = np.meshgrid(x,y)
r = np.sqrt(x**2 + y**2)
z = np.sin(r)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x,y,z, rstride=1, cstride=1, cmap=cm.viridis)

#3D WIREFRAME PLOT
from mpl_toolkits.mplot3d import axes3d
X, Y, Z = axes3d.get_test_data(0.08)
fig = plt.figure()
chart3d = fig.add_subplot(111, projection='3d')
chart3d.plot_wireframe(X, Y, Z, color='r',rstride=15, cstride=10)
plt.show()

#GEPGRAPICAL MAP
import cartopy.crs as ccrs
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax.set_extent((60, 150, 55, -25))
ax.stock_img()
ax.coastlines()
ax.tissot(facecolor='purple', alpha=.8, lats=37, lons=132)
plt.show()

report = pd.read_csv('seoul_report.txt', delimiter='\t')
report.head()
report['12월'].apply(lambda x: str(x).replace(',','')) #4
report['12월'].str.replace(',','') #1
[str(x) for x in report['12월']] #2
[str(x).strip(',') for x in report['12월']] #3

#NULL값 처리
a = np.array([1,np.nan, 3, 4])
a.dtype
1+np.nan
0*np.nan
a.sum()
np.nansum(a)
a.min()
np.nanmin(a)
a.max()
np.nanmax(a)
pd.Series([1, np.nan, 2, None])
data = pd.Series([1, np.nan, 'hello', None])
data.isnull()
data[data.isnull()] #널값만 추출
data.notnull()
data[data.notnull()] #널이 아닌 값을 추출
df = pd.DataFrame([[1,np.nan, 2], [2,3,5], [np.nan, 4, 6]])
df.dropna() #행에서 널값이 있으면 제거
df.dropna(axis=1) #열에서 널값이 있으면 제거
df.dropna(how='all') #전체행이 널값이면 제거
df.dropna(thresh=3)
data = pd.Series([1, np.nan, 2, None, 3])
data = data.fillna(0) #널값을 0으로 채움
#data.fillna(0, inplace=True) #저장

#null values tutorial
#https://chrisalbon.com/python/data_wrangling/pandas_missing_data
raw_data = {'first_name': ['Jason', np.nan, 'Tina', 'Jake', 'Amy'],
        'last_name': ['Miller', np.nan, 'Ali', 'Milner', 'Cooze'],
        'age': [42, np.nan, 36, 24, 73],
        'sex': ['m', np.nan, 'f', 'm', 'f'],
        'preTestScore': [4, np.nan, np.nan, 2, 3],
        'postTestScore': [25, np.nan, np.nan, 62, 70]}
df = pd.DataFrame(raw_data)
df
df_no_missing = df.dropna()
df_no_missing
df_cleaned = df.dropna(how='all')
df_cleaned
df['location'] = np.nan
df
df.dropna(how='all', axis=1)
df.dropna(thresh=5)
df.fillna(0)
df.preTestScore.fillna(df.preTestScore.mean(), inplace=True)
df.groupby('sex').postTestScore.mean()
df.postTestScore.fillna(df.groupby('sex').postTestScore.transform('mean'),inplace=True)
df[df.age.notnull() & df.sex.notnull()]

#널값 체크하기
df.isnull()
df.isna().sum() #열별로 널값체트
df.dtypes
plt.hist(df.age)
df.age.hist()
df.isna().sum().sum() #전체 데이터의 널값갯수

#Read the world food bank tsv file and assign it to a dataframe called food and complete the following tasks. 월드푸드뱅크 데이터를 읽어서 food라는 데이터프레임에 저장한후 다음 작업을 수행하시오.
food = pd.read_csv('worldfoodfact.tsv', delimiter='\t')
food.head()
food.tail()
food.dtypes
food.info()
food.columns[:100]
a = food.describe()

#Check missing values, count them by columns, and count the total number of missing. 널값 체크, 컬럼별로 갯수체크, 전체갯수
food.isna()
food.isna().sum()
food.isna().sum().sum()

#Drop all missing observations. 관측치별로 전체가 널값인 값 제거
food.dropna(how='all', inplace=True)

#Drop columns where all cells in that column is NA. 전체컬럼이 널값인 컬럼제거
food.dropna(how='all', axis=1, inplace=True)

#Fill NA with the means of each column. 각 컬럼의 평균값으로 널값 채우기
food.fillna(food.mean(), inplace=True)

#world food fact example
#Analyze the world food fact data using graphs
#Scatter plot
sample = food.iloc[:100,:100]
sample.isna().sum()
sample.dropna(how='all', inplace=True)
sample.dropna(how='all', axis=1, inplace=True)
sample.fillna(sample.mean(), inplace=True)
sample.isna().sum()
plt.scatter(sample.sugars_100g,sample.fiber_100g)
plt.grid()
plt.show()

#Histogram
plt.hist(sample.sugars_100g)
plt.show()

#Bar chart, horizontal bar chart, stacked bar chart
a = sample.groupby('countries').count().code
plt.bar(a.index, a)
plt.show()

sample[['sugars_100g', 'fiber_100g']].plot.hist(alpha=.8)

#Box plot
sample['sugars_100g'].plot(kind='box')

#Area chart
plt.stackplot(sample.index,sample['sugars_100g'])

#Heat map
b = sample.corr()
plt.pcolor(b)
plt.colorbar()
plt.xticks(np.arange(len(b.index)),b.index,
           rotation=90)
plt.show()

#replace tutorial
#http://queirozf.com/entries/pandas-dataframe-replace-examples
df = pd.DataFrame({
    'name':['john','mary','paul'],
    'age':[30,25,40],
    'city':['new york','los angeles','london']
})
df.replace(25,40)
df.replace([30,25],np.nan)
df.replace({
        25:26,
        'john':'johnny'
        })
df.replace('jo.+', 'FOO', regex=True)
df.replace('[A-Za-z]', '', regex=True)

df = pd.DataFrame({
    'name':['john','mary','paul'],
    'num_children':[0,4,5],
    'num_pets':[0,1,2]
})
df.replace({'num_pets':{0:1}})

df = pd.DataFrame({
        'score': ['exceptional', 'average', 'good', 'poor', 'average', 'exceptional'],
      	'student':['rob', 'maya', 'parthiv', 'tom', 'julian', 'erica']
          })
df.replace(['poor', 'average', 'good', 'exceptional'],[1,2,3,4])