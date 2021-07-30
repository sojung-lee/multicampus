import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('E:/TEACHING/TEACHING KOREA/MULTICAMPUS/LAB/LAB6/')

butler = pd.read_excel('butler.xlsx')
butler.columns

#산점도
plt.scatter(butler.miles, butler.time)
plt.xlabel('Miles Traveled')
plt.ylabel('Total Travel Time')
plt.title('Scatter Chart of Miles Traveled and\n Travel Time')
plt.grid()
plt.xlim(0,120)
plt.ylim(0,10)
plt.show()

#최소제곱법
y = butler.time
x = butler.miles
((x - x.mean()) * (y - y.mean())).sum()
((x - x.mean())**2).sum()
slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean())**2).sum()
intercept = y.mean() - slope * x.mean()
y_pred = slope * x + intercept

#결정계수
error = y - y_pred
sse = (error**2).sum()
deviation = y - y.mean()
sst = (deviation**2).sum()
ssr = sst - sse
r = ssr/sst #66.41%의 설명력을 가짐

import statsmodels.api as sm
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
model.summary()

#다중회귀분석
butler = pd.read_excel('butler.xlsx')
butler.head()
butler.columns
y = butler.time
x = butler[['miles', 'num_delivery']]
#x = butler.drop(['id', 'time'], axis=1)

x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
model.summary()

y_preｄ=-0.8687 + 0.0611* butler.miles + 0.9234 * butler.num_delivery

#잔차분석
y_pred=model.predict(x)
ｒｅｓｉｄｕａｌ=y - y_pred

plt.scatter(butler.miles, residual)
plt.grid()
plt.xlabel('Miles Traveled')
plt.ylabel('Residual')
plt.title('Residual Plot Against Miles Traveled')
plt.show()

plt.scatter(butler.num_delivery, residual)
plt.grid()
plt.xlabel('Number of Deliveries')
plt.ylabel('Residual')
plt.title('Residual Plot Against Number of Deliveries')
plt.show()

plt.scatter(y_pred, residual)
plt.grid()
plt.xlabel('Predicted Travel Time')
plt.ylabel('Residual')
plt.title('Residual Plot Against Predicted Travel Time')
plt.show()
import seaborn as sns
sns.residplot(y_pred, residual)
plt.show()

std_residual = residual / np.std(residual)
plt.scatter(y_pred, std_residual)
plt.grid()
plt.xlabel('Predicted Travel Time')
plt.ylabel('Standard Residual')
plt.title('Standard Residual Plot Against Predicted Travel Time')
plt.show()
sns.residplot(y_pred, std_residual)
plt.show()

#Create a data frame with Johnson data. 존슨데이터를 가지고 데이터프레임을 만드십시오.
johnson = pd.read_excel('johnson.xlsx')
johnson.head()
johnson.dtypes
johnson.type
johnson.type.replace(['mechanical', 'electrical'], [0,1], inplace=True)

#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#johnson.type = le.fit_transform(johnson.type)

#Develop the estimated simple linear regression equation to predict the repair time given the type of repair. Does the equation that you developed provide a good fit for the observed data? Explain. 수리종류를 이용하여 수리시간을 예측하는 단순회귀식을 개발하십시오. 이 식은 관측된 데이터를 잘 나타내고 있습니까? 설명하십시오.
johnson.columns
y = johnson.time
x = johnson.type
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
model.summary()

#표준잔차분석
y_pred = model.predict(x)
residual = y-y_pred
std_residual = residual / np.std(residual)
sns.residplot(y_pred, std_residual)
plt.show()

#예측하기
x_test = pd.DataFrame({'const':1,
                       'type':1}, index=[0])
model.predict(x_test)

#Develop the estimated regression equation to predict the repair time given the number of months since the last maintenance service and the type of repair. At the .05 level of significance, test whether the estimated regression equation represents a significant relationship between the independent variables and dependent variable. 수리종류와 지난유지보수서비스이후 달수를 이용하여 수리시간을 예측하는 회귀식을 개발하십시오. 0.05의 유의수준에서 독립변수와 종속변수의 관계를 나타내는 회귀식이 유의한지 설명하십시오.
y = johnson.time
x = johnson[['type', 'month']]
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
model.summary()

#Is the number of months since the last maintenance service statistically significant? Use alpha=.05. What explanation can you give for the results observed? 지난 서비스이후 달수가 통계적으로 유의합니까? 알파값으로 0.05를 사용하여 설명해 주십시오.
#달수의 p값이 0.05보다 작아서 이 변수는 통계적으로 유의한 변수다

#Is the type of repair statistically significant? Use alpha=.05. What explanation can you give for the results observed? 수리종류가 통계적으로 유의합니까? 알파값으로 0.05를 이용하여 설명해 주십시오.
#수리종류의 p값이 0.05보다 작아서 이 변수는 통계적으로 유의한 변수다

#표준잔차분석
y_pred = model.predict(x)
residual = y-y_pred
std_residual = residual / np.std(residual)
sns.residplot(y_pred, std_residual)
plt.show()

#예측하기
x_test = pd.DataFrame({'const':1,
                       'type':1,
                       'month':10}, index=[0])
model.predict(x_test)

#y - repair time, x - monthe since last service
#예측식, 정확도, 통계적으로 유의한지...
y = johnson.time
x = johnson.month
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
model.summary()

#y_pred =  2.1473 + 0.3041 * johnson.month
#모델은 53.4%의 설명력을 가진다
#F테스트의 p값이 0.05보다 작아서 전체 모델은 통계적으로 유의함
#month의 p값이 0.05보다 작아서 month변수는 통계적으로 유의함
#표잔차분석
y_pred=model.predict(x)
residual = y-y_pred
std_residual = residual / np.std(residual)
sns.residplot(y_pred, std_residual)
plt.show()
#잔차분석결과 지금 사용한 모델은 타당하다고 볼 수 있음

#다 통과했으면 실제로 예측하면 됨
x_test = pd.DataFrame({'const':1,
                       'month':10}, index=[0])
model.predict(x_test)

#encoding tutorial
#https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621
data = {'Country':['France', 'Spain', 'Germany', 'Spain', 'Germany', 'France', 'Spain', 'France', 'Germany', 'France'],
        'Age':[44,27,30,38,40,35,np.nan, 48,50,37],
        'Salary':[72000,48000,54000,61000,np.nan, 58000,52000,79000,83000,67000],
        'Purchased':['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']}
country_df = pd.DataFrame(data)
country_df.to_excel('country_df.xlsx')
country_df.dtypes

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
country_df['Country_encoded'] = le.fit_transform(country_df.Country)
le = LabelEncoder()
country_df['Purchased_encoded'] = le.fit_transform(country_df.Purchased)

ohe = OneHotEncoder()
temp = ohe.fit_transform(country_df.Country.values.reshape(-1,1)).toarray()
temp_df = pd.DataFrame(temp, columns = ohe.get_feature_names())
country_df = pd.concat([country_df, temp_df], axis=1)

#np.arange(10).reshape(5,-1)
#np.arange(12).reshape(3,-1)
#np.arange(5).reshape(1,-1)

dummy = pd.get_dummies(country_df.Country)
country_df = pd.concat([country_df, dummy], axis=1)

#occupation으로 인코딩하기 example
pd.read_csv('occupation.csv', delimiter='\t')


