import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir('E:/TEACHING/TEACHING KOREA/MULTICAMPUS/LAB/LAB7/')

x = [1, 1, 2, 3, 3, 3, 4, 4, 5, 6]
y = [45, 55, 50, 75, 40, 45, 30, 35, 25, 15]
df = pd.DataFrame({'x': x, 'y': y})

# 산점도
plt.scatter(x, y)
plt.grid()
plt.show()
sns.regplot(x, y, data=df)
plt.show()
sns.lmplot('x', 'y', data=df)
plt.show()

# 회귀분석
import statsmodels.api as sm

x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
model.summary()

# 잔차분석
y_pred = model.predict(x)
residual = y - y_pred
std_residual = residual / np.std(residual)
# indexes = [i for i,x in enumerate(std_residual) if (x >= 2) | (x <= -2)]

resid_df = pd.DataFrame({'y': y,
                         'y_pred': y_pred,
                         'residual': residual,
                         'std_residual': std_residual})

# 잔차그래프
plt.scatter(y_pred, std_residual)
plt.grid()
plt.show()
sns.residplot(y_pred, std_residual)
plt.show()

# 이상치 수정
resid_df.std_residual[(resid_df.std_residual > 2) | (resid_df.std_residual < -2)].index
df.loc[3, 'y'] = 30
# df.iloc[3,1] = 30

# 이상치 수정후 회귀분석
y = df.y
x = df.x
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
model.summary()

# 잔차분석
y_pred = model.predict(x)
residual = y - y_pred
std_residual = residual / np.std(residual)
resid_df = pd.DataFrame({'y': y,
                         'y_pred': y_pred,
                         'residual': residual,
                         'std_residual': std_residual})

# 잔차그래프
plt.scatter(y_pred, std_residual)
plt.grid()
plt.show()
sns.residplot(y_pred, std_residual)
plt.show()

# 변환예제

# Miles per Gallon Example mpg예제
# Enter it as a matrix and then save it as a data frame. 매트릭스로 입력한후 데이터프레임으로 저장
weight = [2289, 2113, 2180, 2448, 2026, 2702, 2657, 2106, 3226, 3213, 3607, 2888]
mpg = [28.7, 29.2, 34.2, 27.9, 33.3, 26.4, 23.9, 30.5, 18.1, 19.5, 14.3, 20.9]
df = pd.DataFrame({'weight': weight,
                   'mpg': mpg})

# Scatter chart 산점도
sns.lmplot('weight', 'mpg', data=df)
plt.show()

# Regression analysis 회귀분석
y = df.mpg
x = df.weight
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
model.summary()

# Standard residual plot 표준잔차그래프
y_pred = model.predict(x)
residual = y - y_pred
std_residual = residual / np.std(residual)
resid_df = pd.DataFrame({'y': y,
                         'y_pred': y_pred,
                         'residual': residual,
                         'std_residual': std_residual})
sns.residplot(y_pred, std_residual)
plt.show()


# 로그변환
def regression_model(x, y):
    model = sm.OLS(y, x).fit()
    print(model.summary())
    return model


# ln_mpg =  4.5242 -0.0005 * weight

# Standard residual plot 표준잔차그래프
def residual_analysis(x, y, model):
    y_pred = model.predict(x)
    residual = y - y_pred
    std_residual = residual / np.std(residual)
    resid_df = pd.DataFrame({'y': y,
                             'y_pred': y_pred,
                             'residual': residual,
                             'std_residual': std_residual})
    print(resid_df)
    sns.residplot(y_pred, std_residual)
    plt.show()


ln_mpg = np.log(df.mpg)
y = ln_mpg
x = df.weight
x = sm.add_constant(x)
model = regression_model(x, y)
residual_analysis(x, y, model)

# 상호변환
rcp_mpg = 1 / df.mpg
y = rcp_mpg
x = df.weight
x = sm.add_constant(x)
model = regression_model(x, y)
residual_analysis(x, y, model)

# 레이놀즈 예제
reynolds = pd.read_excel('reynolds.xlsx')
reynolds.head()
reynolds.dtypes
y = reynolds.scales
x = reynolds.month

# 산점도
sns.scatterplot(x, y, data=reynolds)
plt.grid()
plt.show()

# 회귀분석
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
model.summary()

# 잔차분석
y_pred = model.predict(x)
residual = y - y_pred
std_residual = residual / np.std(residual)
resid_df = pd.DataFrame({'y': y,
                         'y_pred': y_pred,
                         'residual': residual,
                         'std_residual': std_residual})
print(resid_df)
sns.residplot(y_pred, std_residual)
plt.ylim(-3, 3)
plt.axhline(y=-2, color='r', linestyle='dashed')
plt.axhline(y=2, color='r', linestyle='dashed')
plt.grid()
plt.title('Residual Plot')
plt.show()

# 이차식으로 변환
reynolds['month_sq'] = reynolds.month ** 2
y = reynolds.scales
x = reynolds[['month', 'month_sq']]

# 회귀분석
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
model.summary()

# 잔차분석
y_pred = model.predict(x)
residual = y - y_pred
std_residual = residual / np.std(residual)
resid_df = pd.DataFrame({'y': y,
                         'y_pred': y_pred,
                         'residual': residual,
                         'std_residual': std_residual})
print(resid_df)
sns.residplot(y_pred, std_residual)
plt.ylim(-2, 2)
plt.grid()
plt.show()

# Tyler Example 타일러 예제
# Save it as an excel file. 엑셀파일로 저장
tyler = pd.read_excel('tyler.xlsx')
tyler.dtypes

# sns.scatterplot('price', 'sales', data=tyler, hue='adv')
# plt.show()

table = pd.pivot_table(tyler, values='sales', index='adv', columns='price', aggfunc=np.mean)
table.plot(kind='bar')
plt.show()

plt.scatter(table.columns, table.iloc[0], label='adv 50')
plt.scatter(table.columns, table.iloc[1], label='adv 100')
plt.legend()
plt.grid()
plt.xlim(0, 4)
plt.ylim(0, 900)
plt.show()

# Regression analysis 회귀분석
y = tyler.sales
x = tyler.drop('sales', axis=1)
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
model.summary()

# Standard residual plot 표준잔차분석
y_pred = model.predict(x)
residual = y - y_pred
std_residual = residual / np.std(residual)
resid_df = pd.DataFrame({'y': y,
                         'y_pred': y_pred,
                         'residual': residual,
                         'std_residual': std_residual})
print(resid_df)
sns.residplot(y_pred, std_residual)
# plt.ylim(-2,2)
plt.grid()
plt.show()

# Transformation 변환
# Add PriceAds to the data frame. PriceAds를 데이터프레임에 추가
tyler['price_ads'] = tyler.price * tyler.adv

# Regression analysis 회귀분석
y = tyler.sales
x = tyler.drop('sales', axis=1)
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
model.summary()

# Standard residual plot 표준잔차그래프
y_pred = model.predict(x)
residual = y - y_pred
std_residual = residual / np.std(residual)
resid_df = pd.DataFrame({'y': y,
                         'y_pred': y_pred,
                         'residual': residual,
                         'std_residual': std_residual})
print(resid_df)
sns.residplot(y_pred, std_residual)
# plt.ylim(-2,2)
plt.grid()
plt.axhline(y=-2, color='r', linestyle='dashed')
plt.axhline(y=2, color='r', linestyle='dashed')
plt.show()

resid_df.std_residual[resid_df.std_residual >= 2].index

# CRAVENS EXAMPLE
cravens = pd.read_excel('cravens.xlsx')
cravens.head()
cravens.dtypes
cravens.isna().sum()

# Descriptive statistics 기술통계
a = cravens.describe(include='all')

# Summary statistics for all variables (describe). 모든 변수에 대한 기초통계
# Summary statistics (mean, median, mode, variance, standard deviation, coefficient of variance, and skewness) for Sales. 총판매량에 대한 통계
sales = cravens.Sales
sales.mean()
sales.median()
from scipy.stats import mode

sales.mode()
sales.var()
(sales.std() / sales.mean()) * 100
sales.skew()
sales.hist()

# Univariate analysis 일변량분석
# Frequency, percent frequency, and histogram using bar chart. 빈도표, 백분율빈도표, 히스토그램
(sales.max() - sales.min()) / 9  # 500
bins = np.arange(1500, 7500, 500)
sales_cut = pd.cut(sales, bins, include_lowest=True)
freq = sales_cut.value_counts(sort=False)
r_freq = freq / len(sales)
p_freq = r_freq * 100
table = pd.concat([freq, r_freq, p_freq], axis=1)
freq.plot(kind='bar')
plt.show()
sales.hist(bins=bins)
plt.show()

for col in cravens.columns:
    bins = np.linspace(cravens[col].min(), cravens[col].max(), 10)
    bins_cut = pd.cut(cravens[col], bins, include_lowest=True)
    freq = bins_cut.value_counts(sort=False)
    r_freq = freq / len(col)
    p_freq = r_freq * 100
    table = pd.concat([freq, r_freq, p_freq], axis=1)
    print(table)

cravens.hist()
plt.tight_layout()
plt.show()

cravens.boxplot()
plt.tight_layout()
plt.show()

# Multivariate analysis 다변량분석
# Correlation matrix, scatter plot (pairplot)
상관관계, 산점도
cravens.corr()
sns.heatmap(cravens.corr(), annot=True)
plt.show()

g = sns.pairplot(cravens)
g.fig.set_figwidth(10)
g.fig.set_figheight(10)
plt.show()

# Multiple regression with: 회귀분석
# all variables 모든 변수사용
y = cravens.Sales
x = cravens.drop('Sales', axis=1)
x = sm.add_constant(x)
model1 = sm.OLS(y, x).fit()
model1.summary()
# r sqaured = 0.922

# 'Poten', 'AdvExp', 'Share‘
y = cravens.Sales
x = cravens[['Poten', 'AdvExp', 'Share']]
x = sm.add_constant(x)
model2 = sm.OLS(y, x).fit()
model2.summary()
#  0.849

# 'Poten', 'AdvExp', 'Share', 'Accounts‘
y = cravens.Sales
x = cravens[['Poten', 'AdvExp', 'Share', 'Accounts']]
x = sm.add_constant(x)
model3 = sm.OLS(y, x).fit()
model3.summary()
# 0.900

# 'Share', 'Change', 'Accounts', 'Work', 'Rating‘
y = cravens.Sales
x = cravens[['Share', 'Change', 'Accounts', 'Work', 'Rating']]
x = sm.add_constant(x)
model4 = sm.OLS(y, x).fit()
model4.summary()
# 0.700

# 'Poten', 'AdvExp', 'Share', 'Change', 'Time‘
y = cravens.Sales
x = cravens[['Share', 'Change', 'Accounts', 'Work', 'Rating']]
x = sm.add_constant(x)
model4 = sm.OLS(y, x).fit()
model4.summary()
# 0.915

# Compare above models and determine a best model.
위
모델들을
비교하고
가장
좋은
모델을
선택
# 'Poten', 'AdvExp', 'Share', 'Accounts‘
y = cravens.Sales
x = cravens[['Poten', 'AdvExp', 'Share', 'Accounts']]
x = sm.add_constant(x)
model3 = sm.OLS(y, x).fit()
model3.summary()
# 0.900

# Standard residual plot for the best model. 가장 좋은 모델의 표준잔차그래프
y_pred = model3.predict(x)
residual = y - y_pred
std_residual = residual / np.std(residual)
sns.residplot(y_pred, std_residual)
plt.show()

# Prediction with test data (Poten=74065.1, AdvExp=4582.9, Share=2.51, Accounts:74.86). 테스트데이터로 예측
test_data = pd.DataFrame({'const': 1,
                          'Poten': 74065.1,
                          'AdvExp': 4582.9,
                          'Share': 2.51,
                          'Accounts': 74.86}, index=[0])
model3.predict(test_data)