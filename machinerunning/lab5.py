import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
import os

os.chdir('E:/TEACHING/TEACHING KOREA/MULTICAMPUS/LAB/LAB5/')

loan = pd.read_csv('loan.csv')

# 1. gender, education, loan_status컬럼 선택, female, not gradute, loan_Status Y
loan[(loan.Gender == 'Female') & (loan.Education == 'Not Graduate') & (loan.Loan_Status == 'Y')][
    ['Gender', 'Education', 'Loan_Status']]

# 2. 널값체트 - 행별로, 열별로
loan.isna().sum(axis=1)  # 행별로
loan.isna().sum()  # 열별로

# 3. 범주형 변수의 널값처리시 최빈수 대체하는 방법
loan.Gender.fillna(mode(loan.Gender).mode[0], inplace=True)
loan.Gender.value_counts()
loan.Gender.fillna('Male')
loan.Married.fillna(mode(loan.Married).mode[0], inplace=True)
loan.Dependents.fillna(mode(loan.Dependents).mode[0], inplace=True)
loan.Self_Employed.fillna(mode(loan.Self_Employed).mode[0], inplace=True)

# 4. 피벗테이블 만들기
# gender, married, self_employed가 인덱스, LoanAmount가 value (평균)
table = pd.pivot_table(loan, values='LoanAmount', index=['Gender', 'Married', 'Self_Employed'], aggfunc=np.mean)
table
table.plot(kind='bar')
plt.show()
type(table)
table.index
table.loc[('Female', 'No', 'No')].values[0]

# 5. LoanAmount널값을 위 피벗테이블의 인덱스값에 맞춰서 채우기
loan.set_index('Loan_ID', inplace=True)
for i, row in loan[loan.LoanAmount.isnull()].iterrows():
    loan.loc[i, 'LoanAmount'] = table.loc[row['Gender'], row['Married'], row['Self_Employed']].values[0]

# for i, row in loan.iterrows():
#    print(row['Married'])

# 6.Credit_History와 Loan_Status로 크로스탭 만들어서 막대그래프그리기
# %matplotlib auto
table = pd.crosstab(loan.Credit_History, loan.Loan_Status)
table.plot(kind='bar')
plt.show()

# 7. Property_Area - Rural, Semiurban, Urban
# rates - 1000,5000,1200인 테이블 만들어서 loan테이블과 인너조인한 후 피벗테이블 만들기 (Property_Area와 rates가 인덱스,Credit_History를 values (갯수))
df = pd.DataFrame({'Property_Area': ['Rural', 'Semiurban', 'Urban'],
                   'rates': [1000, 5000, 1200]})
df
loan.shape
loan_merged = pd.merge(loan, df, on='Property_Area')
table = pd.pivot_table(loan_merged, index=['Property_Area', 'rates'], values=['Credit_History'], aggfunc=len)
table.plot(kind='bar')
plt.show()

# 8. ApplicantIncome과 CoapplicantIncome을 내림차순으로 정리해서 ApplicantIncome과 CoapplicantIncome의 처음 10줄을 출력
# loan.reset_index(inplace=True)
loan.sort_values(['ApplicantIncome', 'CoapplicantIncome'], ascending=[False, False])[
    ['ApplicantIncome', 'CoapplicantIncome']].head(10)

# 9. Loan_Status에 따른 ApplicantIncome 박스플랏과 히스토그램
loan.boxplot()
loan.boxplot(column='ApplicantIncome', by='Loan_Status')
loan.hist()
loan.hist(column='ApplicantIncome', bins=30, by='Loan_Status')

# 10. ApplicantIncome를 잘라서 빈도표 만들기 (low, medium, high, very high)
loan['ApplicantIncome_Bin'] = pd.cut(loan.ApplicantIncome, bins=4, labels=['low', 'medium', 'high', 'very high'],
                                     include_lowest=True)
loan.ApplicantIncome_Bin.value_counts()

# 11. Loan_Status N을 0으로, Y를 1로 대체한후 빈도표
loan.Loan_Status.replace({'N': 0, 'Y': 1}, inplace=True)
loan.Loan_Status.value_counts()

# SEABORN EXAMPLE
import seaborn as sns

tips = sns.load_dataset('tips')
tips.head()
tips.dtypes
tips.describe()
tips.isna().sum()

sns.scatterplot('total_bill', 'tip', data=tips)
plt.show()

sns.lmplot('total_bill', 'tip', data=tips, hue='smoker', markers=['o', 'x'], row='sex', col='time')
plt.show()

sns.lmplot('total_bill', 'tip', data=tips, col='day', col_wrap=2, hue='day')
plt.show()

sns.regplot('total_bill', 'tip', data=tips)  # 회귀그래프
plt.show()
sns.residplot('total_bill', 'tip', data=tips)  # 잔차그래프
plt.show()

sns.jointplot('total_bill', 'tip', data=tips, kind='reg')
plt.show()

sns.distplot(tips.total_bill, bins=30)
plt.show()

sns.heatmap(data=tips.corr().round(2), annot=True)
plt.show()

tips.columns
sns.pairplot(tips)
plt.show()

# FACET GRID
g = sns.FacetGrid(data=tips, col='time', row='sex')
g.map(sns.scatterplot, 'total_bill', 'tip')