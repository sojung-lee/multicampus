import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('E:/TEACHING/TEACHING KOREA/MULTICAMPUS/LAB/')


#MERGE EXAMPLE
left = pd.DataFrame({ 'id':[1,2,3,4,5],
                     'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
                     'subject_id':['sub1','sub2','sub4','sub6','sub5']})
right = pd.DataFrame( {'id':[1,2,3,4,5],
                       'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
                       'subject_id':['sub2','sub4','sub3','sub6','sub5']})
print(pd.merge(left, right, on='subject_id', how='left'))
print(pd.merge(left, right, on='subject_id', how='right'))
print(pd.merge(left, right, on='subject_id', how='outer'))
print(pd.merge(left, right, on='subject_id', how='inner'))
print(pd.merge(left, right, on='subject_id'))
import pandas as pd
left = pd.DataFrame({ 'id':[1,2,3,4,5],
                     'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
                     'subject_id':['sub1','sub2','sub4','sub6','sub5']})
right = pd.DataFrame( {'id':[1,2,3,4,5],
                       'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
                       'subject_id':['sub2','sub4','sub3','sub6','sub5']})
print(pd.merge(left, right, on='subject_id', how='left'))
print(pd.merge(left, right, on='subject_id', how='right'))
print(pd.merge(left, right, on='subject_id', how='outer'))
print(pd.merge(left, right, on='subject_id', how='inner'))
print(pd.merge(left, right, on=['subject_id','id']))

#CONCAT EXAMPLE
df1 = pd.DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
df2 = pd.DataFrame([['c', 3], ['d', 4]], columns=['letter', 'number'])
print(pd.concat([df1, df2]))
print(pd.concat([df1, df2], axis=1))

#MERGE TUTORIAL
#https://chrisalbon.com/python/data_wrangling/pandas_join_merge_dataframe/
raw_data = {
        'subject_id': ['1', '2', '3', '4', '5'],
        'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
        'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}
df_a = pd.DataFrame(raw_data)

raw_data = {
        'subject_id': ['4', '5', '6', '7', '8'],
        'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
        'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}
df_b = pd.DataFrame(raw_data)

raw_data = {
        'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
        'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}
df_n = pd.DataFrame(raw_data)
df_new = pd.concat([df_a, df_b])
pd.concat([df_a, df_b], axis=1)
pd.merge(df_n, df_new, on='subject_id')
pd.merge(df_new, df_n, left_on='subject_id', right_on='subject_id')
pd.merge(df_a, df_b, on='subject_id', how='outer')
pd.merge(df_a, df_b, on='subject_id', how='inner')
pd.merge(df_a, df_b, on='subject_id', how='left')
pd.merge(df_a, df_b, on='subject_id', how='right')
pd.merge(df_a, df_b, on='subject_id', how='left', suffixes=('_left', '_right'))
pd.merge(df_a, df_b, right_index=True,left_index=True)

#PIVOT TABLE TUTORIAL
#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.pivot_table.html
df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
                         "bar", "bar", "bar", "bar"],
                   "B": ["one", "one", "one", "two", "two",
                         "one", "one", "two", "two"],
                   "C": ["small", "large", "large", "small",
                         "small", "large", "small", "small",
                         "large"],
                   "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                   "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})
table = pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'], aggfunc=np.sum, fill_value=0)
table.plot(kind='bar')
table = pd.pivot_table(df, values=['D', 'E'], index=['A', 'C'],aggfunc={'D':np.mean,'E':np.mean})
table.plot(kind='bar')
table = pd.pivot_table(df, values=['D', 'E'], index=['A', 'C'],aggfunc={'D':np.mean,'E':[np.min, np.max, np.mean]})
table.plot(kind='bar')

#CROSSTAB TUTORIAL
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.crosstab.html
a = np.array(["foo", "foo", "foo", "foo", "bar", "bar",
              "bar", "bar", "foo", "foo", "foo"], dtype=object)
b = np.array(["one", "one", "one", "two", "one", "one",
              "one", "two", "two", "two", "one"], dtype=object)
c = np.array(["dull", "dull", "shiny", "dull", "dull", "shiny","shiny", "dull", "shiny", "shiny", "shiny"],           dtype=object)
table = pd.crosstab(a, [b, c], rownames=['a'], colnames=['b','c'])
table.plot(kind='bar')
foo = pd.Categorical(['a', 'b'], categories=['a', 'b', 'c'])
bar = pd.Categorical(['d', 'e'], categories=['d', 'e', 'f'])
table = pd.crosstab(foo, bar, dropna=False)
table.plot(kind='bar', grid=True)

#APPLY EXAMPLE
df = pd.DataFrame([[4,9],]*3, columns=['A', 'B'])
df.apply(np.sqrt)
np.sqrt(df)
df.apply(np.sum, axis=0) #컬럼별로 합
df.apply(np.sum, axis=1) #행별로 합

#MAP EXAMPLE
x = map(lambda x: len(x), ('apple', 'banana', 'cherry'))
list(x)

def myfunc(x):
    return len(x)
x = map(myfunc,('apple', 'banana', 'cherry'))
list(x)

data = pd.DataFrame({'fruit':['apple', 'banana', 'cherry']})
data.fruit.apply(lambda x: len(x))
data.fruit.apply(myfunc)

#MAP EXAMPLE2
x = pd.Series([1,2,3], index=['one', 'two', 'three'])
y = pd.Series(['foo', 'bar', 'baz'], index=[1,2,3])
z = {1: 'A', 2: 'B', 3: 'C'}
x.map(y)
x.map(z)

#LOAN ANALYSIS
loan = pd.read_csv('loan.csv')
loan.head()
loan.tail()
loan.dtypes
loan.describe()
loan.isna().sum()

#계량형변수를 범주형 변수로 바꿔서 분석
bins = [loan.LoanAmount.min()] + [90,140,190] + [loan.LoanAmount.max()]
loan['LoanAmount_Bin'] = pd.cut(loan.LoanAmount, bins, labels=['low', 'medium', 'high', 'very high'], include_lowest = True)
loan.LoanAmount_Bin.value_counts(sort=False)

#범주형에 따른 계량형 변수 분석
loan.boxplot(column='ApplicantIncome', by='Education')
plt.show()

#범주형에 따른 범주형 변수 분석
table = pd.crosstab(loan.Loan_Status, loan.Credit_History)
table.plot(kind='bar', stacked=True)

#계량형에 따른 계량형 변수 분석 - scatterplot
#그래프를 여러개 넣는 경우
table = pd.crosstab(loan.Loan_Status, loan.Credit_History)
table.plot(kind='bar', stacked=True)

#Read loan.csv to create a data frame called df.
loan = pd.read_csv('loan.csv')

#Explore numeric data. 계량형 데이터분석
#head(), describe()
loan.head()
loan.describe()

#histogram using plt.hist or df.hist for ApplicantIncome. 수입히스토그램
loan.hist(column='ApplicantIncome', bins=30)

#boxplot using plt.boxplot or df.boxplot for ApplicantIncome. 수입박스플랏
loan.boxplot(column='ApplicantIncome')

#histogram using df.hist for LoanAmount. 대출금에 대한 히스토그램
loan.hist(column='LoanAmount', bins=30)

#boxplot using df.boxplot for LoanAmount. 대출금에 대한 박스플랏
loan.boxplot(column='LoanAmount')

plt.figure(figsize=(7,7))
plt.subplot(221)
plt.hist(loan.ApplicantIncome, bins=30)
plt.subplot(222)
loan.boxplot(column='ApplicantIncome')
plt.subplot(223)
plt.hist(loan.LoanAmount, bins=30)
plt.subplot(224)
loan.boxplot(column='LoanAmount')
plt.show()

#boxplot using df.boxplot for ApplicantIncome by Education. 교육에 따른 수입 박스플랏
loan.boxplot(column='ApplicantIncome', by='Education')
plt.show()
loan.barplot(column='ApplicantIncome', by='Education')
plt.show()

table = loan.pivot_table(values='ApplicantIncome', index=['Education'],aggfunc=np.sum)
table.plot(kind='bar')
plt.show()

#Explore non-numeric data. 범주형 데이터 분석
#Frequency table using value_counts for Property_Area and Credit_History. 부동산지역과 신용기록 빈도표
loan.Property_Area.value_counts()
loan.Credit_History.value_counts()

#crosstab and bar plot using pd.crosstab for Loan_Status by Credit_History. 신용기록에 따른 대출여부
table = pd.crosstab(loan.Loan_Status, loan.Credit_History)
table.plot(kind='bar')
plt.show()

#crosstab and bar plot using pd.crosstab for Loan_Status by Married. 결혼여부에 따른 대출여부
table = pd.crosstab(loan.Loan_Status, loan.Married)
table.plot(kind='bar')
plt.show()

#crosstab and bar plot using pd.crosstab for Loan_Status by Self_Employed. 자영업여부에 따른 대출여부
table = pd.crosstab(loan.Loan_Status, loan.Self_Employed)
table.plot(kind='bar')
plt.show()

#crosstab and bar plot using pd.crosstab for Loan_Status by Property_Area. 부동산지역에 따른 대출여부
table = pd.crosstab(loan.Loan_Status, loan.Property_Area)
table.plot(kind='bar')
plt.show()

#crosstab and stacked bar plot using pd.crosstab for Loan_Status by Credit_History and Gender. 신용기록과 성별에 따른 대출여부
table = pd.crosstab(loan.Loan_Status, [loan.Credit_History, loan.Gender])
table.plot(kind='bar', stacked=True)
plt.show()

#DATA MANIPULATION TUTORIAL
import pandas as pd
loan = pd.read_csv('loan.csv')
loan.set_index('Loan_ID', inplace=True)

#1. gender, education, loan_status컬럼 선택, female, not gradute, loan_Status Y
a = loan[loan.Gender == 'Female'][loan.Education == 'Not Graduate'][ loan.Loan_Status == 'Y']
a[['Gender','Education','Loan_Status']]

loan[(loan.Gender == 'Female') & (loan.Education == 'Not Graduate') & (loan.Loan_Status == 'Y')][['Gender','Education','Loan_Status']]

#2. 널값체트 - 행별로, 열별로
loan.isna().sum(axis = 1)
loan.isna().sum()

#3. 범주형 변수의 널값처리시 최빈수 대체하는 방법
loan.dtypes
loan.isna().sum()
from scipy.stats import mode
loan.Gender.fillna(mode(loan.Gender).mode[0], inplace=True)
loan.Gender.fillna(loan.Gender.value_counts().index[0], inplace=True) #대체하는 방법
loan.Married.fillna(mode(loan.Married).mode[0], inplace=True)
loan.Dependents.fillna(mode(loan.Dependents).mode[0], inplace=True)
loan.Self_Employed.fillna(mode(loan.Self_Employed).mode[0], inplace=True)
loan.Gender.fillna(mode(loan.Gender).mode[0], inplace=True)
loan.Credit_History.value_counts(dropna=False)