import pandas as pd
import os
os.chdir('E:/TEACHING/TEACHING KOREA/MULTICAMPUS/LAB/')

s = pd.Series([1,2,3,4], index=['a', 'b', 'c', 'd'])
print(s)

data = {'country':['belgium', 'india', 'brazil'],
        'capital':['brussels', 'new delhi', 'brasilia'],
        'population':[11111, 22222, 333333]
        }
df = pd.DataFrame(data)
print(df)

data = [['belgium', 'brussels', 11111],
         ['india', 'new delhi', 22222],
         ['brazil', 'brasilia', 33333]]
df = pd.DataFrame(data, columns=['country', 'capital', 'population'])
print(df)
df.to_csv('country.csv')

#Create a data frame with any class information (courseID, courseName, creditHours, Days) and save it as an Excel file. 아무 클래스 정보를 이용해서 데이터프레임을 만든 후 엑셀파일로 저장하시오.
data = {'courseID':[111, 222, 333],
        'courseName':['math', 'science' ,'eng'],
        'creditHours':[2, 3, 3],
        'days':['mon', 'tues', 'thurs']}
df = pd.DataFrame(data)
df.to_excel('courses.xlsx')

#Read weather.txt file into a dataframe. 날씨파일을 데이터프레임으로 읽기
weather = pd.read_csv('weather.txt')
weather

#Print first 5 or last 3 rows of df. 처음 다섯줄, 마지막 세줄 출력
weather.head()
weather.tail(3)

#Get data types, index, columns, values. 데이터타입, 인덱스, 컬럼들, 값들
weather.dtypes
weather.index
weather.columns
weather.values

#Statistical summary of each column. 각 컬럼에 대한 통계요약
weather.describe()

#Sort records by any column (descending order). 한 컬럼을 정해 내림차순으로 정렬
weather.sort_values(by='month', ascending=False)

#Slice the records and display the following columns and rows:
레코드를 짤라서 다음의 열과 행을 출력하시오.
#avg_low
weather.columns
weather.avg_low

#Rows 1 to 2
weather[0:2]

#avg_low and avg_high
weather[['avg_low', 'avg_high']]

#9 row of avg_precipitation column
weather.loc[8,'avg_precipitation']
weather.avg_precipitation[8]

#4 to 5 rows of 1 and 4 columns
weather.iloc[3:5, [0,3]]

#Filter the data and display the following columns and rows: 데이터를 검색하여 다음의 열과 행을 출력하시오.
#avg_precipitation > 1.0
weather[weather.avg_precipitation > 1]

#Month is in either June, July, or August
weather[weather.month.isin(['Jun', 'Jul', 'Aug'])]

#Assign new values in the following locations: 새로운 값으로 대체하시오.
#101.3 for avg_precipitation column at index 9
weather.avg_precipitation[9] = 101.3

#Np null values for avg_precipitation column at index 9 (np.nan)
import numpy as np
weather.avg_precipitation[9] = np.nan

#5 for all rows in avg_low column. avg_low 컬럼 전체를 5로 바꾸기
weather.avg_low = 5

#Add new column named avg_day that is the average of avg_low and avg_high. avg_low와 avg_high의 평균을 구하여 avg_day컬럼을 만드시오.
weather['avg_day'] = (weather.avg_low + weather.avg_high)/2

#Rename columns 컬럼이름 바꾸기
#avg_precipitation to avg_rain
weather.rename(columns={'avg_precipitation':'avg_rain'}, inplace=True)

#Change columns’ name to 'month','av_hi','av_lo','rec_hi','rec_lo','av_rain‘, ‘av_day’
weather.columns = ['month','av_hi','av_lo','rec_hi','rec_lo','av_rain', 'av_day']

#Save the result data frame to a csv file. 데이터프레임을 csv파일로 저장하시오.
weather.to_csv('weather.csv')

#CHIPOTLE EXAMPLE
#Import the chipo dataset and assign it to a variable called chipo.
데이터셋을 임포트해서 chipo라고 하시오.
chipo = pd.read_excel('chipotle.xlsx')

#Display the first 10 rows. 처음 10줄
chipo.head(10)

#What is the number of observations in the dataset? 관측치의 갯수는?
chipo.shape[0]
len(chipo)
len(chipo.index)
chipo.index.size

#What is the number of columns in the dataset? 컬럼의 수는?
chipo.shape[1]
chipo.columns.size
len(chipo.columns)

#Print the name of all the columns. 모든 컬럼의 이름을 출력
chipo.columns

#How is the dataset indexed? 데이터셋은 어떻게 인덱스 되었나요?
chipo.index

#How many items were ordered in total? 전체 주문 아이템수는? Quantity의 합
chipo.quantity.sum()

#How much was the revenue for the period in the dataset? 데이터셋 안에 있는 기간에 대한 수익은 얼마인가요?
(chipo.quantity * chipo.item_price).sum()

#Which was the most-ordered item? 가장 많이 주문된 아이템은? Item_name에 따라 groupby된 것중 가장 카운트가 많이 된 것을 찾음
chipo.item_name.value_counts().index[0]
chipo.groupby('item_name').count().sort_values(by='order_id', ascending=False).order_id.index[0]

#For the most-ordered item, how many items were ordered? 가장 많이 주문된 아이템은 몇개나 주문됐나요? Groupby된것의 카운트수
chipo.item_name.value_counts()[0]

#What was the most ordered item in the choice_description column? choice_description컬럼에서 가장 많이 주문된 아이템은? choice_description에 따라 groupby된 것중 가장 카운트가 많은 것
chipo.choice_description.value_counts().index[0]
chipo.groupby('choice_description').count().order_id.sort_values(ascending=False).index[0]

#How many different items are sold? 얼마나 많은 다른 종류의 아이템이 팔렸나요?
chipo.item_name.nunique()

#WORLD FOOD FACTS EXAMPLE
#Import the en.openfoodfacts.org.products.tsv dataset and assign it to a dataframe called food. 세계음식팩트 파일을 임포트해서 food라는 데이터프레임에 넣으시오.
food = pd.read_csv('worldfoodfact.tsv', delimiter='\t')

#Display the first 5 rows. 처음 5줄
food.head()

#What is the number of observations in the dataset? 관측치의 갯수
food.shape[0]

#What is the number of columns in the dataset? 컬럼의 수
food.shape[1]

#Print the name of all the columns. 모든 컬럼의 이름
food.columns

#What is the name of 105th column? 105번째 컬럼의 이름
food.columns[104]

#What is the data type of the observations of the 105th column? 105번째 컬럼의 데이터타입
food.dtypes[104]

#How is the dataset indexed? 데이터셋은 어떻게 인덱스되었나요?
food.index

#What is the product name of the 19th observation? 19번째 관측치의 상품이름은?
food.product_name[18]

#Import the occupation dataset and assign it to a variable called users and use the 'user_id' as index 데이터셋을 임포트해서 users라는 변수에 저장한후 user_id를 인덱스로 사용하시오.
users = pd.read_csv('occupation.txt', delimiter='|')
users.set_index('user_id', inplace=True)

#Display the first 25 rows. 처음 25줄
users.head(25)

#Display the last 10 rows. 마지막 10줄
users.tail(10)

#What is the number of observations in the dataset? 관측치의 수
users.shape[0]

#What is the number of columns in the dataset? 컬럼의 수
users.shape[1]

#Print the name of all the columns. 모든 컬럼의 이름
users.columns

#How is the dataset indexed? 데이터셋은 어떻게 인덱스 되었나요?
users.index

#What is the data type of each column? 각 컬럼의 데이터타입은?
users.dtypes

#Print only the occupation column 직업컬럼만 출력
users.occupation
users['occupation']

#Summarize the data frame (descriptive statistics). 기술통계를 이용하여 데이터요약
users.describe()

#Summarize all the columns. 모든 컬럼을 요약
users.describe(include='all')

#Summarize only the occupation column. 직업 컬럼만 요약
users.occupation.describe()

#What is the mean age of users? 모든 사용자의 나이의 평균
users.age.mean()

#How many different occupations there are in this dataset? 직업의 종류는?
users.occupation.nunique()
len(users.occupation.value_counts())

#What is the most frequent occupation? 가장 많은 직업은?
users.occupation.value_counts().index[0]

#What is the age with least occurrence? 가장 적게 나오는 나이는?
users.age.value_counts().tail().index