
#%matplotlib auto 그래프를 독립적으로 그릴 때
#%matplotlib inline 그래프를 콘솔창에 그릴 때

import pandas as pd
import matplotlib.pyplot as plt

#선그래프
plt.plot([1,2,3], [5,7,4])
plt.ylim(0,8)
plt.show()

x = [1,2,3]
y1 = [5,5,8]
y2 = [10,15,6]
plt.plot(x,y1, label='first line')
plt.plot(x,y2, label='second line')
plt.legend()
plt.title('Line Chart')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#산점도
x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [4, 5, 6, 7, 7, 8, 4, 3]
plt.scatter(x,y, marker='x', s=100, color='k')
plt.title('Scatter Plot\n Example')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#히스토그램
population_ages = [22, 55, 62, 45, 21, 22, 34, 42, 42, 4, 99, 102, 110, 120, 121, 122, 130, 111, 115, 112, 80,75, 65, 54, 44, 43, 42, 48]
bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130]
plt.hist(population_ages, bins, rwidth=.8)
plt.title('Histogram')
plt.xlabel('ages')
plt.ylabel('count')
plt.show()

#막대그래프
x = [1,2,3,4,5]
y = [8,4,5,6,7]

plt.bar(x,y, color='r')
plt.title('Bar Chart')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#수평막대그래프
x = [1,2,3,4,5]
y = [8,4,5,6,7]

plt.barh(x,y, color='r')
plt.title('Bar Chart')
plt.xlabel('y')
plt.ylabel('x')
plt.show()

#Combined Bar Chart
x1 = [1, 3, 5, 7, 9]
x2 = [2, 4, 6, 8, 10]
y1 = [3, 6, 3, 7, 8]
y2 = [2, 5, 1, 8, 6]
plt.bar(x1,y1, label='bar1', color='r')
plt.bar(x2,y2, label='bar2', color='c')
plt.title('Two Bar Charts')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#Stacked Bar Chart
x = [1, 2, 3, 4, 5]
y1 = [2, 4, 6, 3, 1]
y2 = [4, 7, 5, 7, 3]
plt.bar(x,y1, label='bar1', color='r')
plt.bar(x,y2,bottom=y1, label='bar2', color='c')
plt.title('Stacked Bar Chart')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#Boxplot
data = [[2, 4, 6, 8, 10],[6 ,7, 8, 2, 4],[1, 3, 5, 7, 9],[7, 8, 2, 4, 2]]
df = pd.DataFrame(data)
plt.boxplot(df)
plt.title('Box Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#Area Chart
days = [1, 2, 3, 4, 5]
sleeping  = [7, 8, 6, 11, 7]
eating = [2, 3, 4, 3, 2]
working = [7, 8, 7, 2, 2]
playing = [8, 5, 7, 8, 13]

plt.stackplot(days, sleeping, eating, working, playing,
              colors=['m', 'c', 'r', 'k'])
plt.title('Area Chart')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(labels=['sleeping', 'eating', 'working', 'playing'])
plt.show()

#Heatmap
data = [[2, 4, 6, 8, 10],[6 ,7, 8, 2, 4],[1, 3, 5, 7, 9],[7, 8, 2, 4, 2]] #5x4
df = pd.DataFrame(data)
plt.pcolor(df)
plt.colorbar()
plt.show()

#Pie Chart
data = [3,5,6,2]
plt.pie(data, labels=['sleeping', 'eating', 'working', 'playing'], colors=['c', 'm', 'r', 'k'], startangle=90,
        shadow=True,explode=(0,.3,0,0),autopct='%.1f%%')
plt.title('Pie Chart')
plt.show()

#CHART USING PANDAS
data = pd.read_csv('graph_pd.txt')
data.columns = ['x', 'y']
plt.plot(data.x, data.y)
plt.show()

#CHART USING NUMPY
import numpy as np
data = np.loadtxt('graph_np.txt', delimiter=',')
x = data[0,:]
y = data[1,:]
plt.plot(x,y)
plt.ylim(0,10)
plt.show()

#CHIPOTLE EXAMPLE
#Analyze the Chipotle data using graphs. 치폴레 데이터를 가지고 그래프를 그려 분석하시오.
import os
os.chdir('E:/TEACHING/TEACHING KOREA/MULTICAMPUS/LAB/')
chipo = pd.read_excel('chipotle.xlsx')
chipo.columns
chipo.dtypes

#Scatter plot (이변량 연속, 연속)
plt.scatter(chipo.quantity, chipo.item_price)
plt.show()

#Histogram (일변량 연속)
plt.hist(chipo.quantity, bins=30)
plt.show()

plt.hist(chipo.item_price)
plt.show()

#Bar chart (이변량 범주, 연속)
a = chipo.groupby('item_name').sum().quantity
plt.bar(a.index, a)
plt.xticks(rotation=90)
plt.show()

#Horizontal bar chart (일변량 범주)
b = chipo.choice_description.value_counts()
plt.barh(b.index, b)
plt.show()

#씨리즈
s = pd.Series([2,3,4,8], index=['a', 'b', 'b', 'b'])
s.sum()
s.index

#Box plot (일변량 연속)
c = chipo.quantity.value_counts()
plt.boxplot(chipo[['quantity', 'item_price']])
plt.show()

#plt.boxplot(chipo.quantity)
#plt.show()
#
#plt.boxplot(chipo.item_price)
#plt.show()

#Heat map (이변량 연속, 연속)
plt.matshow(chipo[['quantity', 'item_price']].corr())
plt.colorbar()
plt.show()

plt.pcolor(chipo[['quantity', 'item_price']].corr())
plt.colorbar()
plt.show()

#연속변수를 범주형으로 바꾸기
d = pd.cut(chipo.quantity, bins=3, labels=['low', 'middle', 'high']).value_counts(sort=False)

d.plot(kind='bar')
plt.show()

plt.style.available
plt.style.use('ggplot')

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

#서울시 교통사고 현황
#1월에서 12월까지 발생건수
report = pd.read_csv('seoul_report.txt', delimiter='\t')
a = report.iloc[0,:]['1월':'12월']
a = a.apply(lambda x: x.replace(',', '')).astype('int')
plt.plot(a)
plt.show()

#12월 사망자수
b = report[report.구분 == '사망자수'][2:][['자치구별','12월']]
plt.bar(b.자치구별, b['12월'])
plt.xticks(rotation=90)
plt.show()