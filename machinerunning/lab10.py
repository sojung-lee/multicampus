import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import os

os.chdir('E:/TEACHING/TEACHING KOREA/MULTICAMPUS/LAB/LAB10/')

# BOOK INFO SCRAPING EXAMPLE
f = open('test.html', encoding='utf-8')
page = f.read()
soup = BeautifulSoup(page, 'html.parser')
print(soup.prettify())
soup.title.text
soup.body.p
soup.find_all('p')
titles = soup.find_all('p', {'id': 'book_title'})
authors = soup.find_all('p', {'id': 'author'})
for title, author in zip(titles, authors):
# print(title.text + ' / ' + author.text)

title_list = [x.text.strip() for x in titles]
author_list = [x.text.strip() for x in authors]
df = pd.DataFrame({'title': title_list,
                   'author': author_list})
df
soup.select('body p')
soup.select('p')
soup.select('#book_title')
soup.select('p#author')

# WEB ADDRESS SCRAPING EXAMPLE
page = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>사이트 모음</title>
  </head>
  <body>
    <p id="title"><b>자주 가는 사이트 모음</b></p>
    <p id="contents">이곳은 자주 가는 사이트를 모아둔 곳입니다.</p>
    <a href="http://www.naver.com" class="portal" id="naver">네이버</a> <br>
    <a href="https://www.google.com" class="search" id="google">구글</a> <br>
    <a href="http://www.daum.net" class="portal" id="daum">다음</a> <br>
    <a href="http://www.nl.go.kr" class="government" id="nl">국립중앙도서관</a>
  </body>
</html>"""
soup = BeautifulSoup(page, 'lxml')
print(soup.prettify())
soup.a
soup.find('a').get('href')
soup.find_all('a')
soup.find_all('a')
soup.find_all('a', {'class': 'portal'})
soup.find_all('a', {'id': 'naver'})
soup.find_all(id='naver')
soup.select('a#naver')
soup.select('a.portal')

# WEATHER SCRAPING EXAMPLE
url = 'http://forecast.weather.gov/MapClick.php?lat=37.7772&lon=-122.4168'
page = requests.get(url)
page.content
soup = BeautifulSoup(page.content, 'html.parser')
seven_day = soup.find(id='seven-day-forecast')
seven_day.find(class_='period-name').text
seven_day.find(class_='short-desc').text
seven_day.find(class_='temp').text
seven_day.find('img')['title']

# period_tags = seven_day.find_all(class_='period-name')
period_tags = seven_day.select('.period-name')
periods = [p.text for p in period_tags]
short_desc_tags = seven_day.select('.short-desc')
short_descs = [s.text for s in short_desc_tags]
temp_tags = seven_day.select('.temp')
temps = [t.text for t in temp_tags]
desc_tags = seven_day.select('img')
descs = [i['title'] for i in desc_tags]

weather = pd.DataFrame({'period': periods,
                        'short_desc': short_descs,
                        'temp': temps,
                        'desc': descs})
weather['temp_num'] = weather.temp.apply(lambda x: x.split(' ')[1]).astype('int')
weather.dtypes

plt.plot(weather.period, weather.temp_num)
plt.show()

weather_day = weather[weather.temp.str.contains('Low')]

plt.plot(weather_day.period, weather_day.temp_num)
plt.show()

day_avg = weather_day.temp_num.mean()  # 낮온도 평균
print('day temperature : {:.3f}'.format(day_avg))

url = 'https://www.index.go.kr/potal/stts/idxMain/selectPoSttsIdxMainPrint.do?idx_cd=2820&board_cd=INDX_001'
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
table = soup.find(id='t_Table_282001')
col = table.select('tbody td')
y12 = []
for i in np.arange(0, 63, 7):
    y12.append(col[i].text)

url = 'https://movie.naver.com/movie/running/current.nhn#'
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
lst_detail = soup.find(class_='lst_detail_t1')
movies = lst_detail.find_all('li')

movies[0].img['alt']
movies[0].find(class_='info_txt1').find(class_='link_txt').text.strip()  # 개요

movies[0].find_all('dd')[3].find_all('dd')[1]  # 감독

movies[0].find_all('dd')[3].find_all('dd')[2].find_all('a')  # 출연


