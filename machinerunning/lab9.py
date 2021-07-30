import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os

os.chdir('E:/TEACHING/TEACHING KOREA/MULTICAMPUS/LAB/LAB9/')

# BODY TEMPERATURE EXAMPLE
# Read the file in numpy using the command np.loadtxt() and put it into a numpy 2d array. 파일을 읽어서 2d 배열로 저장
body = np.loadtxt('BodyTemperature.txt', skiprows=1)
body.shape

# Extract the number of Males and Females in the dataset. 남자와 여자 데이터의 갯수
len(body[body[:, 1] == 1])
len(body[body[:, 1] == 2])

# Compute the overall mean for Temperature and HeartRate. 체온과 심장박동수의 평균
np.mean(body, axis=0)[0]
np.mean(body, axis=0)[2]

# Compute the mean, max and min of Temperature and HeartRate for Male and Females separately and write the results on the file BodyTemperature_gender.txt in a table format. 성별 체온, 심장박동수의 평균, 최대값, 최소값을 구하여 합친 후 텍스트파일로 저장
male = body[body[:, 1] == 1]
male_temp = np.hstack((np.mean(male, axis=0)[0], np.max(male, axis=0)[0], np.min(male, axis=0)[0]))
male_hr = np.hstack((np.mean(male, axis=0)[2], np.max(male, axis=0)[2], np.min(male, axis=0)[2]))

female = body[body[:, 1] == 2]
female_temp = np.hstack((np.mean(female, axis=0)[0], np.max(female, axis=0)[0], np.min(female, axis=0)[0]))
female_hr = np.hstack((np.mean(female, axis=0)[2], np.max(female, axis=0)[2], np.min(female, axis=0)[2]))
gender_sum = np.vstack((male_temp, male_hr, female_temp, female_hr))
gender_sum
# np.set_printoptions(precision=2, suppress=False)
gender_sum
np.savetxt('gender_sum.txt', gender_sum, delimiter='\t', fmt='%.2f')

# Normalize Temperature to 0-1 range. 체온을 0에서 1사이의 값으로 정규화. ((temp – min) / (max – min))을 이용하시오.
temp = body[:, 0]
temp_norm = (temp - temp.min()) / (temp.max() - temp.min())
temp_norm.min()
temp_norm.max()

# WEB SCRAPING EXAMPLE
import requests
from bs4 import BeautifulSoup

page = requests.get('http://dataquestio.github.io/web-scraping-pages/simple.html')
page.content
soup = BeautifulSoup(page.content, 'html.parser')
print(soup.prettify())
soup.p.get_text()
soup.title.text

page = requests.get('http://dataquestio.github.io/web-scraping-pages/ids_and_classes.html')
soup = BeautifulSoup(page.content, 'html.parser')
print(soup.prettify())
soup.p
soup.find('p')
soup.find_all('p')[3].text.strip()
soup.find_all(class_='outer-text')[0].text.strip()
soup.find_all(id='first')[0].text.strip()
soup.find_all('p', {'class': 'outer-text'})  # ******
soup.find_all('p', class_='outer-text')
soup.find_all('p', {'class': 'inner-text'})
soup.find_all('p', class_='inner-text')
soup.select('p.outer-text')
soup.select('p.inner-text')
soup.select('p#first')
soup.select('div p#first')

page = requests.get('https://en.wikipedia.org/wiki/List_of_state_and_union_territory_capitals_in_India')
soup = BeautifulSoup(page.content, 'html.parser')
print(soup.prettify())
soup.table
soup.find('table')
len(soup.find_all('table'))
table = soup.find_all('table', {'class': 'wikitable sortable plainrowheaders'})[0]

A = []
B = []
C = []
D = []
E = []
F = []
G = []

for row in table.findAll('tr'):
    cells = row.findAll('td')
    states = row.findAll('th')
    if len(cells) == 6:
        A.append(cells[0].text.strip())
        B.append(states[0].text.strip())
        C.append(cells[1].text.strip())
        D.append(cells[2].text.strip())
        E.append(cells[3].text.strip())
        F.append(cells[4].text.strip())
        G.append(cells[5].text.strip())

df = pd.DataFrame({'No': A,
                   'State': B,
                   'Adm. Capital': C,
                   'Leg. Capital': D,
                   'Jud. Capital': E,
                   'Year': F,
                   'Former Capital': G})
df.to_excel('capital.xlsx')

soup.find_all('a')[1]['href']

all_links = soup.find_all('a')
for link in all_links:
    print(link.get('href'))

# WEB API EXAMPLE
url = 'http://data.insight.go.kr:8080/openapi/service/PriceInfo/getPriceInfo?serviceKey=aSkTEXbtl4XRymqEwiFSQEGeMPo3tK%2B7pQ8nLVw4qlxJOhgTYTvPvODJ1nshAM2iu6R9EvCvAh5OvCu01y2Myg%3D%3D&itemCode=A019170&startDate=20150101&endDate=20150101&pageNo=1&numOfRows=10'
page = requests.get(url)  ##
soup = BeautifulSoup(page.content, 'html.parser')
print(soup.prettify())
ic = soup.ic.text
iname = soup.find('in').text
pi = soup.find_all('pi')
# pi_list = []
# for i in pi:
#    pi_list.append(i.text)
pi_list = [i.text for i in pi]
pn = soup.find_all('pn')
pn_list = [i.text for i in pn]
sp = soup.find_all('sp')
sp_list = [i.text for i in sp]
sd = soup.find_all('sd')
sd_list = [i.text for i in sd]

df = pd.DataFrame({'pi': pi_list,
                   'pn': pn_list,
                   'sp': sp_list,
                   'sd': sd_list})
df.to_excel('price_info.xlsx')

import json

# json을 파이썬 딕셔너리로 변환
x = '{ "name":"John", "age":30, "city":"New York"}'
y = json.loads(x)
y['name']
y['age']
y['city']
# 파이썬 딕셔너리를 json으로 변환
x = {
    "name": "John",
    "age": 30,
    "city": "New York"
}
json.dumps(x)

x = {
    "name": "John",
    "age": 30,
    "married": True,
    "divorced": False,
    "children": ("Ann", "Billy"),
    "pets": None,
    "cars": [
        {"model": "BMW 230", "mpg": 27.5},
        {"model": "Ford Edge", "mpg": 24.1}
    ]
}
x['cars'][0]['model']  # BMW 230 추출
x['children'][1]

url = 'http://api.open-notify.org/iss-now.json'
page = requests.get(url)
data = json.loads(page.content)
data['iss_position']['longitude']