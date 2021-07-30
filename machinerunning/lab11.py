import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
help(nltk)
nltk.download('popular')

text="""Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.
The sky is pinkish-blue. You shouldn't eat cardboard"""
sent_tokenize(text)
words = word_tokenize(text)
fdist = FreqDist(words)
fdist.keys()
fdist.values()

#빈도표
df = pd.DataFrame({'word':list(fdist.keys()),
                   'freq':list(fdist.values())})
df.sort_values('freq', ascending=False, inplace=True)

top5 = fdist.most_common(5)
top5_df = pd.DataFrame(top5)
top5_df.columns = ['word', 'freq']

#빈도그래프
fdist.plot(10, cumulative=True)
plt.show()

plt.bar(top5_df.word, top5_df.freq)
plt.show()

#Split the text into individual words and create a frequency table and plot. 다음 글을 단어로 쪼개서 빈도테이블과 그래프 그리기
text = """Now, I understand that because it's an election season expectations for what we will achieve this year are low But, Mister Speaker, I appreciate the constructive approach that you and other leaders took at the end of last year to pass a budget and make tax cuts permanent for working\
families. So I hope we can work together this year on some bipartisan priorities like criminal justice reform and helping people who are battling prescription drug abuse and heroin abuse. So, who knows, we might surprise the cynics again"""

words = word_tokenize(text)
fdist=FreqDist(words)
df = pd.DataFrame({'word':list(fdist.keys()),
                   'freq':list(fdist.values())})
df.sort_values('freq', ascending=False, inplace=True)
plt.bar(df.word, df.freq)
plt.xticks(rotation=90)
plt.show()

#Split the text into sentences and tokenize the sentences and count the number of words. Draw the bar plot. 먼저 문장으로 쪼개고 다시 문장을 단어로 쪼개서 문장별 단어의 갯수를 카운트하고 막대그래프 그리기
sents = sent_tokenize(text)
len(sents)
sent_len = []
for s in sents:
    words = word_tokenize(s)
#    print(len(words))
    sent_len.append(len(words))
plt.bar(np.arange(1,4), sent_len)
plt.show()

#불용어제거
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
len(set(stop_words))
stop_words.append("'s") #or ('\'s')
filter_words = [w for w in words if w.lower() not in stop_words]
print(len(words), len(filter_words))

from nltk.stem import PorterStemmer
ps = PorterStemmer()
stem_words = [ps.stem(w) for w in filter_words]

#lemmatization
from nltk.stem.wordnet import WordNetLemmatizer
wnl = WordNetLemmatizer()
wnl.lemmatize('flying', 'v')

#pos tagging
sent = "Albert Einstein was born in Ulm, Germany in 1879."
words = word_tokenize(sent)
nltk.pos_tag(words)

#punctuation
import string
string.punctuation
punc_word = [w for w in stem_words if w not in string.punctuation]
fdist = FreqDist(punc_word)
fdist.plot()

#텍스트에서 바로 문장부호 뺄때
''.join([char for char in text if char not in string.punctuation])

#altogether
import re
text="Hello Mr. Smith, how are you doing today? \
The weather is great, and city is awesome.\
The sky is pinkish-blue. You shouldn't eat cardboard."
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stop_words]
    return text
clean_text(text)

#word cloud
text1 = ' '.join(punc_word)
from wordcloud import WordCloud, STOPWORDS
STOPWORDS
wordcloud = WordCloud(max_font_size=50,
                      max_words=10,
                      background_color='white',
                      stopwords=STOPWORDS,
                      colormap='Accent_r').generate(text1)
fig = plt.figure(figsize=(10,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

import os
os.chdir('E:/TEACHING/TEACHING KOREA/MULTICAMPUS/LAB/LAB11')
fig.savefig('wordcloud1.png')
wordcloud.to_file('wordcloud2.png')

#링크에 있는 텍스트를 이용해서 불용어처리, 어간추출, 문장부호를 제거한 후 워드클라우드를 그리시오.
import requests
from bs4 import BeautifulSoup
text = requests.get('http://programminghistorian.github.io/ph-submissions/assets/basic-text-processing-in-r/sotu_text/236.txt').text
text.encoding

words = clean_text(text)
text1 = ' '.join(words)
wordcloud = WordCloud(background_color='white',
                      stopwords=STOPWORDS,
                      colormap='Accent_r').generate(text1)
fig = plt.figure(figsize=(10,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#Barack Obama’s 2016 State of the Union Address Example
2016년 오바마연설 예제
url = 'http://programminghistorian.github.io/ph-submissions/assets/basic-text-processing-in-r/sotu_text/236.txt'
page = requests.get(url)
text = page.text
#soup = BeautifulSoup(page.content, 'html.parser')
#text = soup.text

#Split the text into individual words and create a frequency table. 텍스트를 단어로 나누고 빈도테이블
words = word_tokenize(text)
fdist = FreqDist(words)
df = pd.DataFrame({'word':list(fdist.keys()),
                   'freq':list(fdist.values())})
df.sort_values('freq', ascending=False)

#Filter words with a frequency less than 0.1% using word_frequency.csv. 단어빈도표를 사용해서 빈도가 0.1이하인 단어를 선택
freq = pd.read_csv('http://programminghistorian.github.io/ph-submissions/assets/basic-text-processing-in-r/word_frequency.csv')
tab = pd.merge(df, freq, on='word')
tab.sort_values('frequency', ascending=False, inplace=True)
tab[tab.frequency < 0.1].head(15)

#Filter words with a frequency less than 0.002%. 빈도가 0.002이하인 단어를 선택
tab[tab.frequency < 0.002].head(15)

#Summarize the top five most used words that have a frequency less than 0.002% in the Google Web Corpus with metadata.csv. 빈도가 0.002이하 중에서 빈도가 가장 높은 5개의 단어요약
meta = pd.read_csv('http://programminghistorian.github.io/ph-submissions/assets/basic-text-processing-in-r/metadata.csv')
#236 --> 235
meta.columns
meta.president[235]
meta.year[235]
top5_words = '; '.join(list(tab[tab.frequency < 0.002].head(5).word))
print('{}; {}; {}'.format(meta.president[235],meta.year[235], top5_words))
#print('%s; %s; %s' % (meta.president[235],meta.year[235], top5_words))

#Barack Obama’s 2016 State of the Union Address Example.
2016년 오바마연설 예제
#Summarize the number of negative words, positive words, sentiments, year, and president name. 부정단어, 긍정단어의 수, 정서, 년도, 대통령이름을 요약
sent = pd.read_csv('sentiment.csv')
sent_tab = pd.merge(df, sent, on='word')
sent_summary = sent_tab.groupby('sentiment').count()['word']
positive = sent_summary[1]
negative = sent_summary[0]
sent_score = positive - negative
print('{}; {}; {}; {}; {}'.format(negative, positive, sent_score, meta.year[235], meta.president[235]))

from textblob import TextBlob
obama_address = TextBlob(text)
obama_address.sentiment

from konlpy.tag import Okt
okt = Okt()

#import os
#os.environ['JAVA_HOME'] = 'C:\\Program Files\\Java\\jdk1.8.0_241'
#os.environ







