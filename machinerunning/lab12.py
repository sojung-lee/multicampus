import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from wordcloud import WordCloud

import os

os.chdir('E:/TEACHING/TEACHING KOREA/MULTICAMPUS/LAB/LAB12/')

from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)  # 그래프에서 한글사용

text = "고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지."
words = word_tokenize(text)

# from konlpy.tag import Okt
# okt = Okt()
# words = okt.nouns(text)

fdist = FreqDist(words)
fdist.plot()
df = pd.DataFrame({'word': list(fdist.keys()),
                   'freq': list(fdist.values())})
df.sort_values('freq', ascending=False)

stop_words = '게 때는 아무렇게나 다 돼'
stop_words = stop_words.split(' ')
filter_words = [w for w in words if (w not in stop_words) & (len(w) >= 2)]
text1 = ' '.join(filter_words)
wordcloud = WordCloud(font_path='C:/USERS/KIM/APPDATA/LOCAL/MICROSOFT/WINDOWS/FONTS/DXRMBXB-KSCPC-EUC-H.TTF').generate(
    text1)
plt.imshow(wordcloud)
plt.show()

# BOW Encoding
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import RegexpTokenizer

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(ngram_range=(1, 1), lowercase=True, stop_words='english', tokenizer=token.tokenize)
text = ['This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?']
x = cv.fit_transform(text)
x_df = pd.DataFrame(x.toarray(), columns=cv.get_feature_names())

text = "Think and wonder, wonder and think"
token = RegexpTokenizer(r'\w+')
token.tokenize(text)
import re

re.split('\W+', text)

# TFIDF ENCODING
text = ['This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?']
tfidf = TfidfVectorizer()
x = tfidf.fit_transform(text)
x_df = pd.DataFrame(x.toarray(), columns=tfidf.get_feature_names())

# bayesian model
# weahter example
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('PlayTennis.csv')
tennis = df.copy()
tennis.head()
tennis.dtypes

# encoding
le = LabelEncoder()
for col in tennis.columns:
    tennis[col] = le.fit_transform(tennis[col])

# x and y split
y = tennis['Play Tennis']
x = tennis.drop('Play Tennis', axis=1)

# train and test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=2)
len(x_train), len(x_test)

# model building
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
from sklearn.metrics import accuracy_score

gnb_score = accuracy_score(y_test, y_pred)

bnb = BernoulliNB()
bnb.fit(x_train, y_train)
y_pred = bnb.predict(x_test)
from sklearn.metrics import accuracy_score

bnb_score = accuracy_score(y_test, y_pred)

mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_pred = mnb.predict(x_test)
from sklearn.metrics import accuracy_score

mnb_score = accuracy_score(y_test, y_pred)

# comparison
print(gnb_score, bnb_score, mnb_score)

# precition with test data
x_train[:3]
test_data = pd.DataFrame([[2, 2, 1, 0]], columns=tennis.columns[:4])
gnb.predict(test_data)

# fruit example
fruit = pd.read_csv('fruit_data_with_colors.txt', delimiter='\t')
fruit.head()
fruit.dtypes

# x and y split
y = fruit.fruit_label
x = fruit[['mass', 'width', 'height', 'color_score']]

x.boxplot()
plt.show()

# scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x = scaler.fit_transform(x)
pd.DataFrame(x).boxplot()

# train and test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=2)

# model building
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
from sklearn.metrics import accuracy_score

gnb_score = accuracy_score(y_test, y_pred)

bnb = BernoulliNB()
bnb.fit(x_train, y_train)
y_pred = bnb.predict(x_test)
from sklearn.metrics import accuracy_score

bnb_score = accuracy_score(y_test, y_pred)

mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_pred = mnb.predict(x_test)
from sklearn.metrics import accuracy_score

mnb_score = accuracy_score(y_test, y_pred)

# comparison
print(gnb_score, bnb_score, mnb_score)

# precition with test data
test_data = x_train[:1]
gnb.predict(test_data)

# text classification
sent = pd.read_csv('sent_train.tsv', delimiter='\t')
sent.head()
sent.dtypes

plt.style.available
plt.style.use('ggplot')
tab = sent.groupby('Sentiment').count()
plt.bar(tab.index, tab.PhraseId)
plt.show()

# x and y split
y = sent.Sentiment
x = sent.Phrase

# BOA encoding
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
x = cv.fit_transform(x)
x_df = pd.DataFrame(x.toarray(), columns=cv.get_feature_names())

# train and test plit
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=3)

# model building
# gnb = GaussianNB()
# gnb.fit(x_train, y_train)
# y_pred = gnb.predict(x_test)
# from sklearn.metrics import accuracy_score
# gnb_score = accuracy_score(y_test, y_pred)

bnb = BernoulliNB()
bnb.fit(x_train, y_train)
y_pred = bnb.predict(x_test)
from sklearn.metrics import accuracy_score

bnb_score = accuracy_score(y_test, y_pred)

mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_pred = mnb.predict(x_test)
from sklearn.metrics import accuracy_score

mnb_score = accuracy_score(y_test, y_pred)

# comparison
print(bnb_score, mnb_score)

# prediction with test data
test_data = pd.DataFrame({'Phrase': 'hi how are you??'}, index=[0])
test_data = cv.transform(test_data)
bnb.predict(test_data)
mnb.predict(test_data)

# tfidf encoding

# college example
college = pd.read_csv('College.csv')
college.head()
college.dtypes
college.set_index('Unnamed: 0', inplace=True)

# x and y split
y = college.Private
x = college.drop('Private', axis=1)
x.boxplot()
plt.show()

# Normalize x before building model
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
plt.boxplot(x)
plt.show()

# Encode y
le = LabelEncoder()
y = le.fit_transform(y)

# Train and test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=4)

# Build a MLP model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import r2_score

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), random_state=1, max_iter=500)
mlp.fit(x_train, y_train)
y_pred = mlp.predict(x_test)
mlp_score = accuracy_score(y_test, y_pred)
# r2_score(y_test, y_pred)

# Build a and NN model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=90, validation_data=(x_test, y_test))
y_pred = model.predict_classes(x_test)
model.evaluate(x_test, y_test)
nn_score = accuracy_score(y_test, y_pred)  # test performance

# prediction with test data
test_data = college.drop('Private', axis=1)[:1]
test_data = scaler.transform(test_data)
model.predict_classes(test_data)
le.inverse_transform(model.predict_classes(test_data))

# evaluate the results
print(mlp_score, nn_score)

# prediction using mlp classifier
mlp.predict(test_data)