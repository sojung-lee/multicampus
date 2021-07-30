import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import fashion_mnist
import os
os.chdir('E:/TEACHING/TEACHING KOREA/MULTICAMPUS/LAB/LAB18/')

#Load fashion_mnist dataset and split it into train and test set (keras.datasets.fashion_mnist). 패션이미지를 가지고 와서 훈련, 테스트세트로 분리
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
y_test_copy = y_test.copy()
#Exploratory data analysis (imshow of 9 images of train and the 9 images of test, shape of train and test, and number of classes and names of classes). 데이터 탐색
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[i], cmap='gray')
plt.show()
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[i], cmap='gray')
plt.show()
x_train.shape, y_train.shape, x_test.shape, y_test.shape
print('number of class is {}'.format(len(np.unique(y_train))))
print('names of class are {}'.format(np.unique(y_train)))
set(y_train)
#0 T-shirt/top
#1 Trouser
#2 Pullover
#3 Dress
#4 Coat
#5 Sandal
#6 Shirt
#7 Sneaker
#8 Bag
#9 Ankle boot
sns.countplot(y_train)
plt.show()

#Data preprocessing (reshape, divided by 255 to normalize, encoding y) 전처리
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
x_train.shape, x_test.shape
x_train, x_test = x_train/255.0, x_test/255.0
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Build a CNN model. 컨볼루션 신경망 구축
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

x_train.shape
model = Sequential()
#Conv2d with 32 nodes, 3x3 filters, relu activation function
model.add(Conv2D(32, (3,3),activation='relu', input_shape=(28,28,1), padding='same'))
#Maxpooling with 2x2 filers
model.add(MaxPooling2D((2,2), padding='same'))
#Dropout .25
model.add(Dropout(.25))
#Conv2d with 64 nodes, 3x3 filters, relu activation function
model.add(Conv2D(64, (3,3),activation='relu', padding='same'))
#Maxpooling with 2x2 filers
model.add(MaxPooling2D((2,2),padding='same'))
#Dropout .25
model.add(Dropout(.25))
#Conv2d with 128 nodes, 3x3 filters, relu activation function
model.add(Conv2D(128, (3,3),activation='relu', padding='same'))
#Maxpooling with 2x2 filers
model.add(MaxPooling2D((2,2),padding='same'))
#Dropout .25
model.add(Dropout(.25))
#Flatten
model.add(Flatten())
#Dense with 128 nodes with relu activation
model.add(Dense(128, activation='relu'))
#Dropout .3
#Dense with softmax function
model.add(Dense(10,activation='softmax'))

#Compile the model. 컴파일 모델
#loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
model.summary()

#Fit the model. 모델훈련
#batch_size=64, epochs=5, verbose=1
history = model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1, validation_data=(x_test, y_test))

#Prediction. 예측
model.predict(x_test).round() #인코딩값으로 출력
y_pred = model.predict_classes(x_test) #클래스 이름으로 출력
y_test_copy

#Evaluate the model. 모델평가
#Loss and accuracy score (model.evaluate(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=1) #91.52% accuracy
model.metrics_names #90.93% accuracy with dropout

#Training and validation accuracy plot and Training and validation loss plot
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()

#Classification report (classification_report(y_test, y_pred))
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test_copy, y_pred))

sns.heatmap(confusion_matrix(y_test_copy, y_pred), annot=True, fmt='d')
len(np.where(y_test_copy == y_pred)[0]) #9152 correctly classified / #9093 with dropout
len(np.where(y_test_copy != y_pred)[0]) #848 incorrectly classified / 907 with dropout

#COLLEAGE DATA EXAMPLE
#Load College data (pd.read_csv). 데이터로링
college = pd.read_csv('College.csv')

#Data exploration (head, describe, isna.sum(), set_index). 데이터탐색
college.head()
college.shape
a = college.describe()
college.isna().sum()
college.dtypes
college.rename(columns={'Unnamed: 0':'University'}, inplace=True)
college.set_index('University', inplace=True)

college.boxplot()
plt.show()
college.hist()
plt.tight_layout()
plt.show()
pd.plotting.scatter_matrix(college)
plt.show()
sns.heatmap(college.corr(), annot=True)
#%matplotlib auto
len(college.columns)
cols = college.columns[1:]
fig, axs = plt.subplots(6,3)
cnt = 0
for i in range(6):
    for j in range(3):
        sns.kdeplot(cols[cnt], data=college, hue='Private', ax=axs[i,j])
        cnt+=1
plt.show()

#X and y split. 독립, 종속변수 분리
y = college.Private
x = college.drop('Private', axis=1)
x.boxplot()
plt.show()
sns.countplot(y)
#Data preprocessing (y - LabelEncoder, x - StandardScaler). 데이터전처리
from sklearn.preprocessing import LabelEncoder, StandardScaler
le = LabelEncoder()
y = le.fit_transform(y) #y값 인코딩
scaler = StandardScaler()
x = scaler.fit_transform(x) #x값을 정규화
plt.boxplot(x)
plt.show()

#Train and test split (sklearn.model_selection.train_test_split). 훈련, 테스트데이터 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2, random_state=0)
x_train.shape, x_test.shape

#Build a NN Model (First layer: 12 nodes, with relu activation, 2nd layer: 8 nodes, with relu activation, 3rd layer: output, with sigmoid activation). 신경망모델 구축
#Use init='uniform‘ for weight and bias initializers.
np.random.seed(10)
from keras.models import Sequential
from keras.layers import Dense
x_train.shape
model = Sequential()
model.add(Dense(12, activation='relu', input_dim=17, init='uniform'))
model.add(Dense(8, activation='relu', init='uniform'))
model.add(Dense(1, activation='sigmoid', init='uniform'))

#Compile the model (loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']). 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#Fit the model (epochs=1000, batch_size=16). 모델훈련
history = model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_data=(x_test, y_test), verbose=1)

#Prediction and evaluation (model.predict_classes). 예측 및 평가
model.predict(x_test)
y_pred = model.predict_classes(x_test)
score = model.evaluate(x_test, y_test, verbose=1)
model.metrics_names[0], score[0]
model.metrics_names[1], score[1] #91.67%

#Accuracy and loss chart. 정확도, 로스그래프
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
print(classification_report(y_test, y_pred))

######################################################################
#boston housing dataset
from sklearn.datasets import load_boston
boston = load_boston()
x, y = boston.data, boston.target
plt.boxplot(x)
plt.show()

from sklearn.preprocessing import StandardScaler
x_s = x.copy()
s = StandardScaler()
x = s.fit_transform(x)
x.mean() #0
x.std() #1

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2, random_state=1)

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=13, kernel_initializer='normal'))
model.add(Dense(64, activation='relu', kernel_initializer='normal'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

from keras.callbacks import ModelCheckpoint
ckpt_model = 'boston-weights.best.hdf5'
checkpoint = ModelCheckpoint(ckpt_model,
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True,
                            mode='max')
history = model.fit(x_train, y_train, epochs=100,
                     validation_data=(x_test, y_test),
                     callbacks=[checkpoint])

y_pred = model.predict(x_test)
y_test
model.metrics_names, model.evaluate(x_test, y_test)
((y_pred.T-y_test)**2).mean() #MSE

#history graphs
plt.plot(history.history['mse'], label='train')
plt.plot(history.history['val_mse'], label='test')
plt.legend()
plt.show()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

#prediction with test data
test_data = np.array([0.17, 0, 5.96, 0, 0.49, 5.96, 30.20, 3.84, 5, 279, 19.2, 393.43, 10.13])
test_data = test_data.reshape(1,-1)
test_data = s.fit_transform(test_data)
model.predict(test_data)