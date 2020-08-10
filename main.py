# -*- coding: utf-8 -*-
"""
Created on Sat May 23 11:38:45 2020

@author: 원기
main.py
"""

#관련 Library import
import numpy as np
import json
import matplotlib.pylab as plt
import sys
sys.path.append('C:/Users/원기/Korean/')
from PIL import Image
from keras.utils import to_categorical
from keras import models, layers

#Image Data Open
with open('C:/Users/원기/Korean/handwriting_data_info1.json','r',encoding='utf-8') as f:
    Korean_data_hand = json.load(f)

with open('C:/Users/원기/Korean/printed_data_info.json','r',encoding='utf-8') as f:
    Korean_data_print = json.load(f)

with open('C:/Users/원기/Korean/Korean_2350.txt','r',encoding='utf-8') as f:
    Korean_2350 = f.read()
    
Korean_2350 = Korean_2350.split()
    
#Training을 위한 Data Save
Korean_id = []
Korean_text = []

for i,Data in enumerate(Korean_data_hand['annotations']):
    if Data['attributes']['type']== '글자(음절)':
        if Data['text'] in Korean_2350:
            Korean_id.extend([Data['id']])
            Korean_text.extend([Data['text']])
            
Korean_id = Korean_id[0:28345]+Korean_id[29602:-631]
Korean_text = Korean_text[0:28345]+Korean_text[29602:-631]

for i,Data in enumerate(Korean_data_print['annotations']):
    if Data['attributes']['type']== '글자(음절)':
        if Data['text'] in Korean_2350:
            Korean_id.extend([Data['id']])
            Korean_text.extend([Data['text']])

#메모리 확보를 위해 file Data 정리
del(Korean_data_hand)
del(Korean_data_print)
del(Korean_2350)

#Train & test data Split
Korean_train_id = Korean_id[0:240000]
Korean_train_text = Korean_text[0:240000]

Korean_test_id = Korean_id[240000:264385]
Korean_test_text = Korean_text[240000:264385]

syllable = list(set(Korean_text))
print('"음절 총 개수: ', len(syllable))

#Training에 필요한 만큼 Data가 있는지 확인
sample=0
for Data in Korean_text:
    if Data == '최':
        sample+=1
print('"최"라는 단어를 사용한 횟수: ', sample)

#메모리 확보를 위해 file Data 정리
del(Korean_id)
del(Korean_text)

#Image File 불러오기
x_train = np.zeros((240000,32,32,3),'uint8')
x_test = np.zeros((24385,32,32,3),'uint8')

for i, ID in enumerate(Korean_train_id):
    if i < 28345:
        Image_addr = 'C:/Users/원기/Korean/01_handwriting_syllable_images/1_syllable/' \
            +str(ID)+'.png'
    elif i < 152432:
        Image_addr = 'C:/Users/원기/Korean/2_syllable/' \
            +str(ID)+'.png'
    else:
        Image_addr = 'C:/Users/원기/Korean/syllable/' \
            +str(ID)+'.png'        
    Korean_Image = Image.open(Image_addr)
    Korean_Image = Korean_Image.resize((32,32))
    Korean_Image_Array = np.array(Korean_Image,'uint8')
    x_train[i] = Korean_Image_Array

for i,ID in enumerate(Korean_test_id):
    Image_addr = 'C:/Users/원기/Korean/syllable/' \
        +str(ID)+'.png'
    Korean_Image = Image.open(Image_addr)
    Korean_Image = Korean_Image.resize((32,32))
    Korean_Image_Array = np.array(Korean_Image,'uint8')
    x_test[i] = Korean_Image_Array

#CNN을 위해 array shape 변경 후 0 ~ 1 사이로 값 Normalize
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0


#메모리 확보를 위해 file Data 정리
del(Korean_train_id)
del(Korean_test_id)

#각 음절 Indexing
syllable_to_index = {syllable: index for index, syllable in enumerate(syllable)}
index_to_syllable = {index: syllable for index, syllable in enumerate(syllable)}

label_train = []
label_test = []

for syllables in Korean_train_text:
    if syllable_to_index.get(syllables) is not None:
        label_train.extend([syllable_to_index[syllables]])

for syllables in Korean_test_text:
    if syllable_to_index.get(syllables) is not None:
        label_test.extend([syllable_to_index[syllables]])

#메모리 확보를 위해 file Data 정리
del(Korean_train_text)
del(Korean_test_text)

#One - hot encoding
label_train = to_categorical(label_train)
label_test = to_categorical(label_test)

#CNN layer 추가 후 Parameter 설정
CNN = models.Sequential()
CNN.add(layers.Conv2D(128, (3,3), activation='relu',input_shape=(32,32,3)))
CNN.add(layers.MaxPooling2D((2,2)))

CNN.add(layers.Conv2D(256, (3,3), activation='relu'))
CNN.add(layers.MaxPooling2D((2,2)))

CNN.add(layers.Conv2D(512, (3,3), activation='relu'))
CNN.add(layers.Flatten())

CNN.add(layers.Dense(512, activation='relu'))

CNN.add(layers.Dense(2349, activation='softmax'))

#CNN 함수 설정 후 Training
from keras import optimizers
CNN.compile(optimizer = optimizers.RMSprop(lr=0.001),
loss = 'categorical_crossentropy',
metrics = ['accuracy'])

hist = CNN.fit(x_train, label_train,
epochs = 20, batch_size = 128,
validation_split=0.2)

#test accuracy 출력
test_loss, test_acc = CNN.evaluate(x_test, label_test)
print('test_loss:      ',test_loss)
print('test_accuracy:  ',test_acc)

#Epoch당 loss / accuracy plot
loss = hist.history['loss']
acc = hist.history['accuracy']
val_loss = hist.history['val_loss']
val_acc = hist.history['val_accuracy']
epochs = range(1, len(loss)+1)
plt.figure(figsize=(10,7))
plt.subplots_adjust(wspace=0.5)
plt.subplot(1,2,1)
plt.plot(epochs, loss, 'bo-', label='Training loss')
plt.plot(epochs, val_loss, 'rx-', label='Validation loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.grid()
plt.legend()
plt.subplot(1,2,2)
plt.plot(epochs, acc, 'bo-', label='Training accuracy')
plt.plot(epochs, val_acc, 'rx-', label='Validation accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.grid()
plt.legend()


#Index & model save
#CNN.save('Korean_CNN_model(97.8).h5')

#with open("index_to_syllable(97.8).json", "w") as json_file:
#    json.dump(index_to_syllable, json_file)