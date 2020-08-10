# -*- coding: utf-8 -*-
"""
Created on Tue May 19 01:58:27 2020

@author: 원기
"""
import numpy as np
import json
from gensim.models.word2vec import Word2Vec
from keras.utils import to_categorical

# 태그 단어
PAD = "<PADDING>"   # 패딩
STA = "<START>"     # 시작
END = "<END>"       # 끝
OOV = "<OOV>"       # 없는 단어(Out of Vocabulary)

# 태그 인덱스
PAD_INDEX = 0
STA_INDEX = 1
END_INDEX = 2
OOV_INDEX = 3
# 데이터 타입
ENCODER_INPUT  = 0
DECODER_INPUT  = 1
DECODER_TARGET = 2

# 한 문장에서 단어 시퀀스의 최대 개수
max_sequences = 50

with open('C:/Users/원기/Korean/KorQuAD_v1.0_train.json','r') as f:
    Korean_arrays = json.load(f)

x_train = []
label_train = []


for i in range(len(Korean_arrays['data'])):
#for i in range(10):
    for j in range(len(Korean_arrays['data'][i]['paragraphs'])):
        for k in range(len(Korean_arrays['data'][i]['paragraphs'][j]['qas'])):
            x_train.append(Korean_arrays['data'][i]['paragraphs'][j]['qas'][k]['question'])

for i in range(len(Korean_arrays['data'])):
#for i in range(10): 
    for j in range(len(Korean_arrays['data'][i]['paragraphs'])):
        for k in range(len(Korean_arrays['data'][i]['paragraphs'][j]['qas'])):
            label_train.append(Korean_arrays['data'][i]['paragraphs'][j]['qas'][k]['answers'][0]['text'])

from konlpy.tag import Twitter


Twitter = Twitter() 
words = []
data = []
Q=[]
A=[]
QnA=[]

for i in range(len(x_train)):
    data = Twitter.morphs(x_train[i])
    Q.append(data)
    for j in range(len(data)):
        words.append(data[j])
    data = Twitter.morphs(label_train[i])
    A.append(data)
    for j in range(len(data)):
        words.append(data[j])
    
words = list(set(words))
words[:0] = [PAD, STA, END, OOV]

for i in range(len(Q)):
    QnA.append(Q[i]+A[i])

max([len(line) for line in Q])
max([len(line) for line in A])
max([len(line) for line in QnA])


# 단어와 인덱스의 딕셔너리 생성
word_to_index = {word: index for index, word in enumerate(words)}
index_to_word = {index: word for index, word in enumerate(words)}

# 문장을 인덱스로 변환
def convert_text_to_index(sentences, vocabulary, type): 
    
    sentences_index = []
    
    # 모든 문장에 대해서 반복
    for sentence in sentences:
        sentence_index = []
        
        # 디코더 입력일 경우 맨 앞에 START 태그 추가
        if type == DECODER_INPUT:
            sentence_index.extend([vocabulary[STA]])
        
        # 문장의 단어들을 띄어쓰기로 분리
        for word in sentence:
            if vocabulary.get(word) is not None:
                # 사전에 있는 단어면 해당 인덱스를 추가
                sentence_index.extend([vocabulary[word]])
            else:
                # 사전에 없는 단어면 OOV 인덱스를 추가
                sentence_index.extend([vocabulary[OOV]])

        # 최대 길이 검사
        if type == DECODER_TARGET:
            # 디코더 목표일 경우 맨 뒤에 END 태그 추가
            if len(sentence_index) >= max_sequences:
                sentence_index = sentence_index[:max_sequences-1] + [vocabulary[END]]
            else:
                sentence_index += [vocabulary[END]]
        else:
            if len(sentence_index) > max_sequences:
                sentence_index = sentence_index[:max_sequences]
            
        # 최대 길이에 없는 공간은 패딩 인덱스로 채움
        sentence_index += (max_sequences - len(sentence_index)) * [vocabulary[PAD]]
        
        # 문장의 인덱스 배열을 추가
        sentences_index.append(sentence_index)

    return np.asarray(sentences_index)

# 인코더 입력 인덱스 변환
x_encoder = convert_text_to_index(Q, word_to_index, ENCODER_INPUT)

# 디코더 입력 인덱스 변환
x_decoder = convert_text_to_index(A, word_to_index, DECODER_INPUT)

# 디코더 목표 인덱스 변환
y_decoder = convert_text_to_index(A, word_to_index, DECODER_TARGET)

# 원핫인코딩 초기화
one_hot_data = np.zeros((len(y_decoder), max_sequences, len(words)))

# 디코더 목표를 원핫인코딩으로 변환
# 학습시 입력은 인덱스이지만, 출력은 원핫인코딩 형식임
for i, sequence in enumerate(y_decoder):
    for j, index in enumerate(sequence):
        one_hot_data[i, j, index] = 1

# 디코더 목표 설정
y_decoder = one_hot_data

#from gensim.models import Word2Vec
#model = Word2Vec(QnA, size=100, window=5, min_count=5, workers=4, sg=0)
#model.wv.most_similar("파리")
#model.wv.vectors.shape

from keras import models
from keras import layers
from keras import optimizers, losses, metrics
from keras import preprocessing

# 임베딩 벡터 차원
embedding_dim = 100

# LSTM 히든레이어 차원
lstm_hidden_dim = 128
#--------------------------------------------
# 훈련 모델 인코더 정의
#--------------------------------------------

# 입력 문장의 인덱스 시퀀스를 입력으로 받음
encoder_inputs = layers.Input(shape=(None,))

# 임베딩 레이어
encoder_outputs = layers.Embedding(len(words), embedding_dim)(encoder_inputs)

# return_state가 True면 상태값 리턴
# LSTM은 state_h(hidden state)와 state_c(cell state) 2개의 상태 존재
encoder_outputs, state_h, state_c = layers.LSTM(lstm_hidden_dim,
                                                dropout=0.1,
                                                recurrent_dropout=0.5,
                                                return_state=True)(encoder_outputs)

# 히든 상태와 셀 상태를 하나로 묶음
encoder_states = [state_h, state_c]



#--------------------------------------------
# 훈련 모델 디코더 정의
#--------------------------------------------

# 목표 문장의 인덱스 시퀀스를 입력으로 받음
decoder_inputs = layers.Input(shape=(None,))

# 임베딩 레이어
decoder_embedding = layers.Embedding(len(words), embedding_dim)
decoder_outputs = decoder_embedding(decoder_inputs)

# 인코더와 달리 return_sequences를 True로 설정하여 모든 타임 스텝 출력값 리턴
# 모든 타임 스텝의 출력값들을 다음 레이어의 Dense()로 처리하기 위함
decoder_lstm = layers.LSTM(lstm_hidden_dim,
                           dropout=0.1,
                           recurrent_dropout=0.5,
                           return_state=True,
                           return_sequences=True)

# initial_state를 인코더의 상태로 초기화
decoder_outputs, _, _ = decoder_lstm(decoder_outputs,
                                     initial_state=encoder_states)

# 단어의 개수만큼 노드의 개수를 설정하여 원핫 형식으로 각 단어 인덱스를 출력
decoder_dense = layers.Dense(len(words), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)



#--------------------------------------------
# 훈련 모델 정의
#--------------------------------------------

# 입력과 출력으로 함수형 API 모델 생성
model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 학습 방법 설정
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#--------------------------------------------
#  예측 모델 인코더 정의
#--------------------------------------------

# 훈련 모델의 인코더 상태를 사용하여 예측 모델 인코더 설정
encoder_model = models.Model(encoder_inputs, encoder_states)



#--------------------------------------------
# 예측 모델 디코더 정의
#--------------------------------------------

# 예측시에는 훈련시와 달리 타임 스텝을 한 단계씩 수행
# 매번 이전 디코더 상태를 입력으로 받아서 새로 설정
decoder_state_input_h = layers.Input(shape=(lstm_hidden_dim,))
decoder_state_input_c = layers.Input(shape=(lstm_hidden_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]    

# 임베딩 레이어
decoder_outputs = decoder_embedding(decoder_inputs)

# LSTM 레이어
decoder_outputs, state_h, state_c = decoder_lstm(decoder_outputs,
                                                 initial_state=decoder_states_inputs)

# 히든 상태와 셀 상태를 하나로 묶음
decoder_states = [state_h, state_c]

# Dense 레이어를 통해 원핫 형식으로 각 단어 인덱스를 출력
decoder_outputs = decoder_dense(decoder_outputs)

# 예측 모델 디코더 설정
decoder_model = models.Model([decoder_inputs] + decoder_states_inputs,
                      [decoder_outputs] + decoder_states)

def convert_index_to_text(indexs, vocabulary): 
    
    sentence = ''
    
    # 모든 문장에 대해서 반복
    for index in indexs:
        if index == END_INDEX:
            # 종료 인덱스면 중지
            break;
        if vocabulary.get(index) is not None:
            # 사전에 있는 인덱스면 해당 단어를 추가
            sentence += vocabulary[index]
        else:
            # 사전에 없는 인덱스면 OOV 단어를 추가
            sentence.extend([vocabulary[OOV_INDEX]])
            
        # 빈칸 추가
        sentence += ' '

    return sentence

for epoch in range(20):
    print('Total Epoch :', epoch + 1)

    # 훈련 시작
    history = model.fit([x_encoder, x_decoder],
                        y_decoder,
                        epochs=100,
                        batch_size=64,
                        verbose=0)
    
    # 정확도와 손실 출력
    print('accuracy :', history.history['accuracy'][-1])
    print('loss :', history.history['loss'][-1])
    
    # 문장 예측 테스트
    # (3 박 4일 놀러 가고 싶다) -> (여행 은 언제나 좋죠)
    input_encoder = x_encoder[3].reshape(1, x_encoder[3].shape[0])
    input_decoder = x_decoder[3].reshape(1, x_decoder[3].shape[0])
    results = model.predict([input_encoder, input_decoder])
    
    # 결과의 원핫인코딩 형식을 인덱스로 변환
    # 1축을 기준으로 가장 높은 값의 위치를 구함
    indexs = np.argmax(results[0], 1) 
    
    # 인덱스를 문장으로 변환
    sentence = convert_index_to_text(indexs, index_to_word)
    print(sentence)
    print()

def make_predict_input(sentence):

    sentences = []
    sentences.append(sentence)
    #sentences = pos_tag(sentences)
    
    input_seq = convert_text_to_index(sentences, word_to_index, ENCODER_INPUT)
    
    return input_seq

def generate_text(input_seq):
    
    # 입력을 인코더에 넣어 마지막 상태 구함
    states = encoder_model.predict(input_seq)

    # 목표 시퀀스 초기화
    target_seq = np.zeros((1, 1))
    
    # 목표 시퀀스의 첫 번째에 <START> 태그 추가
    target_seq[0, 0] = STA_INDEX
    
    # 인덱스 초기화
    indexs = []
    
    # 디코더 타임 스텝 반복
    while 1:
        # 디코더로 현재 타임 스텝 출력 구함
        # 처음에는 인코더 상태를, 다음부터 이전 디코더 상태로 초기화
        decoder_outputs, state_h, state_c = decoder_model.predict(
                                                [target_seq] + states)

        # 결과의 원핫인코딩 형식을 인덱스로 변환
        index = np.argmax(decoder_outputs[0, 0, :])
        indexs.append(index)
        
        # 종료 검사
        if index == END_INDEX or len(indexs) >= max_sequences:
            break

        # 목표 시퀀스를 바로 이전의 출력으로 설정
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = index
        
        # 디코더의 이전 상태를 다음 디코더 예측에 사용
        states = [state_h, state_c]

    # 인덱스를 문장으로 변환
    sentence = convert_index_to_text(indexs, index_to_word)
        
    return sentence