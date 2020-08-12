# Korean-OCR-Model-Design-based-on-Keras-CNN(한글 OCR 모델 설계)
 한글의 경우 알파벳, 숫자와 달리 적은 데이터의 수, 이에 비해 많은 음절의 수로 인하여 OCR 인식률이 상대적으로 떨어져 발전에 한계가 있었다. 본 논문에서는 Keras CNN, 정수 인코딩을 이용하여 한글 OCR 프로그램을 설계했다. 학습에 필요한 양질의 데이터를 AI Hub 플랫폼에서 제공하였고, Hyper Parameter을 조절하면서 CNN 모델을 설계하였다. 설계 이후 고찰을 통해서, 향후 한글 OCR 기술이 발전하기 위한 방향을 제시하였다.<br>
 ![image](https://user-images.githubusercontent.com/68767122/89760339-2342a880-db27-11ea-95da-196723db9a98.png)<br>
`※ OCR이란 손글씨나 인쇄 글자를 인식하여 텍스트로 변환하는 기술을 의미`
`※ e-mail : dnjsrl3690@naver.com`
## ①서론
### Keras CNN 구조
![image](https://user-images.githubusercontent.com/68767122/89760971-63eef180-db28-11ea-9cac-8e74de83a485.png)<br>
- Convolutional layer : Edge 등의 각 pixel 간 연관되어 있는 특성을 추출하기 위한 filter 기능을 수행한다. 다수의 데이터를 활용한 학습을 통해서 Filter 의 계수를 획득한다. <br>
- Pooling layer : Subsampling 을 통해서 사이즈를 줄임으로써 Parameter 을 감소시켜 over fitting 문제를 해결한다.<br>
- Relu layer : 0보다 작으면 0으로, 크면 그대로 출력하는 활성화 함수가 포함된 Layer<br>
- Softmax layer : 최종적으로 입력받은 이미지를 특정 한글로 확률 값을 예측하는 Layer<br>
### 한글 OCR(Optical Character Recognition)
우선 한글 자모음을 통해서 만들어지는 음절의 수는 다음과 같이 계산된다.<br>
- 초성 : ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ (19개)<br>
- 중성 : ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ (21개)<br>
- 종성 : ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ (27개)<br>
여기에서 종성이 없는 경우를 포함하여 계산을 하면<br>
- 19 × 21 × ( 27 + 1 ) = 11172자<br>
이 글자를 모두 인식하는 프로그램을 만드는 것은 시간, 메모리 소모가 상당하다. 따라서 실제 발음이 되고 한국어에서 사용빈도가 높은 글자를 추려낸 KS X 1001 완성형 에 포함된 한글 2350 자 만 추려내어 학습을 하기로 하였다.<br>
![image](https://user-images.githubusercontent.com/68767122/89763047-a6b2c880-db2c-11ea-8029-1342e4706732.png)
### 정수 인코딩(Integer Encoding)
- 글자의 경우 그 자체가 Index가 되지 못하기 때문에 글자에 Index 부여
- 이러한 각 글자에 고유한 정수를 mapping시키는 과정이 정수 인코딩(Integer Encoding)
- Dictionary와 enumerate를 조합하여 Encoding, Decoding을 위한 두가지 Dictionary를 생성
- 이러한 과정은 원핫 인코딩(One-Hot encoding) 과정에 앞서 필수적이다.
![image](https://user-images.githubusercontent.com/68767122/89763785-3016ca80-db2e-11ea-9ff0-ce7a82df3bec.png)
## ②설계 과정
### GPU 사용 설정
CPU만을 사용할 시 상당한 시간이 소요된다. 따라서 CUDA Toolkit 및 cuDNN를 설치하여 GPU를 사용하였다.
![image](https://user-images.githubusercontent.com/68767122/89764089-c4812d00-db2e-11ea-82dc-9996c5a98707.png)<br>
` ※ 사용 GPU : GTX 1060 3GB`
### Dataset 선정
- 한글 2350자를 학습하기 위해서는 양질의 데이터가 필요
- AI Hub 플랫폼에서 광대한 양의 한글 Dataset을 보유
- 각 Image 파일에 해당하는 정보를 Dictionary 형식으로 제공
- 빈도 수 높은 <b>264,385개 손글씨 및 인쇄체를 데이터로 선정(학습: 24만개, 평가: 24,385개)</b>
[![image](https://user-images.githubusercontent.com/68767122/89764865-3148f700-db30-11ea-90d7-9df8a151bb5d.png)
](http://www.aihub.or.kr/aidata/133)
### Image & text data Preparation
![image](https://user-images.githubusercontent.com/68767122/89766983-fea0fd80-db33-11ea-9a99-4a612a2bf3f1.png)<br><br>
이미지를 불러온 후 32 x 32 size의 np array 타입으로 압축하였다. 또한 해당 데이터를 0~1 사이의 값으로 Normalize하였다.<br><br>
![image](https://user-images.githubusercontent.com/68767122/89767268-91da3300-db34-11ea-9147-a99dd869e4c0.png)<br><br>
각 텍스트에 맞게 정수 인코딩을 진행한 후 Keras 의 함수 to_categorical 을 이용하여 원 핫 인코딩을 하였다.<br>
### CNN Model 생성 후 Training
- 각 Layer에서 Filter의 수는 시간 및 메모리를 고려하여 정확도를 높이도록 설정
- Optimizer의 경우 RMSprop & Learning rate 0.001에서 최적의 학습효과
- Total Epochs 는 Validation loss를 고려하여 20으로 설정
- 한 Epoch당 약 55초의 시간 소모<br>
![CNN_Model](https://user-images.githubusercontent.com/68767122/89773114-44fb5a00-db3e-11ea-964f-17ac83028d4b.JPG)<br>
![image](https://user-images.githubusercontent.com/68767122/89773595-32cdeb80-db3f-11ea-92ba-8ab4b20663b7.png)<br>
![training_1](https://user-images.githubusercontent.com/68767122/89773422-d2d74500-db3e-11ea-9cc7-82247dccc88e.JPG)
![training_2](https://user-images.githubusercontent.com/68767122/89774140-3e6de200-db40-11ea-9ec6-8040d37c20b8.JPG)
## ③설계 결과
### Loss & Accuracy
![image](https://user-images.githubusercontent.com/68767122/89774552-159a1c80-db41-11ea-9916-3e0d768e21e6.png)
학습을 통해 구한 모델의 스펙 정리:
- Training accuracy: 98.14%
- Training loss : 0.067
- Test accuracy : 97.84%
- Test loss : 0.1666
- Training Time : 55s 286us / Epoch
### 실제 데이터 테스트
![image](https://user-images.githubusercontent.com/68767122/89775006-e3d58580-db41-11ea-8210-9b777b18fe60.png)
### 총 Parameter 개수
![image](https://user-images.githubusercontent.com/68767122/89775107-18e1d800-db42-11ea-8157-f4f6adb85d7d.png)
## ④설계 고찰 및 개선점
1. 98%가량의 정확도 : 낮은 정확도는 아니지만 영어, 숫자에 비해서는 상대적으로 떨어지는 정확도이다. 더 많은 학습 데이터가 필요하다.
1. 모델의 큰 용량(52.5MB) : 한글의 경우 예측을 하기 위한 문자가 2350자 정도이므로 많은 Parameter가 
필요함 정확도를 유지하면서 용량을 감소시킬 방법이 필요하다.
1. 오래 걸리는 학습시간 : 한 Epoch당 1분 가량 소모되어 Hyper Parameter을 여러번 변경시키면서 학습하는 것이 제한되었다. 
Colab의 사용이 필요하다.
