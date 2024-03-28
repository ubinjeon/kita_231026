# 이미지를 읽고 출력하는 다양한 방법

import urllib.request as req # urllib.request 모듈은 웹사이트 데이터 접근
import cv2 # OpenCV import

## 1)urlretrieve 후 openCV로 읽고 출력하는 방법
req.urlretrieve(url, 'lady.png') # download 후 현위치에 저장

img = cv2.imread('lady.png') # openCV로 읽기
### 1번째 출력 방법: matplotlib 사용
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))<br>
plt.show()
### 2번째 출력방법: opencv 사용
from google.colab.patches import cv2_imshow <- colab에서 cv2에서 바로 imshow를 쓰려면 patches를 해야함. <br>
cv2_imshow(img)
####부연설명
cv2.cvtColor(img, cv2.COLOR_BGR2RGB) BGR컬러체계에서 RGB컬러체계로 변환<br>
openCV는 컬러체계 BGR<br>
matplotlib은 컬러체계 RGB

## 2)urlopen 후 이미지 출력방법
resp = urllib.request.urlopen(url) # 웹 리소스의 내용을 바이트(byte) 형태로 읽기 <- 저장하는 것은 아님. 자체에서 저장기능 불포함<br>
- bytearray(...) 함수는 바이트열을 가변적인 바이트 배열로 변환
- np.asarray(...) 함수는 주어진 입력(여기서는 bytearray 객체)을 NumPy 배열로 변환
image = np.asarray(bytearray(resp.read()), dtype="uint8") #uint8 부호가 없는. 이미지에 많이 씀.
-  메모리에서 이미지 디코드
img = cv2.imdecode(image, cv2.IMREAD_COLOR) # 메모리상의 이미지 데이터를 디코드
- BGR 이미지를 RGB 이미지로 변환하여 출력
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))<br>
plt.show()

## 3) imageio 라이브러리로 읽고 출력방법
- imageio 라이브러리: 내부적으로 웹상의 이미지 데이터를 바이트 스트림으로 받아와서 이미지 데이터로 변환하는 과정을 처리<br>
import imageio<br>
from matplotlib import pyplot as plt<br>
img = image.io.imread(url)<br>
plt.imshow(img)<br>
plt.show()<br>

## 4) PIL (파이썬 이미지 처리 라이브러리) 사용해서 이미지 출력방법
from PIL import Image<br>
from io import BytesIO<br>
import requests<br>
response = requests.get(url)<br>
- 바이너리 데이터로부터 직접 이미지 객체를 생성.
pic = Image.open(BytesIO(response.content)) # BytesIO는 바이너리 데이터를 읽고 쓰는데 사용되는 파일과 유사한 객체로, 메모리 내에서만 작동<br>
pic<br>

# OpenCV 도형그리기
## 바탕색 채우기
import numpy as np
img = np.zeros((512,512,3), np.uint8)
img.fill(255) # 흰색으로 채우기, 이 코드 없으면 전부 0이니깐 검정색 배경
plt.imshow(img)
plt.show()

## 선 그리기 cv2.line
img = cv2.line(img,(0,0),(511,511),(255,0,255),3)
### line 입력변수
- img: 선을 그릴 이미지
- (0,0): starting point
- (511,511): ending point
- (255,0,255):BGR color values. In this case, magenta
- 3: 선 굵기(픽셀단위)

##사각형 그리기 cv2.rectangle
img = cv2.rectangle(img,(200,300),(300,200),(255,255,0),5)
## #사각형 입력변수
- img: 사각형을 그릴 이미지
- (200,300): top-left corner point (x,y)
- (300,200): bottom-right corner point (x,y)
- (255,255,0): BGR color values. In this case, cyan
- 5: 선 굵기(픽셀단위)

##원 그리기 cv2.circle
img = cv2.circle(img,(250,250),100,(0,255,255),10)
- img: 원 그릴 이미지
- (250,250): 원의 중심점 좌표
- 100: 원의 반지름
- (0,255,255): 원의 색상
- 10: 선의 두께. -1 이면 원의 내부가 채워진다.

## 타원 그리기 cv2.ellipse
img = cv2.ellipse(img,(350,150),(100,50),310,310,60,(0,255,255),-1)
- cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)

## 다각형 그리기 cv2.polylines
- 다각형, pts는 연결할 꼭지점 좌표, isClosed - 닫힌 도형 여부
pts = np.array([[10,5],[20,30],[70,20],[50,10]],np.int32) # 다각형을 그리기 위한 꼭짓점 좌표를 정의. 이 배열은 4개의 점을 가지며, 각 점은 (x, y) 형태의 좌표<br>
print(pts.shape)<br>
pts = pts.reshape((-1,2,1)) #  각 좌표가 별도의 차원을 가지도록 배열의 형태를 변경<br>
print(pts.shape)<br>
img = cv2.polylines(img,[pts],True, (172,200,255),4)<br>
plt.imshow(img)<br>
plt.show()<br>

## 글씨 넣기 cv2.putText
img = cv2.putText(img, 'OpenCV',(10,500),cv2.FONT_HERSHEY_SIMPLEX,4,(255,255,255),3)<br>
- 'OpenCV': 넣을 글씨
- (10, 500): 텍스트가 시작될 위치의 좌표
- cv2.FONT_HERSHEY_SIMPLEX: 텍스트에 사용될 글꼴 스타일
- 4: 글꼴 크기
- (255,255,255): 텍스트의 색상
- 3: 텍스트의 두께

# Hue, Saturation, Value(검정에서 백색까지의 범위), Lightness(어두움과 빛의 크기)
- HSV: Hue(0~179 정수), Saturation(0~255 정수), Value(0~255 정수)
- HSL: Hue, Saturation, Lightness
- cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
- cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
- cv2.cvtColor(origin_img, cv2.COLOR_BGR2HLS)

# YCrCb Color Space
- Y 성분은 밝기 또는 휘도(luminance),Cr,Cb 성분은 색상 또는 색차를 나타냄
- Cr, Cb는 오직 색상 정보만 가지고 있음. 밝기 정보 없음
- 영상을 GrayScale 정보와 색상 정보로 분리하여 처리할 때 유용
- 0 ~ 255 사이의 정수로 표현
- cv2.cvtColor(origin_img, cv2.COLOR_BGR2YCrCb)

# GrayScale Color Space
- 영상의 밝기 정보를 256단계(0 ~ 255)로 구분하여 표현
- 가장 밝은 흰색 : 255
- 가장 어두운 검은색 : 0
- cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)

# cv2.imread(fileName, flag)
- cv2.IMREAD_COLOR 또는 1: 이미지를 컬러로 읽습니다. 이 경우 알파 채널은 무시됩니다. 컬러 이미지는 기본적으로 BGR 색상 공간으로 읽힙니다.
- cv2.IMREAD_GRAYSCALE 또는 0: 이미지를 그레이스케일로 읽습니다. 즉, 색상 정보 없이 밝기 정보만을 가진 단일 채널 이미지로 변환됩니다.
- cv2.IMREAD_UNCHANGED 또는 -1: 이미지를 원본 그대로 읽습니다,

# cv2.calcHist(images, channels, maskm histSize, ranges, hist=None, accumulate=None)
- images: 입력 영상 리스트
- channels: 히스토그램을 구할 채널을 나타내는 리스트
- mask: 마스크 영상. 입력 영상 전체에서 히스토그램을 구하려면 None 지정
- histSize: 히스토그램 각 차원의 크기(빈(bin)의 개수)를 나타내는 리스트
- ranges: 히스토그램 각 차원의 최솟값과 최댓값으로 구성된 리스트
- hist: 계산된 히스토그램 (numpy.ndarray)
- accumulate: 기존의 hist 히스토그램에 누적하려면 True, 새로 만들려면 False
- hist1 = cv2.calcHist([img1],[0],None,[256],[0,256])
- [0]: 그레이 스케일이미지
- None: 마스크를 사용하지 않음
- [256]: bin 개수,
- [0,256]: 히스토그램 픽셀값 범위

# cv2.resize (이미지,(직접 지정할 크기), fx=현재 이미지 가로의 몇배, fy, = 현재 이미지 세로의 몇배, interpolation)
- cv2.resize(image,dsize=(320,200), interpolation=cv2.INTER_AREA)
- cv2.INTER_AREA: 사이즈 줄일 때
- cv2.INTER_CUBIC: 사이즈 늘릴 때 (16개 픽셀 사용. LINEAR 보다 속도 느림)
- cv2.INTER_LINEAR: 사이즈 늘릴때 (4개 픽셀 사용. default)

# 이미지 임계처리(주로 병변 확인시)
## Simple thresholding, Adaptive thresholding, Otsu’s thresholding
## Simple thresholding
- src – input image로 single-channel 이미지.(grayscale 이미지)
- thresh – 임계값
- maxval – 임계값을 넘었을 때 적용할 value
- type – thresholding type
### 예제
- 이진화란 영상을 흑/백으로 분류하여 처리하는 것.
- 지정된 임계값을 기준으로 픽셀 값을 0(검은색) 또는 최대 값(여기서는 127)으로 설정하여 이미지의 이진화를 수행
- 127: 임계값(threshold)으로 사용됩니다. 이 값보다 큰 픽셀 값은 모두 최대 값으로 설정되고, 이 값 이하인 픽셀 값은 0으로 설정
_, bin_image = cv2.threshold(gray_image,127,127,cv2.THRESH_BINARY)<br>
cv2_imshow(bin_image)<br>

# 비트연산
bit_and = cv2.bitwise_and(img1_resized, img2) # and 연산 : 두 이미지에서 모두 흰색인 부분만 흰색<br>
bit_or = cv2.bitwise_or(img1_resized, img2) # or 연산 : 두 이미지 중 하나에서만 흰색이여도 그 부분을 흰색<br>
bit_not = cv2.bitwise_not(img2)  # not 연산 : 해당 이미지의 반대<br>
bit_xor = cv2.bitwise_xor(img1_resized, img2) # xor 연산 : 두 이미지에서 값이 서로 같으면 검은색, 같지 않으면 흰색<br>

# 이미지 Region of Image(ROI)
 Numpy의 indexing을 사용 , 특정 영역을 copy할 수도 있음
