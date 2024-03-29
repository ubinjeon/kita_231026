# NumPy 기초
- 기본 데이터 타입은 ndarray. ndarray
- 배열 생성 및 조작
  - np.array(): 리스트나 튜플로부터 배열 생성
  - np.arange(): 연속된 값으로 배열 생성
  - reshape(): 배열의 형태 변경
- 배열 인덱싱 및 슬라이싱
- NumPy의 수학적 연산
- 부울 인덱싱 및 팬시 인덱싱
- 파일 입출력

## 코드 예제
- import numpy as np
- np.sqrt(array)
- np.arange(10)
- np.array([1,2,3])
- print(data.shape) # 크기
- print(data.dtype) # 자료형
- print(data.ndim)  # 차원
- a = np.arange(10).reshape(2,5)
- zeros, ones, full, eye
  - z = np.zeros_like(a)
  - o = np.ones_like(a)
  - f = np.full_like(a,5)
- array.tolist()
- type(array)
- array6_concat = np.concatenate((array6_1, array6_2),axis=1)
- split1,split2,split3 = np.split(array8,3)
- a_t = np.transpose(a,(1,2,0))
- arr_sw = np.swapaxes(arr,0,1)
- np.where(arr > 0, 2, arr) # 양수이면 2 아니면 그냥 arr 숫자 그대로
- arr.sort(0) array에 바로 반영하는 거라서, 그다음에 array를 프린트를 해야한다.
- arr.sort(1)
- np.percentile(large_arr,5)
- array_9[1::2,::2] = 1 # 홀수 행의 짝수열
- array_9[::2,1::2] = 1 # 짝수 행의 홀수열
- z = np.tile(np.array([[0,1],[1,0]]),(4,4)) # [[0,1],[1,0]] 행과 열로 4번씩 반복해서 만들기
- np.random.seed       # seed를 통한 난수 생성
- np.random.randint    # 균일분포의 정수 난수 1개 생성
- np.random.rand       # 0부터 1사이의 균일분포에서 난수 매트릭스 array 생성
- np.random.randn      # 가우시안 표준 정규 분포에서 난수 매트릭스 array 생성
- np.random.shuffle    # 기존의 데이터의 순서 바꾸기
- np.random.choice     # 기존의 데이터에서 sampling

# 선형대수
arr = np.array([[1,2],[3,4]])
- 역렬
arr_inv = np.linalg.inv(arr)
np.dot(arr,arr_inv)
np.matmul(arr,arr_inv)
- 정방행렬: Square matrix nxn
a = np.full((2,2),7)
- 대각행렬: Diagonal matrix
np.diag([1,2,3])
- 상삼각행렬: np.triu(np.ones((3,3)))
- 하삼각행렬: np.tril(np.ones((3,3)))
- 항등행렬: identity matrix
np.identity(3)
- 행렬식: 선형 종속성 판단. 다중공선성 문제를 확인하는데 유용
determinant_A = np.linalg.det(A)
