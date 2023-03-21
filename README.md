# machineLearning

* 머신러닝 레포지토리입니다.
* 사용 라이브러리 : jblas (행렬 연산을 위한 라이브러리)
* 언어 : java

### 자료의 출처
* 학습 책 : 박성호, 머신러닝을 위한 파이썬 한 조각 (비제이퍼블릭, 2020)
* 사진 출처 : https://heung-bae-lee.github.io/2019/12/08/deep_learning_02/ (오차 함수, 경사하강 그래프, 교차 엔트로피 오차 함수 등)
## 선형 회귀

* 선형 회귀란, 사전에 입력받은 자료를 통해 오차율을 최소화하는 직선(정확히는 가중치와 바이어스)을 구하는 머신러닝 방식이다.
* 오차율은 현재 가지고 있는 가중치 W와 바이어스 b에 대해, (t - wx + b)^2 으로 표기한다. 즉, 실제 값과 구하는 값의 차이의 제곱으로 표기된다.
![스크린샷 2023-03-21 오전 10 17 38](https://user-images.githubusercontent.com/97227920/226498370-90907833-729c-46aa-ad5f-db4ea0fb0dcf.png)
* 선형 회귀는 오차율을 최소화하는 W, b을 구하기 위해, 경사하강법을 사용하게 된다.
* 가중치의 변화에 따른 오차율 함수는 아래로 볼록한 2차 함수이 된다. 해당 2차 함수에서의 최솟값은 w에 대한 미분값이 0이 되는 시점이다.
* 이 지점을 구하는 방법이 경사하강법이다.
* 경사하강법은 미분값이 음수일 경우에는 그 값을 +하고, 미분값이 양수일 때에는 - 하여, 점차 그 간격을 좁혀나가는 방식이다.
![스크린샷 2023-03-21 오전 10 17 11](https://user-images.githubusercontent.com/97227920/226498406-1d5806ec-a020-463d-8228-67af8bed0f2c.png)
* 프로젝트 내 코드 : https://github.com/ChangDaeJun/machineLearning/blob/main/src/main/java/org/example/linearRegression/LinearRegression.java

### 테스트 결과
![스크린샷 2023-03-21 오전 8 43 05](https://user-images.githubusercontent.com/97227920/226489047-6c21091c-a02a-49b4-b0c2-1f749de8f9e9.png)
* 실행횟수(index)가 증가할 수록, 오차율(lossVar)은 감소하고, 가중치와 바이어스는 1에 수렴하는 것을 알 수 있다.
* 실제로 입력된 데이터는 x -> y에 대해 x : {1, 2, 3, 4, 5}, y : {2, 3, 4, 5, 6}이기에, y = Wx + b인 W, b가 각각 1, 1로 기대되며, 실제로 유사하게 나타났다.
* 마지막에는 특정 값(44)에 대한 결과를 예측하였고, 예측 결과는 45.0000000027으로, 거의 유사하게 예측하였다.

## 선형 분류

* 선형 분류는 데이터를 가장 잘 나누는 직선을 찾고, 직선을 기준으로 데이터를 분류하는 알고리즘이다.
* data -> regression -> classification -> result
* regression은 앞선 선형 회귀와 유사한 방식을 채택할 수 있다.
* classification에서는 여러 함수를 사용할 수 있는데, 대표적으로 sigmoid 함수를 사용할 수 있다.
* 분류에서의 오차값을 나타내는 손실함수는 log 식을 포함하며, 크로스 엔트로피라고 부른다.
![스크린샷 2023-03-21 오전 10 17 42](https://user-images.githubusercontent.com/97227920/226498455-8d0ef6fe-448b-469d-905c-52ebcfdf867d.png)
* 프로젝트 내 코드 : https://github.com/ChangDaeJun/machineLearning/blob/main/src/main/java/org/example/linearClassification/LinearClassification.java

### 테스트 결과
![스크린샷 2023-03-21 오전 10 14 41](https://user-images.githubusercontent.com/97227920/226498549-83bf0d14-a514-4653-9145-74abfe742556.png)

