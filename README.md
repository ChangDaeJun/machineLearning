# machineLearning

* 머신러닝 레포지토리입니다.
* 사용 라이브러리 : jblas (행렬 연산을 위한 라이브러리)
* 언어 : java

## 선형 회귀

* 선형 회귀란, 사전에 입력받은 자료를 통해 오차율을 최소화하는 직선(정확히는 가중치와 바이어스)을 구하는 머신러닝 방식이다.
* 오차율은 현재 가지고 있는 가중치 W와 바이어스 b에 대해, (t - wx + b)^2 으로 표기한다. 즉, 실제 값과 구하는 값의 차이의 제곱으로 표기된다.
* 선형 회귀는 오차율을 최소화하는 W, b을 구하기 위해, 경사하강법을 사용하게 된다.
* 가중치의 변화에 따른 오차율 함수는 아래로 볼록한 2차 함수이 된다. 해당 2차 함수에서의 최솟값은 w에 대한 미분값이 0이 되는 시점이다.
* 이 지점을 구하는 방법이 경사하강법이다.
* 경사하강법은 미분값이 음수일 경우에는 그 값을 +하고, 미분값이 양수일 때에는 - 하여, 점차 그 간격을 좁혀나가는 방식이다.