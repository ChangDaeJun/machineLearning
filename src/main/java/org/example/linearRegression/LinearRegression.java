package org.example.linearRegression;

import org.jblas.DoubleMatrix;


import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;

public class LinearRegression {

    public static double W = Math.random();
    public static double b = Math.random();

    public static void main(String[] args) {
        //학습 비중 : 가중치와 bias의 값을 변경할 시 어떤 비율로 변경할 지 정하는 상수
        double learning_rate = 1e-2;

        //x : 원인값, y : 결과값
        double[] x_data = {1, 2, 3, 4, 5};
        double[] y_data = {2, 3, 4, 5, 6};

        //데이터 계산을 행렬식으로 하기 위해 행렬로 변환함.
        DoubleMatrix matrixX = new DoubleMatrix(x_data).reshape(5,1);
        DoubleMatrix matrixY = new DoubleMatrix(y_data).reshape(5, 1);

        //오차율을 구하는 함수에 행렬 X, Y을 대입한 함수. W 미분, b 미분을 위한 함수이다.
        BiFunction f = loss_func().apply(matrixX, matrixY);

        //현재 오차율을 구하는 함수, 현제 함수를 구한다.
        double lossVar = loss_val().apply(matrixX, matrixY);
        System.out.println("lossVar = " + lossVar + ", Initial W = " + W + ", b = " + b);

        //i번 만큼 실행하여, 오차율을 최소화하도록 변수 W, b 값을 조정한다.
        for(int i = 0; i < 12000; i++){
            W -= learning_rate * derivative1(f, W);
            b -= learning_rate * derivative2(f, b);
            if(i % 600 == 0){
                System.out.println("index : "+ i + ", lossVar = " + loss_val().apply(matrixX, matrixY) + ", Initial W = " + W + ", b = " + b);
            }
        }

        //44에 대한 예상 값을 구하는 과정. 결과 : 45
        double[] z_data = {44};
        DoubleMatrix matrix_z = new DoubleMatrix(z_data).reshape(1, 1);
        DoubleMatrix result = predict().apply(matrix_z);
        System.out.println(Arrays.toString(result.toArray()));
    }


    //행렬 X, Y값을 받아, 해당 값에 따라 오차율을 계산하는 함수를 리턴한다.
    public static BiFunction<DoubleMatrix, DoubleMatrix, BiFunction<Double, Double, Double>> loss_func(){

        return (x, y) -> {
            BiFunction<Double, Double, Double> result = (W, b) -> {
                DoubleMatrix z = x.mul(W).add(b).add(y.mul(-1));
                return z.mul(z).sum() / x.length;
            };
            return result;
        };
    }

    //행렬 X, Y 값을 받아, 해당 값에 따라 현제 가중치와 bias 값을 통해 오챠율을 계산한다.
    public static BiFunction<DoubleMatrix, DoubleMatrix, Double> loss_val(){
        return (x, y) -> {
            DoubleMatrix z = x.mul(W).add(b).add(y.mul(-1));
            return z.mul(z).sum() / x.length;
        };
    }

    //미래에 예상되는 결과 출력
    public static Function<DoubleMatrix, DoubleMatrix> predict(){
        return x -> {
            DoubleMatrix z = x.mul(W).add(b);
            return z;
        };
    }

    //W에 대한 편미분
    public static double derivative1(BiFunction<Double, Double, Double> f, double var){
        double delta = 1e-5;
        double diff_val = (f.apply(var + delta, b) - f.apply(var - delta, b)) / (2 * delta);
        return diff_val;
    }

    //bias에 대한 편미분
    public static double derivative2(BiFunction<Double, Double, Double> f, double var){
        double delta = 1e-5;
        double diff_val = (f.apply(W, var + delta) - f.apply(W, var - delta)) / (2 * delta);
        return diff_val;
    }
}
