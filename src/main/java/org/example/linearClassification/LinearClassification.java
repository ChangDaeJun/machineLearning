package org.example.linearClassification;

import org.jblas.DoubleMatrix;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;

public class LinearClassification {
    public static double W = Math.random();
    public static double b = Math.random();
    public static void main(String[] args) {
        double learning_rate = 1e-2;
        double[] studyTime = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
        double[] passResult = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1};

        DoubleMatrix matrixTime = new DoubleMatrix(studyTime).reshape(10,1);
        DoubleMatrix matrixPass = new DoubleMatrix(passResult).reshape(10, 1);

        BiFunction<Double, Double, Double> f = loss_func().apply(matrixTime, matrixPass);

        double lossVar = loss_val().apply(matrixTime, matrixPass);
        System.out.println("lossVar = " + lossVar + ", Initial W = " + W + ", b = " + b);

        for(int i = 0; i < 50000; i++){
            W -= learning_rate * derivative1(f, W);
            b -= learning_rate * derivative2(f, b);
            if(i % 5000 == 0){
                System.out.println("index : "+ i + ", lossVar = " + loss_val().apply(matrixTime, matrixPass) + ", Initial W = " + W + ", b = " + b);
            }
        }

        double[] z1_data = {3};
        DoubleMatrix matrix_z = new DoubleMatrix(z1_data).reshape(1, 1);
        double result = predict().apply(matrix_z).get(0);
        if(result < 0.5) System.out.println(z1_data[0] +" 시간 공부는 "+ (1 - result)*100 +"% 확률로 불합격이 예상됩니다.");
        else System.out.println(z1_data[0] +" 시간 공부는 "+result*100 + "확률로 합격이 예상됩니다.");

        double[] z2_data = {17};
        DoubleMatrix matrix_z2 = new DoubleMatrix(z2_data).reshape(1, 1);
        double result2 = predict().apply(matrix_z2).get(0);
        if(result2 < 0.5) System.out.println(z2_data[0] +" 시간 공부는 "+ (1 - result2)*100 +"% 확률로 불합격이 예상됩니다.");
        else System.out.println(z2_data[0] +" 시간 공부는 "+result2*100 + "% 확률로 합격이 예상됩니다.");
    }

    public static Function<DoubleMatrix, DoubleMatrix> sigmoid(){
        return x -> {
            double[] arr = new double[x.length];
            double[] x_arr = x.toArray();
            for(int i = 0; i < x.length; i++){
                arr[i] = 1/ (1 + Math.exp(-x_arr[i]));
            }
            return new DoubleMatrix(arr).reshape(x.getRows(), x.getColumns());
        };
    }

    public static BiFunction<DoubleMatrix, DoubleMatrix, BiFunction<Double, Double, Double>> loss_func(){

        return (x, y) -> {
            BiFunction<Double, Double, Double> result = (W, b) -> {
                DoubleMatrix z1 = x.mul(W).add(b);
                DoubleMatrix z2 = sigmoid().apply(z1);
                return cross_entropy().apply(y, z2).sum();
            };
            return result;
        };
    }

    public static BiFunction<DoubleMatrix, DoubleMatrix, Double> loss_val(){
        return (x, y) -> {
            DoubleMatrix z1 = x.mul(W).add(b);
            DoubleMatrix z2 = sigmoid().apply(z1);
            return cross_entropy().apply(y, z2).sum();
        };
    }

    public static BiFunction<DoubleMatrix, DoubleMatrix, DoubleMatrix> cross_entropy(){
        return(x, y)-> {
            double delta = 1e-7;
            double[] arr = new double[x.length];
            double[] x_arr = x.toArray();
            double[] y_arr = y.toArray();
            for(int i = 0; i < x.length; i++){
                if(x_arr[i] == 0){
                    arr[i] = Math.log((1 - y_arr[i]) + delta);
                }else{
                    arr[i] = Math.log(y_arr[i] + delta);
                }
            }
            return new DoubleMatrix(arr).reshape(x.getRows(), x.getColumns()).mul(-1);
        };
    }

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

    public static Function<DoubleMatrix, DoubleMatrix> predict(){
        return x -> {
            DoubleMatrix z = x.mul(W).add(b);
            DoubleMatrix y = sigmoid().apply(z);
            return y;
        };
    }
}
