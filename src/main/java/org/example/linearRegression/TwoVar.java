package org.example.linearRegression;

import java.util.function.BiFunction;
import java.util.function.Function;

public class TwoVar {
    public static void main(String[] args) {
        System.out.println(simpleDerivative(func(), 1.0)); // x 편미분
        System.out.println(simpleDerivative2(func(), 2.0));// y 편미분
    }

    public static double simpleDerivative(BiFunction<Double, Double, Double> f, double var) {
        double delta = 1e-5;
        double diff_val = (f.apply(var + delta, 2.0) - f.apply(var - delta, 2.0))/(2 * delta);
        return diff_val;
    }

    public static double simpleDerivative2(BiFunction<Double, Double, Double> f, double var){
        double delta = 1e-5;
        double diff_val = (f.apply(1.0, var + delta) - f.apply(1.0, var - delta))/(2 * delta);
        return diff_val;
    }

    public static BiFunction<Double, Double, Double> func() {
        return (x, y) -> 2 * x + 3 * x * y + y * y * y;
    }


}
