package org.example;

import java.util.function.Function;

public class OneVar {
    public static void main(String[] args) {
        System.out.println(simpleDerivative(func1(), 3));
    }

    public static double simpleDerivative(Function<Double, Double> f, double var){
        double delta = 1e-5;
        double diff_val = (f.apply(var + delta) - f.apply(var - delta)) / (2 * delta);
        return diff_val;
    }

    public static Function<Double, Double> func1(){
        return x -> x*x;
    }
}