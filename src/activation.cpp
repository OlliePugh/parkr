#include <cmath>
#include <stdexcept>
#include "activation.h"

double sigmoid(double value) {
    return 1/(1+exp(-value));
}

double derivativeSigmoid(double value) {
    return value * (1-value);
}

double derivativeTanh(double value) {
    return 1-pow(std::tanh(value), 2.0);
}

double relu(double value) {
    return std::fmax(0.0, value);
}

double derivativeRelu(double value) {
    if (value>0) return value;
    else return 0.01*value;
}

double Activation::activate(Activation::method method, double value) {
    switch (method)
    {
    case Activation::LINEAR:
        return value;
        break;

    case Activation::SIGMOID:
        return sigmoid(value);
        break;

    case Activation::TANH:
        return std::tanh(value);
        break;

    case Activation::RELU:
        return relu(value);
        break;
    
    default:
        throw std::invalid_argument("Unkown method called");
        break;
    }
}

double Activation::derivativeActivate(Activation::method method, double value) {
    switch (method)
    {
    case Activation::LINEAR:
        return 1.0;
        break;

    case Activation::SIGMOID:
        return derivativeSigmoid(value);
        break;

    case Activation::TANH:
        return derivativeTanh(value);
        break;

    case Activation::RELU:
        return derivativeRelu(value);
        break;
    
    default:
        throw std::invalid_argument("Unkown method called");
        break;
    }
}