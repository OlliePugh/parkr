#include <math.h>
#include <stdexcept>
#include "activation.h"

double sigmoid(double value) {
    return 1/(1+exp(-value));
}

double derivativeSigmoid(double value) {
    return value * (1-value);
}

double Activation::activate(Activation::method method, double value) {
    switch (method)
    {
    case Activation::SIGMOID:
        return sigmoid(value);
        break;
    
    default:
        throw std::invalid_argument("Unkown method called");
        break;
    }
}

double Activation::derivativeActivate(Activation::method method, double value) {
    switch (method)
    {
    case Activation::SIGMOID:
        return derivativeSigmoid(value);
        break;
    
    default:
        throw std::invalid_argument("Unkown method called");
        break;
    }
}