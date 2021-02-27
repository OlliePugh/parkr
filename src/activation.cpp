#include <math.h>
#include "activation.h"

double sigmoid(double value) {
    return 1/(1+exp(-value));
}

double integralSigmoid(double value) {
    return value * (1-value);
}

double Activation::activate(Activation::method method, double value) {
    switch (method)
    {
    case Activation::SIGMOID:
        return sigmoid(value);
        break;
    
    default:
        return value;
        break;
    }
}

double Activation::integralActivate(Activation::method method, double value) {
    switch (method)
    {
    case Activation::SIGMOID:
        return integralSigmoid(value);
        break;
    
    default:
        return 1;
        break;
    }
}