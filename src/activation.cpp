#include <math.h>
#include "activation.h"

double sigmoid(double value) {
    return 1/(1+exp(-value));
}

double Activation::activate(int method, double value) {
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