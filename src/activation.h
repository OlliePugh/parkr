#ifndef _ACTIVATIONH_
#define _ACTIVATIONH_

struct Activation {
    enum method { 
        SIGMOID,
        TANH
    };
    static double activate(Activation::method, double);
    static double derivativeActivate(Activation::method, double);
};

#endif