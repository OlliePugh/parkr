#ifndef _ACTIVATIONH_
#define _ACTIVATIONH_

struct Activation {
    enum method { 
        NONE,
        SIGMOID
    };
    static double activate(Activation::method, double);
};

#endif