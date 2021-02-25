#ifndef _ACTIVATIONH_
#define _ACTIVATIONH_

struct Activation {
    enum methods { 
        NONE,
        SIGMOID
    };
    static double activate(int, double);
};

#endif