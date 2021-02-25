#ifndef _NETWORKH_
#define _NETWORKH_

#include <vector>
#include <string>
#include "layer.h"
#include "activation.h"

class Network {
    private:
        std::vector<Layer*> layers;  // store each layer
        Activation::method activationMethod;
    public:
        Network(int, int, std::vector<int>, Activation::method=Activation::NONE);  // set default activation to none
        void print();
        std::vector<double> forwardPass(std::vector<double>, bool);  // forward pass of the network with the input values as a vector
        Activation::method getActivationMethod() { return this->activationMethod; };
        double train(int, std::vector<std::vector<double>>, std::vector<std::vector<double>>, double=0.1);
        std::vector<Layer*> getLayers() {return this->layers; };
};

#endif