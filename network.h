#ifndef _NETWORKH_
#define _NETWORKH_

#include <vector>
#include <string>
#include "layer.h"

class Network {
    private:
        std::vector<Layer*> layers;  // store each layer
    public:
        Network(int, int, std::vector<int>);
        void print();
        std::vector<double> forwardPass(std::vector<double>, bool);  // forward pass of the network with the input values as a vector
        double train();
};

#endif