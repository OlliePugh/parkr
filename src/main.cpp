#include <iostream>
#include <string>

#include "network.h"

int main() {
    std::vector<int> hiddenLayers = {2};
    Network myNetwork = Network(3, 1, hiddenLayers, Activation::SIGMOID);
    myNetwork.print();

    std::vector<double> params = {1,1,1};
    std::vector<double> results = myNetwork.forwardPass(params, true);
    
};