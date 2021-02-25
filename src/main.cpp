#include <iostream>
#include <string>

#include "network.h"

int main() {
    std::vector<int> hiddenLayers = {};
    Network myNetwork = Network(2, 1, hiddenLayers, Activation::SIGMOID);
    myNetwork.print();

    std::vector<std::vector<double>> trainingData = {{1, 4},
                                                    {2, 9},
                                                    {5, 6},
                                                    {4, 5},
                                                    {6, 0.7},
                                                    {1, 1.5}};
    std::vector<std::vector<double>> expectedResults = {{0},{1},{1},{1},{0},{0}};

    myNetwork.train(10000, trainingData, expectedResults);

    myNetwork.print();

    /*std::vector<double> params = {1,1,1};
    std::vector<double> results = myNetwork.forwardPass(params, true);*/
    
};