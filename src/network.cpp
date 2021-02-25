#include <iostream>
#include <time.h>
#include <stdexcept>
#include <string>

#include "network.h"

Network::Network(int inputNodes, int outputNodes, std::vector<int> hiddenLayers, Activation::method activationMethod) {
    srand(time(NULL));  // init for randomised values

    this->activationMethod = activationMethod;

    this->layers.push_back(new Layer(this, LayerType::INPUT, inputNodes));  // create a new layer of type input

    for (int i=0; i<hiddenLayers.size(); i++) {
        this->layers.push_back(new Layer(this, LayerType::HIDDEN, hiddenLayers.at(i), this->layers.back()));  // create a new hidden layer 
    }

    this->layers.push_back(new Layer(this, LayerType::OUTPUT, outputNodes, this->layers.back()));  // create output layer
}

void Network::print() {
    int nodeId = 0;
    for (int i=0; i < this->layers.size(); i++) {
        std::string toPrint = "";
        for (int j = 0; j < this->layers.at(i)->getNodes().size(); j++){
            toPrint += (char) (65+nodeId++); // incriment the node id
        }
        std::cout << toPrint << std::endl; 
    }

    std::cout << std::endl; 

    nodeId = 0;  // reset the node ID counter

    // print all links and their information
    
    for (size_t layerCounter = 0; layerCounter < this->layers.size()-1; layerCounter++) {  // go through each layer but the last one as their are no outbound connections on the last node
        
        Layer* currentLayer = this->layers.at(layerCounter);
        
        int nodeIdAtTopNode = nodeId;

        for (size_t nodeCounter = 0; nodeCounter < currentLayer->getNodes().size(); nodeCounter++) {
            
            Node* currentNode = currentLayer->getNodes().at(nodeCounter);
            for (size_t linkCounter = 0; linkCounter < currentNode->getOutLinks().size(); linkCounter++) {
                Link* currentLink = currentNode->getOutLinks().at(linkCounter);

                std::cout << (char) (65+nodeId) << "->" << (char) (65+(nodeId-nodeCounter+currentLayer->getNodes().size()+linkCounter));  // calculate the ID of the node we are currently comparing to
                
                std::cout << " B: " << currentLink->getBias() << " W: " << currentLink->getWeight() << std::endl;
            }
            nodeId++;
        }
        
    }
    

    std::cout << std::endl << "Press enter to continue" << std::endl;
    std::cin.get();  // wait for the user
}

std::vector<double> Network::forwardPass(std::vector<double> inputValues, bool printResult=false) {

    std::vector<double> outputValues;

    for (size_t layerCount = 0; layerCount < this->layers.size(); layerCount++) {  // for each layer
        
        Layer* currentLayer = this->layers.at(layerCount);  // set as the current layer
        
        for (size_t nodeCount = 0; nodeCount < currentLayer->getNodes().size(); nodeCount++) { // for each node in the layer

            Node* currentNode = currentLayer->getNodes().at(nodeCount);

            if (currentLayer->getType() == LayerType::INPUT) {  // if first layer
                
                if (inputValues.size() != currentLayer->getNodes().size()) {  // if the amount of inputs doesnt match the amount of input nodes
                    throw std::invalid_argument("Expecting " + std::to_string(currentLayer->getNodes().size()) + " input values, received " + std::to_string(inputValues.size()));
                }

                currentNode->setValue(inputValues.at(nodeCount));  // set the value of the nth input to the nth input node
            }
            
            else if (currentLayer->getType() == LayerType::OUTPUT) {  // this is the output layer
                outputValues.push_back(currentNode->calcValue());  // add the value of one of the output nodes to the output vector
            }
                
            else {  // this is a hidden layer
                currentNode->calcValue();  // calc the value at the current node
            }
        } 
    }
    if (printResult) {
        for (size_t i = 0; i < outputValues.size(); i++) {
            std::cout << std::to_string(outputValues.at(i)) << std::endl;
        }
    }
    return outputValues;
}