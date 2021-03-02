#include <iostream>
#include <time.h>
#include <stdexcept>
#include <string>
#include <map>

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
        for (int j = 0; j < this->layers.at(i)->getNodes().size(); j++){
            std::cout << (char) (65+nodeId++) << " (" << this->layers.at(i)->getNodes().at(j)->getBias() << ") ";
        }
        std::cout << std::endl; 
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
                
                std::cout << " W: " << currentLink->getWeight() << std::endl;
            }
            nodeId++;
        }
        
    }
    std::cout << std::endl;
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
                currentNode->setRawValue(inputValues.at(nodeCount));  // set the raw value because it is an input
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

void backPropogate(Network* network, std::vector<double> expectedResults, std::vector<double> obtainedResults, double stepSize) {
    double delta;  // store current delta of node
    double activationDerivative;  // store the value of the node put through the integrated activation function
    double sumOfOutputs;  // the sum of all of the weights for the inputs

    std::map<Link*, double> sumWeightMap;  // stores the sum of the updates weights (this will need to be divided by the toatl number of rows)
    std::map<Node*, double> sumBiasMap;
    std::vector<double> prevLayerDeltas;
    std::vector<double> currentDeltas;
    
    std::vector<Node*> outputNodes = network->getLayers().at(network->getLayers().size()-1)->getNodes();

            for (size_t outputNode = 0; outputNode < outputNodes.size(); outputNode++) {  // for each output node
                Node* currentNode = outputNodes.at(outputNode);

                activationDerivative = Activation::integralActivate(network->getActivationMethod(), currentNode->getValue());
                delta = (expectedResults.at(outputNode)-currentNode->getValue()) * activationDerivative;  // calculate delta
                prevLayerDeltas.push_back(delta);

                sumBiasMap[currentNode] += (currentNode->getBias() + (stepSize * delta * 1));
            } 

            for (int hiddenLayer = network->getLayers().size() - 2; hiddenLayer >= 0; hiddenLayer--) {  // for each hidden layer
                Layer* currentLayer = network->getLayers().at(hiddenLayer);
                currentDeltas.clear();  // clear the current deltas vector
                for (size_t nodeCounter = 0; nodeCounter < currentLayer->getNodes().size(); nodeCounter++) {  // for each node in the layer
                    Node* currentNode = currentLayer->getNodes().at(nodeCounter);

                    activationDerivative = Activation::integralActivate(network->getActivationMethod(), currentNode->getValue());
                    sumOfOutputs = 0;  // reset the sum of outputs value

            activationDerivative = currentNode->getValue() * (1-currentNode->getValue());  // calculate derived output value
            sumOfOutputs = 0;  // reset the sum of outputs value

            for (size_t linkCounter = 0; linkCounter < currentNode->getOutLinks().size(); linkCounter++) {
                sumOfOutputs += (currentNode->getOutLinks().at(linkCounter)->getWeight()*prevLayerDeltas.at(linkCounter));
            }

            delta = sumOfOutputs*activationDerivative;
            currentDeltas.push_back(delta);  // add delta to current deltas for the next layer 
        }

        // once all the deltas have been calculated go through and update all weights and bias' for that layer

        for (size_t nodeCounter = 0; nodeCounter < currentLayer->getNodes().size(); nodeCounter++) {
            Node* currentNode = currentLayer->getNodes().at(nodeCounter);
            for (size_t linkCounter = 0; linkCounter < currentNode->getOutLinks().size(); linkCounter++) {  // for each out link of that node
                Link* currentLink = currentNode->getOutLinks().at(linkCounter);

                sumWeightMap[currentLink] += currentLink->getWeight()+(stepSize*prevLayerDeltas.at(linkCounter)*currentNode->getRawValue());
            }
            if (currentLayer->getType() != LayerType::INPUT) {
                sumBiasMap[currentNode] += (currentNode->getBias() + (stepSize * currentDeltas.at(nodeCounter) * 1));
            }
            
        }

        prevLayerDeltas.clear();
        prevLayerDeltas = currentDeltas;
    }

    std::map<Link*, double>::iterator weightIt = sumWeightMap.begin();
    for (auto weightIt = sumWeightMap.begin(); weightIt != sumWeightMap.end(); ++weightIt)  {  // for each node in the map
        weightIt->first->setWeight((weightIt->second)/obtainedResults.size());  // divide the changes to make to the weight by the total amounts of data it has been trained on
    }

    std::map<Node*, double>::iterator biasIt = sumBiasMap.begin();
    for (auto biasIt = sumBiasMap.begin(); biasIt != sumBiasMap.end(); ++biasIt)  {  // for each node in the map
        biasIt->first->setBias((biasIt->second)/obtainedResults.size());  // divide the changes to make to the weight by the total amounts of data it has been trained on
    }
        
}

double Network::train(int epochs, std::vector<std::vector<double>> trainingData, std::vector<std::vector<double>> expectedResults, double stepSize) {
    
    if (expectedResults.size() != trainingData.size()) throw std::invalid_argument("Amount of expected results does not match amount of training data");
    
    std::vector<double> forwardPassResult;
    std::vector<double>correctResult;
    double trainingLoss;

    
    for (size_t epoch = 0; epoch < epochs; epoch++) {
       
       for (size_t i = 0; i < trainingData.size(); i++) {  // for each row of training data 

            forwardPassResult = this->forwardPass(trainingData.at(i));  // do a forward pass
            correctResult = expectedResults.at(i);
            
            trainingLoss=0;
            for (size_t j = 0; j < correctResult.size(); j++) {
                trainingLoss += (correctResult.at(j)-forwardPassResult.at(j));
            }

            trainingLoss = (trainingLoss*trainingLoss)/correctResult.size();

            std::cout << trainingLoss << " training loss " << std::endl;
            
            backPropogate(this, correctResult, forwardPassResult, stepSize);
            
       } 
    }

    return 1.0;
    
}