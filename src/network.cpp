#include <iostream>
#include <math.h>
#include <time.h>
#include <stdexcept>
#include <string>
#include <fstream>
#include <map>
#include <cstring>
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

typedef std::tuple<std::map<Link*, double>, std::map<Node*, double>> nodechangemap_t;
typedef std::map<Node*, double> deltamap_t;

std::map<Node*, double> generateDeltas(Network* network, std::vector<double> *expectedResults) {  // will only work if called while value are still stored in the nodes 
    std::map<Node*, double> deltaMap;

    // generate deltas for output nodes
    Layer* outputLayer = network->getLayers().back();
    
    for (size_t i = 0; i < outputLayer->getNodes().size(); i++)  {
        Node* currentNode = outputLayer->getNodes().at(i);

        double derivedValue = Activation::derivativeActivate(network->getActivationMethod(), currentNode->getValue());
        deltaMap[currentNode] = (expectedResults->at(i)-currentNode->getValue())*(derivedValue);
    }
    
    // for all other layers
    for (int i = network->getLayers().size() - 2; i >= 0; i--) {  // loop in reverse order
        Layer* currentLayer = network->getLayers().at(i);
        for (size_t j = 0; j < currentLayer->getNodes().size(); j++) {  // for every node in the layer
            Node* currentNode = currentLayer->getNodes().at(j);

            double derivedValue = Activation::derivativeActivate(network->getActivationMethod(), currentNode->getValue());
            
            double sumOfDeltaWeights=0;

            for (size_t outLinksCounter = 0; outLinksCounter < currentNode->getOutLinks().size(); outLinksCounter++) {  // for each outlink of the node
                Link* currentLink = currentNode->getOutLinks().at(outLinksCounter);
                sumOfDeltaWeights += currentLink->getWeight() * deltaMap[currentLink->getChild()];  // add the weight of the node to the delta of the node it points to
            }

            deltaMap[currentNode] = sumOfDeltaWeights * derivedValue;
        }
    }
    
    return deltaMap;
}

nodechangemap_t generateChanges(Network* network, deltamap_t* deltaMap, double stepSize) {
    std::map<Link*, double> weightMap;  // create a map to store the changes to the weights
    std::map<Node*, double> biasMap;  // create a map to store the changes to bias

    for (size_t i = 0; i < network->getLayers().size(); i++)  {  // for all layers
        Layer* currentLayer = network->getLayers().at(i);

        for (size_t j = 0; j < currentLayer->getNodes().size(); j++){  // for each node in the layer
            Node* currentNode = currentLayer->getNodes().at(j);

            // calculate weights

            for (size_t k = 0; k < currentNode->getOutLinks().size(); k++) {  // for each out link of the layer
                Link* currentLink = currentNode->getOutLinks().at(k);
                double weight = currentLink->getWeight();
                double delta = (*deltaMap)[currentLink->getChild()];
                double parentVal = currentLink->getParent()->getValue();

                weightMap[currentLink] = currentLink->getWeight() + (stepSize *  (*deltaMap)[currentLink->getChild()] * currentLink->getParent()->getValue());
            }

            // calculate new bias if not input layer
            if(i!=0) biasMap[currentNode] = currentNode->getBias() + (stepSize * (*deltaMap)[currentNode]);
        }
    }
    
    return std::make_tuple(weightMap, biasMap);

}

void backPropogate(Network* network, std::vector<nodechangemap_t> deltaMaps) {

    std::map<Link*, double> finalWeightValues;

    for (size_t mapCount = 0; mapCount < deltaMaps.size(); mapCount++) {  // for each map of values that each training data would like to change
        std::map<Link*, double>* currentWeightMap = &std::get<0>(deltaMaps.at(mapCount));

        std::map<Link*, double>::iterator weightIt = currentWeightMap->begin();
        for (auto weightIt = currentWeightMap->begin(); weightIt != currentWeightMap->end(); ++weightIt)  {  // for each node in the map
            finalWeightValues[weightIt->first] += weightIt->second / deltaMaps.size(); // add the change to the final value map
        }
    }

    std::map<Link*, double>::iterator finalWeightIt = finalWeightValues.begin();
    for (auto finalWeightIt = finalWeightValues.begin(); finalWeightIt != finalWeightValues.end(); ++finalWeightIt)  {  // for each node in the map
        finalWeightIt->first->setWeight(finalWeightIt->second);  // divide the changes to make to the weight by the total amounts of data it has been trained on
    }

    std::map<Node*, double> finalBiasValues; 

    for (size_t mapCount = 0; mapCount < deltaMaps.size(); mapCount++) {  // for each map of values that each training data would like to change
        std::map<Node*, double>* finalBiasMap = &std::get<1>(deltaMaps.at(mapCount));

        std::map<Node*, double>::iterator biasIt = finalBiasMap->begin();
        for (auto biasIt = finalBiasMap->begin(); biasIt != finalBiasMap->end(); ++biasIt)  {  // for each node in the map
            finalBiasValues[biasIt->first] += biasIt->second / deltaMaps.size(); // add the change to the final value map
        }
    }

    std::map<Node*, double>::iterator finalBiasIt = finalBiasValues.begin();
    for (auto finalBiasIt = finalBiasValues.begin(); finalBiasIt != finalBiasValues.end(); ++finalBiasIt)  {  // for each node in the map
        finalBiasIt->first->setBias(finalBiasIt->second);  // set the bias of the node to the average of all the deltas
    }
}

double _train(Network* network, dataset* trainingData, dataset* expectedResults, dataset* validationData, dataset* validationExpectedResults, double stepSize) {
    std::vector<nodechangemap_t> changeMap;
    for (size_t i = 0; i < trainingData->size(); i++) {  // for each row of training data  
        std::vector<double> forwardPassResults = network->forwardPass(trainingData->at(i));  // perform a forward pass
        deltamap_t deltas = generateDeltas(network, &expectedResults->at(i));  // generate the deltas 
        changeMap.push_back(generateChanges(network, &deltas, stepSize));  // add the requested changes from that forward pass
            
    }
    backPropogate(network, changeMap);  // apply changes to the weights and bias

    //calaculte loss
    double trainingLoss = 0.0;
    for (size_t validationCounter = 0; validationCounter < validationData->size(); validationCounter++) {
        for (size_t outputNodeCount = 0; outputNodeCount < validationExpectedResults->at(0).size(); outputNodeCount++) {  // add the loss for that pass
            std::vector<double> result = network->forwardPass(validationData->at(validationCounter));  // pass validation data forward
            
            double toMult = validationExpectedResults->at(validationCounter).at(outputNodeCount)-result.at(outputNodeCount);
            trainingLoss += std::pow(toMult,2.0) / (validationExpectedResults->at(0).size() * validationExpectedResults->size());
        }
    }

    return trainingLoss;
}

void Network::train(int epochs, dataset* trainingData, dataset* expectedResults, dataset* validationData, dataset* expectedValidation, unsigned char options, double stepSize) {
    
    if (expectedResults->size() != trainingData->size()) throw std::invalid_argument("Amount of expected results does not match amount of training data");

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        double trainingLoss = _train(this, trainingData, expectedResults, validationData, expectedValidation, stepSize);
        if (!(options & Network::trainingOptions::silencePrintLoss)) {
            std::cout << trainingLoss << " training loss at epoch " << epoch+1 << std::endl;  // display the loss for that forward pass
        }
        trainingLoss = 0;
    }
}

typedef std::vector<std::vector<std::vector<double>>> batchVector;

void Network::batchTrain(int epochs, dataset* trainingData, dataset* expectedResults, dataset* validationData, dataset* expectedValidation, int batchSize, unsigned char options, double stepSize) {
    if (expectedResults->size() != trainingData->size()) throw std::invalid_argument("Amount of expected results does not match amount of training data");
    if (trainingData->size() < batchSize) throw std::invalid_argument("Batch size can not be greater than training results");

    batchVector trainingBatches;
    batchVector expectedBatches;

    //generate batches
    int startVal = 0;
    for (size_t i = 0; i < std::ceil((double) trainingData->size() / (double)  batchSize); i++) {
        int endingIndex = std::min((int) trainingData->size()-1, startVal+batchSize);

        // slice the vectors into batches
        dataset trainBatch = dataset(trainingData->begin() + startVal, trainingData->begin()+endingIndex);
        dataset expectBatch = dataset(expectedResults->begin() + startVal, expectedResults->begin()+endingIndex);

        trainingBatches.push_back(trainBatch);
        expectedBatches.push_back(expectBatch);
        
        startVal += batchSize;  // add the batch size onto the start index
    }
    
    double trainingLoss = 0;
    for (size_t epoch = 0; epoch < epochs; epoch++) {  // for each epoch
        for (size_t batchCounter = 0; batchCounter < trainingBatches.size(); batchCounter++) {
            trainingLoss += _train(this, &(trainingBatches.at(batchCounter)), &(expectedBatches.at(batchCounter)), validationData, expectedValidation, stepSize);
        }
        if (!(options & Network::trainingOptions::silencePrintLoss)) {
            std::cout << trainingLoss/(double)trainingBatches.size() << " training loss at epoch " << epoch+1 << std::endl;  // display the loss for that forward pass
        }
        trainingLoss = 0;
    }
}

void Network::save(std::string fileName) {  // save the network to a .prkr file
    std::fstream outputFile;
    outputFile.open(fileName+".prkr", std::ios::trunc | std::ios::out | std::ios::binary);

    // save activation method

    outputFile.write( (char*) &this->activationMethod, sizeof(this->activationMethod));

    // enter how many hidden layers there are
    uint16_t layerAmount = this->layers.size()-2;
    outputFile.write((char*) &layerAmount, sizeof(layerAmount));

    uint16_t nodeAmount;
    for (size_t i = 0; i < this->layers.size(); i++) {  // write how many nodes are in each layer
        nodeAmount = this->layers.at(i)->getNodes().size();
        outputFile.write((char*) &nodeAmount, sizeof(nodeAmount));
    }

    // output bias for each node

    Layer* currentLayer;
    Node* currentNode;
    double bias;
    for (size_t i = 0; i < this->layers.size(); i++) {  // write how many nodes are in each layer
        currentLayer = this->layers.at(i);
        for (size_t j = 0; j <currentLayer->getNodes().size(); j++)
        {
            currentNode = currentLayer->getNodes().at(j);
            bias = currentNode->getBias();
            outputFile.write((char*) &bias, sizeof(bias));
        }
    }
    
    // output every weight of every link

    Link* currentLink;
    double weight;
    for (size_t i = 0; i < this->layers.size()-1; i++) {  // loop through each layer except output layer
        currentLayer = this->layers.at(i);
        for (size_t j = 0; j <currentLayer->getNodes().size(); j++)
        {
            currentNode = currentLayer->getNodes().at(j);
            
            for (size_t linkCounter = 0; linkCounter < currentNode->getOutLinks().size(); linkCounter++) {  // for each out link for the node
                currentLink = currentNode->getOutLinks().at(linkCounter);
                weight = currentLink->getWeight();

                outputFile.write((char*) &weight, sizeof(weight));
            }
            
        }
    }
    outputFile.close();
 }

Network Network::open(std::string fileName) {

    std::ifstream inputFile = std::ifstream(fileName+".prkr", std::ios::in | std::ios::binary);  // open the file
    
    inputFile.seekg(0, std::ios::end); 
    int length = inputFile.tellg();  // get the length of the file
    inputFile.seekg(0, std::ios::beg); 
    char * buffer = new char[length];  // create a buffer array of size of the file

    inputFile.read(buffer, length);
    
    inputFile.close();  // close the file

    char* memPointer = &buffer[0];  // point to the first element in the array

    Activation::method activationMethod;
    std::memcpy(&activationMethod, memPointer, sizeof(Activation::method));
    memPointer += sizeof(Activation::method);  // move the pointer the amount of bytes forward

    uint16_t hiddenLayerNum;
    std::memcpy(&hiddenLayerNum, memPointer, sizeof(uint16_t));
    memPointer += sizeof(uint16_t);  // move the pointer the amount of bytes forward

    std::vector<int> nodesInLayers;
    uint16_t tempNodeCount=0;
    
    for (size_t i = 0; i < hiddenLayerNum+2; i++)  {
        std::memcpy(&tempNodeCount, memPointer, sizeof(uint16_t));
        nodesInLayers.push_back(tempNodeCount);
        memPointer += sizeof(uint16_t);  // incriment the pointer
    }

    std::vector<double> biass;
    double nextBias;

    for (size_t i = 0; i < hiddenLayerNum+2; i++)  {  // for each layer
        for (size_t j = 0; j < nodesInLayers.at(i); j++) {
            std::memcpy(&nextBias, memPointer, sizeof(double));
            biass.push_back(nextBias);
            memPointer += sizeof(double);  // incriment the pointer
        }
    } 

    std::vector<double> weights;
    double nextWeight;
    
    for (size_t i = 0; i < hiddenLayerNum+1; i++)  {  // for each layer except output layer
        for (size_t j = 0; j < (nodesInLayers.at(i)*nodesInLayers.at(i+1)); j++) {
            std::memcpy(&nextWeight, memPointer, sizeof(double));
            weights.push_back(nextWeight);
            memPointer += sizeof(double);  // incriment the pointer
        }
    } 

    delete[] buffer;  // free the heap memory

    std::vector<int> hiddenLayerNodeCount;
    for (size_t i = 1; i < nodesInLayers.size()-1; i++)  {
        hiddenLayerNodeCount.push_back(nodesInLayers.at(i));
    }

    Network newNetwork = Network(nodesInLayers.at(0), nodesInLayers.at(nodesInLayers.size()-1), hiddenLayerNodeCount, activationMethod);

    newNetwork.setAllBias(biass);
    newNetwork.setAllWeights(weights);

    return newNetwork;
}

void Network::setAllBias(std::vector<double> biass) {
    Layer* currentLayer;  // store the current layer

    int posCounter = 0;

    for (size_t layerCounter = 0; layerCounter < this->getLayers().size(); layerCounter++) {
        currentLayer = this->getLayers().at(layerCounter);
        
        for (size_t nodeCounter = 0; nodeCounter < currentLayer->getNodes().size(); nodeCounter++) {
            currentLayer->getNodes().at(nodeCounter)->setBias(biass.at(posCounter++));  // set the bias of the node to the current position of the index then incriment the index
        }
        
    }
    
}

void Network::setAllWeights(std::vector<double> weights) {
    Layer* currentLayer;  // store the current layer
    Node* currentNode;

    int posCounter = 0;

    for (size_t layerCounter = 0; layerCounter < this->getLayers().size()-1; layerCounter++) {
        currentLayer = this->getLayers().at(layerCounter);
        
        for (size_t nodeCounter = 0; nodeCounter < currentLayer->getNodes().size(); nodeCounter++) {
            currentNode = currentLayer->getNodes().at(nodeCounter);

            for (size_t linkCounter = 0; linkCounter < currentNode->getOutLinks().size(); linkCounter++) {
                currentNode->getOutLinks().at(linkCounter)->setWeight(weights.at(posCounter++));
            }
            
        }
        
    }
}