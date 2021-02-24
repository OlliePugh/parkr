#include <stdexcept>

#include "layer.h"

Layer::Layer(int type, int nodeCount) {
    if (type != Layer::INPUT) throw std::invalid_argument("Layer without prev layer must be of type INPUT");
    this->type = type;

    for (int i=0; i<nodeCount; i++) {  // add the input nodes to the first layer
        this->nodes.push_back(new Node(this));
    }
    
}

Layer::Layer(int type, int nodeCount, Layer* prevLayer) {
    if (type == Layer::INPUT) throw std::invalid_argument("Layer with prev layer should not be an input layer");
    this->type = type; 

    for (int i=0; i<nodeCount; i++) { 
        this->nodes.push_back(new Node(this));  // add blank nodes to the layer
    }

    std::vector<Node*> prevNodes = prevLayer->getNodes();

    for (int i=0; i<prevNodes.size(); i++) {  // for each node in the previous layer
        for (int j=0; j<this->nodes.size(); j++) {  // for each node in this layer
            Link* newLink = new Link(prevNodes[i], this->nodes[j]);  // add a link between the two nodes
        }
    }
}