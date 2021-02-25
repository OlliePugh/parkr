#ifndef _LAYERH_
#define _LAYERH_

#include <vector>

class Node;
class Network;

enum LayerType {
    INPUT, 
    HIDDEN, 
    OUTPUT
};

class Layer {
    public:

    private:
        std::vector<Node*> nodes;
        LayerType type;
        Network* network;
    
    public:
        Layer() {};
        Layer(Network*, LayerType, int);
        Layer(Network*, LayerType, int, Layer*);

        std::vector<Node*> getNodes() {return this->nodes;};
        int getType() { return this->type; };
        Network* getNetwork() { return this->network; };
};

#include "node.h"
#include "network.h"

#endif