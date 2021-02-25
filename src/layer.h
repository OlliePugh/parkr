#ifndef _LAYERH_
#define _LAYERH_

#include <vector>

class Node;
class Network;

class Layer {
    public:
        enum types { INPUT, // store the different types of layer types
                    HIDDEN,
                    OUTPUT };

    private:
        std::vector<Node*> nodes;
        int type;
        Network* network;
    
    public:
        Layer() {};
        Layer(Network*, int, int);
        Layer(Network*, int, int, Layer*);

        std::vector<Node*> getNodes() {return this->nodes;};
        int getType() { return this->type; };
        Network* getNetwork() { return this->network; };
};

#include "node.h"
#include "network.h"

#endif