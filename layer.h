#ifndef _LAYERH_
#define _LAYERH_

#include <vector>

class Node;

class Layer {
    private:
        std::vector<Node*> nodes;
        int type;
    
    public:
        Layer() {};
        Layer(int, int);
        Layer(int, int, Layer*);
        enum types { INPUT, // store the different types of layer types
                    HIDDEN,
                    OUTPUT };
        
        std::vector<Node*> getNodes() {return this->nodes;};
        int getType() { return this->type; };
};

#include "node.h"

#endif