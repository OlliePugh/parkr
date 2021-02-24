#include <stdlib.h> 
#include <time.h>

#include "link.h"


Link::Link(Node* parent, Node* child) {
    this->parent = parent;
    this->child = child;

    Layer* parentLayer = parent->getLayer();

    double maxBiasWeight = 2/((double) (parent->getLayer()->getNodes().size()));  // get the amount of nodes in the previous layer and set 2 over that to the max and min value

    this->weight = (double)rand()/RAND_MAX*(maxBiasWeight*2)-maxBiasWeight;//float in range -1 to 1
    this->bias =  (double)rand()/RAND_MAX*(maxBiasWeight*2)-maxBiasWeight;//float in range -1 to 1

    parent->addOutLink(this);  // add the link to the out links of the parent node
    child->addInLink(this);  // add the link to the in of the child node
}