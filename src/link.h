#ifndef _LINKH_
#define _LINKH_

#include "node.h"

class Link {
    private:
        double weight;
        double bias;
        Node* child;
        Node* parent;

    public:
        Link(Node*, Node*);
        Node* getChild() { return this->child; };
        Node* getParent() { return this->parent; };
        double getBias() { return this->bias; };
        double getWeight() { return this->weight; };
};

#endif