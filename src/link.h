#ifndef _LINKH_
#define _LINKH_

#include "node.h"

class Link {
    private:
        double weight;
        Node* child;
        Node* parent;

    public:
        Link(Node*, Node*);
        Node* getChild() { return this->child; };
        Node* getParent() { return this->parent; };
        double getWeight() { return this->weight; };
        void setWeight(double _weight) { this->weight = _weight; };
};

#endif