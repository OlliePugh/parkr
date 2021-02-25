#ifndef _NODEH_
#define _NODEH_

#include <vector>
#include "layer.h"
class Link;

class Node {
    private:
        std::vector<Link*> inputLinks;  // store the links going into the node
        std::vector<Link*> outputLinks; // store the links going out of the node
        Layer* layer;
        double value;

    public:
        Node(Layer* _layer) { this->layer = _layer; };
        void addInLink(Link*);
        void addOutLink(Link*);
        double calcValue();
        void setValue(double _value) { this->value = _value; };  // set the value of a node
        double getValue() { return this->value; };
        std::vector<Link*> getInLinks() { return this->inputLinks;};
        std::vector<Link*> getOutLinks() { return this->outputLinks;};
        Layer* getLayer() { return this->layer; };  // return pointer to the layer this node is in 
};

#include "link.h"

#endif