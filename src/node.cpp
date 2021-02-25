#include <iostream>
#include <stdexcept>
#include "node.h"

void Node::addInLink(Link* newLink) {  // add a link to the list of input links
    this->inputLinks.push_back(newLink);
}

void Node::addOutLink(Link* newLink) {  // add a link to the list of output links
    this->outputLinks.push_back(newLink);
}

double Node::calcValue() {  // calculate the value that this node represents
    if (this->layer->getType() == Layer::INPUT) {  // if the layer is in the input need to get input value from user
        throw std::logic_error("Can't calculate value of node in input layer");
    }

    this->value = 0;

    for (size_t i = 0; i < this->inputLinks.size(); i++) {  // for each link in the inputs 
        //std::cout << this->inputLinks.at(i)->getBias() <<  " + (" << this->inputLinks.at(i)->getWeight() << " * " << inputLinks.at(i)->getParent()->getValue() << ") = " << this->inputLinks.at(i)->getBias() + (this->inputLinks.at(i)->getWeight() * inputLinks.at(i)->getParent()->getValue()) << std::endl;
        this->value += this->inputLinks.at(i)->getBias() + (this->inputLinks.at(i)->getWeight() * inputLinks.at(i)->getParent()->getValue());  // get the value of the previous node and add it to the 
    }

    
    return this->value;
}