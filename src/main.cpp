#include <iostream>
#include "dataset.hpp"

int main(int argc, char **argv) {
    std::cout << "Hello World!" << std::endl;
    capstone::base::DatasetImage i("/home/bingostarr/bingo/cppnd/5_capstone_project/data/train-images-idx3-ubyte");
    i.wait();
    std::cout << i(2).show() << std::endl;
    capstone::base::DatasetLabel l("/home/bingostarr/bingo/cppnd/5_capstone_project/data/train-labels-idx1-ubyte");
    l.wait();
    std::cout << l(2) << std::endl;
    return 0;
}
