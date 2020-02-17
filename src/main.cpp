#include <iostream>
#include <cassert>
#include "dataset.hpp"
#include "imgproc.hpp"

int main(int argc, char **argv) {
    std::cout << "Hello World!" << std::endl;
    capstone::base::DatasetImage img("/home/bingostarr/bingo/cppnd/5_capstone_project/CppND-Capstone-Project/data/train-images-idx3-ubyte");
    img.wait();
    capstone::base::DatasetLabel lab("/home/bingostarr/bingo/cppnd/5_capstone_project/CppND-Capstone-Project/data/train-labels-idx1-ubyte");
    lab.wait();
    assert(img.getNImages() == lab.getNImages());
    for (int i = 0; i < 10; ++i) {
        capstone::base::show(img(i).getRows(), img(i).getCols(), img(i).vectorizeuc(), std::to_string(static_cast<int>(lab(i))));
    }
    return 0;
}
