#include <iostream>
#include <cassert>
#include "dataset.hpp"
#include "imgproc.hpp"
#include "cnn.hpp"
int main(int argc, char **argv) {
    std::cout << "Hello World!" << std::endl;
//    capstone::base::DatasetImage imgTr("train-images-idx3-ubyte", capstone::base::DATATYPE::TRAIN);
//    imgTr.wait();
//    std::cout << imgTr.show() << std::endl;
//    capstone::base::DatasetLabel labTr("train-labels-idx1-ubyte", capstone::base::DATATYPE::TRAIN);
//    labTr.wait();
//    std::cout << labTr.show() << std::endl;
//    assert(imgTr.getNImages() == labTr.getNImages());
//    for (int i = 0; i < 3; ++i) {
//        capstone::base::show(imgTr(i), std::to_string(labTr(i)));
//    }
//    capstone::base::DatasetImage imgTe("t10k-images-idx3-ubyte", capstone::base::DATATYPE::TEST);
//    imgTe.wait();
//    std::cout << imgTe.show() << std::endl;
//    capstone::base::DatasetLabel labTe("t10k-labels-idx1-ubyte", capstone::base::DATATYPE::TEST);
//    labTe.wait();
//    std::cout << labTe.show() << std::endl;
//    assert(imgTe.getNImages() == labTe.getNImages());
//    for (int i = 0; i < 3; ++i) {
//        capstone::base::show(imgTe(i), std::to_string(labTe(i)));
//    }
    capstone::base::Cnn cnn;
    std::cout << cnn.show() << std::endl;
    return 0;
}
