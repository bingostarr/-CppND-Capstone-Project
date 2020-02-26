#include <iostream>
#include <cassert>
#include "dataset.hpp"
#include "cnn.hpp"

int main(int argc, char **argv) {
    std::cout << "Hello World!" << std::endl;
    capstone::base::DatasetImage imgTr("train-images-idx3-ubyte", capstone::base::DATATYPE::TRAIN);
    imgTr.wait();
    std::cout << imgTr.show() << std::endl;
    capstone::base::DatasetLabel labTr("train-labels-idx1-ubyte", capstone::base::DATATYPE::TRAIN);
    labTr.wait();
    std::cout << labTr.show() << std::endl;
    assert(imgTr.getNImages() == labTr.getNImages());
    capstone::base::DatasetImage imgTe("t10k-images-idx3-ubyte", capstone::base::DATATYPE::TEST);
    imgTe.wait();
    std::cout << imgTe.show() << std::endl;
    capstone::base::DatasetLabel labTe("t10k-labels-idx1-ubyte", capstone::base::DATATYPE::TEST);
    labTe.wait();
    std::cout << labTe.show() << std::endl;
    assert(imgTe.getNImages() == labTe.getNImages());
    capstone::base::Cnn& cnn = capstone::base::Cnn::getInstance();
    std::cout << cnn.show() << std::endl;
    cnn.train(imgTr, labTr);
    int nTests = 10;
    capstone::base::TestResult_t t = cnn.test(imgTe, labTe, nTests);
    std::cout << t.showAll() << std::endl;
    return 0;
}
