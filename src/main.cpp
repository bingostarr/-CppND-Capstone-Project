#include <iostream>
#include <chrono>
#include "dataset.hpp"
#include "cnn.hpp"

int main(int argc, char **argv) {
    std::cout << "Hello World!" << std::endl;
    auto start = std::chrono::steady_clock::now();
    capstone::base::DatasetImage imgTr("train-images-idx3-ubyte", capstone::base::DATATYPE::TRAIN);
    imgTr.wait();
    std::cout << imgTr.show() << std::endl;
    capstone::base::DatasetLabel labTr("train-labels-idx1-ubyte", capstone::base::DATATYPE::TRAIN);
    labTr.wait();
    std::cout << labTr.show() << std::endl;
    capstone::base::DatapointSet dataTr(imgTr, labTr);

    capstone::base::DatasetImage imgTe("t10k-images-idx3-ubyte", capstone::base::DATATYPE::TEST);
    imgTe.wait();
    std::cout << imgTe.show() << std::endl;
    capstone::base::DatasetLabel labTe("t10k-labels-idx1-ubyte", capstone::base::DATATYPE::TEST);
    labTe.wait();
    std::cout << labTe.show() << std::endl;
    capstone::base::DatapointSet dataTe(imgTe, labTe);

    capstone::base::Cnn& cnn = capstone::base::Cnn::getInstance();
    std::cout << cnn.show() << std::endl;
    cnn.train(dataTr);
    std::cout << cnn.showAll() << std::endl;
    int nTests = imgTe.getNImages();
    capstone::base::TestResult_t t(cnn.test(dataTe.getValidData()));
    std::cout << t.showAll() << std::endl;
    auto end = std::chrono::steady_clock::now();
    std::cout << "elapsed time: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s" << std::endl;
    return 0;
}
