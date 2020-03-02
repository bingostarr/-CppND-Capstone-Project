/*
 * cnn.cpp
 *
 *  Created on: Feb 23, 2020
 *      Author: bingo
 */

#include "cnn.hpp"
#include <cassert>
#include <algorithm>
#include <iostream>
#include <cmath>
#include "defines.hpp"

namespace capstone {
namespace base {

Cnn::Cnn() {
    m_layers.clear();
    m_outputs.clear();

    LayerAttr_t lac0(LAYERTYPE::CONV, INPUTLAYERS, INPUTSIZE, NFILTERS1, FILTERSIZE);
    std::shared_ptr<LayerConv> lc0 = std::make_shared<LayerConv>(lac0);
    m_layers.push_back(lc0);
    Matrix3d olc0(lac0.outputSize);
    m_outputs.push_back(olc0);

    LayerAttr_t lar0(LAYERTYPE::RELU, lac0.outputSize.getNLayers(), lac0.outputSize.getSize(), 0, 0);
    std::shared_ptr<LayerRelu> lr0 = std::make_shared<LayerRelu>(lar0);
    m_layers.push_back(lr0);
    Matrix3d olr0(lar0.outputSize);
    m_outputs.push_back(olr0);

    LayerAttr_t lap0(LAYERTYPE::POOL, lar0.outputSize.getNLayers(), lar0.outputSize.getSize(), 0, POOLSIZE);
    std::shared_ptr<LayerPool> lp0 = std::make_shared<LayerPool>(lap0);
    m_layers.push_back(lp0);
    Matrix3d olp0(lap0.outputSize);
    m_outputs.push_back(olp0);

    LayerAttr_t lac1(LAYERTYPE::CONV, lap0.outputSize.getNLayers(), lap0.outputSize.getSize(), NFILTERS2, FILTERSIZE);
    std::shared_ptr<LayerConv> lc1 = std::make_shared<LayerConv>(lac1);
    m_layers.push_back(lc1);
    Matrix3d olc1(lac1.outputSize);
    m_outputs.push_back(olc1);

    LayerAttr_t lar1(LAYERTYPE::RELU, lac1.outputSize.getNLayers(), lac1.outputSize.getSize(), 0, 0);
    std::shared_ptr<LayerRelu> lr1 = std::make_shared<LayerRelu>(lar1);
    m_layers.push_back(lr1);
    Matrix3d olr1(lar1.outputSize);
    m_outputs.push_back(olr1);

    LayerAttr_t lap1(LAYERTYPE::POOL, lar1.outputSize.getNLayers(), lar1.outputSize.getSize(), 0, POOLSIZE);
    std::shared_ptr<LayerPool> lp1 = std::make_shared<LayerPool>(lap1);
    m_layers.push_back(lp1);
    Matrix3d olp1(lap1.outputSize);
    m_outputs.push_back(olp1);

    LayerAttr_t laf(LAYERTYPE::FULL, lap1.outputSize.getNLayers(), lap1.outputSize.getSize(), OUTPUTSIZE, 0);
    std::shared_ptr<LayerFull> lf = std::make_shared<LayerFull>(laf);
    m_layers.push_back(lf);
    Matrix3d olf(laf.outputSize);
    m_outputs.push_back(olf);

    LayerAttr_t las(LAYERTYPE::SMAX, laf.outputSize.getNLayers(), laf.outputSize.getSize(), 0, 0);
    std::shared_ptr<LayerSmax> ls = std::make_shared<LayerSmax>(las);
    m_layers.push_back(ls);
    Matrix3d ols(las.outputSize);
    m_outputs.push_back(ols);

    LayerAttr_t lal(LAYERTYPE::LOSS, las.outputSize.getNLayers(), las.outputSize.getSize(), 0, 0);
    std::shared_ptr<LayerLoss> ll = std::make_shared<LayerLoss>(lal);
    m_layers.push_back(ll);
    Matrix3d oll(lal.outputSize);
    m_outputs.push_back(oll);

}

void Cnn::train(DatapointSet& data) {
    uint32_t nImages = data.getTrainSize();
    uint32_t nBatches = nImages/BATCHSIZE;
    double loss = 0.0;
    double lossCumul = 0.0;
    int nlayers = m_layers.size();
    double lrate = LRATE/BATCHSIZE;
    std::cout << "TRAIN START" << std::endl;
    std::cout << "epoch\tbatch\timage\tinput\toutput\tpass\tloss" << std::endl;
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        data.shuffle();
        for (int bIndex = 0; bIndex < nBatches; ++bIndex) {
            for (int imgIndex = 0; imgIndex < BATCHSIZE; ++imgIndex) {
                Datapoint_t datapoint = data.nextTrainData();
                int inp = static_cast<int>(datapoint.first);
                Matrix3d inputImage(datapoint.second);
                int outp = -1;
                std::cout << epoch << "\t" << bIndex << "\t" << imgIndex << "\t" << inp << "\t";
                for (int lIndex = 0; lIndex < nlayers; ++lIndex) {
                    if (0 == lIndex) {
                        m_layers[lIndex]->forward(inputImage, m_outputs[lIndex]);
                    } else {
                        if (nlayers - 1 == lIndex) {
                            m_outputs[lIndex].zero();
                            m_outputs[lIndex].at(inp, 0, 0) = 1;
                        }
                        m_layers[lIndex]->forward(m_outputs[lIndex - 1], m_outputs[lIndex]);
                    }
                    if (LAYERTYPE::LOSS == m_layers[lIndex]->getType()) {
                        std::shared_ptr<LayerLoss> x = std::static_pointer_cast<LayerLoss>(m_layers[lIndex]);
                        loss = x->getLoss();
                        if (std::isnan(loss)) {
                            std::cout << "\n" << showAll() << std::endl;;
                            assert(false);
                        }
                        lossCumul += loss;
                        std::vector<double> v = m_outputs[nlayers - 2].vectorize();
                        outp = std::max_element(v.begin(), v.end()) - v.begin();
                    }
                }
                Matrix3d y(CubeSize_t(1,1));
                for (int lIndex = nlayers - 1; lIndex >= 0; lIndex--) {
                    if (nlayers - 1 == lIndex) {
                        m_layers[lIndex]->backward(y);
                    } else {
                        m_layers[lIndex]->backward(m_layers[lIndex + 1]->getGradient());
                    }
                }
                std::cout << outp << "\t" << (inp == outp) << "\t" << loss << std::endl;
            }
//            std::cout << "\nbefore update\n" << showAll() << std::endl;;
            for (int lIndex = 0; lIndex < nlayers; ++lIndex) {
                m_layers[lIndex]->update(lrate);
            }
//            std::cout << "\nafter update\n" << showAll() << std::endl;;
        }
        std::cout << "epoch: " << epoch << "\t" << lossCumul / (nImages) << "\t";
        TestResult_t t(test(data.getTrainData(), false));
        std::cout << t.getAvgLoss() << "\t" << t.getAccuracy() << "\t";
        TestResult_t t1(test(data.getValidData(), false));
        std::cout << t1.getAvgLoss() << "\t" << t1.getAccuracy() << std::endl;
        loss = 0;
        lossCumul = 0;
    }
    std::cout << "\nTRAIN END" << std::endl;
}

TestResult_t Cnn::test(const std::vector<Datapoint_t>& data,
                       const bool& show) {
    uint32_t nImages = data.size();
    TestResult_t t {};
    double loss = 0.0;
    int nlayers = m_layers.size();
    if (show) {
        std::cout << "TEST" << std::endl;
        std::cout << "image\tinput\toutput\tpass\tloss" << std::endl;
    }
    for (int imgIndex = 0; imgIndex < nImages; ++imgIndex) {
        Datapoint_t datapoint = data[imgIndex];
        int inp = static_cast<int>(datapoint.first);
        Matrix3d inputImage(datapoint.second);
        int outp = -1;
        if (show) {
            std::cout << imgIndex << "\t" << inp;
        }
        for (int lIndex = 0; lIndex < nlayers; ++lIndex) {
            if (0 == lIndex) {
                m_layers[lIndex]->forward(inputImage, m_outputs[lIndex]);
            } else {
                if (nlayers - 1 == lIndex) {
                    m_outputs[lIndex].zero();
                    m_outputs[lIndex].at(inp, 0, 0) = 1;
                }
                m_layers[lIndex]->forward(m_outputs[lIndex - 1], m_outputs[lIndex]);
            }
            if (LAYERTYPE::LOSS == m_layers[lIndex]->getType()) {
                std::shared_ptr<LayerLoss> x = std::static_pointer_cast<LayerLoss>(m_layers[lIndex]);
                loss = x->getLoss();
            }
        }
        std::vector<double> v = m_outputs[nlayers - 2].vectorize();
        outp = std::max_element(v.begin(), v.end()) - v.begin();
        t.log(inp, outp, loss);
        if (show) {
            std::cout << "\t" << outp << "\t" << (inp == outp) << "\t" << loss << std::endl;
        }
    }
    return t;
}

std::string Cnn::show() {
    std::string x = "CNN Architecture\n";
    for (std::shared_ptr<Layer> p : m_layers) {
        x += p->show();
    }
    return x;
}

std::string Cnn::showAll() {
    std::string x = "CNN Values\n";
    for (std::shared_ptr<Layer> p : m_layers) {
        x += p->showAll();
    }
    return x;
}

} /* base */
} /* capstone */
