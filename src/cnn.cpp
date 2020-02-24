/*
 * cnn.cpp
 *
 *  Created on: Feb 23, 2020
 *      Author: bingo
 */

#include "cnn.hpp"
#include <cassert>

namespace capstone {
namespace base {

Cnn::Cnn() {
    m_layers.clear();

    LayerAttr_t lac0(LAYERTYPE::CONV, INPUTLAYERS, INPUTSIZE, NFILTERS1, FILTERSIZE);
    std::shared_ptr<LayerConv> lc0 = std::make_shared<LayerConv>(lac0);
    m_layers.push_back(lc0);

    LayerAttr_t lar0(LAYERTYPE::RELU, lac0.outputSize.getNLayers(), lac0.outputSize.getSize(), 0, 0);
    std::shared_ptr<LayerRelu> lr0 = std::make_shared<LayerRelu>(lar0);
    m_layers.push_back(lr0);

    LayerAttr_t lap0(LAYERTYPE::POOL, lar0.outputSize.getNLayers(), lar0.outputSize.getSize(), 0, POOLSIZE);
    std::shared_ptr<LayerPool> lp0 = std::make_shared<LayerPool>(lap0);
    m_layers.push_back(lp0);

    LayerAttr_t lac1(LAYERTYPE::CONV, lap0.outputSize.getNLayers(), lap0.outputSize.getSize(), NFILTERS2, FILTERSIZE);
    std::shared_ptr<LayerConv> lc1 = std::make_shared<LayerConv>(lac1);
    m_layers.push_back(lc1);

    LayerAttr_t lar1(LAYERTYPE::RELU, lac1.outputSize.getNLayers(), lac1.outputSize.getSize(), 0, 0);
    std::shared_ptr<LayerRelu> lr1 = std::make_shared<LayerRelu>(lar1);
    m_layers.push_back(lr1);

    LayerAttr_t lap1(LAYERTYPE::POOL, lar1.outputSize.getNLayers(), lar1.outputSize.getSize(), 0, POOLSIZE);
    std::shared_ptr<LayerPool> lp1 = std::make_shared<LayerPool>(lap1);
    m_layers.push_back(lp1);

    LayerAttr_t laf(LAYERTYPE::FULL, lap1.outputSize.getNLayers(), lap1.outputSize.getSize(), OUTPUTSIZE, 0);
    std::shared_ptr<LayerFull> lf = std::make_shared<LayerFull>(laf);
    m_layers.push_back(lf);

    LayerAttr_t las(LAYERTYPE::SMAX, laf.outputSize.getNLayers(), laf.outputSize.getSize(), 0, 0);
    std::shared_ptr<LayerSmax> ls = std::make_shared<LayerSmax>(las);
    m_layers.push_back(ls);

    LayerAttr_t lal(LAYERTYPE::LOSS, las.outputSize.getNLayers(), las.outputSize.getSize(), 0, 0);
    std::shared_ptr<LayerLoss> ll = std::make_shared<LayerLoss>(lal);
    m_layers.push_back(ll);

}

void Cnn::train(const DatasetImage& images,
                const DatasetLabel& labels) {
}

void Cnn::test(const DatasetImage& images,
               const DatasetLabel& labels) {

}

std::string Cnn::show() {
    std::string x = "CNN Architecture\n";
    for (std::shared_ptr<Layer> p : m_layers) {
        x += p->show();
    }
    return x;
}

} /* base */
} /* capstone */
