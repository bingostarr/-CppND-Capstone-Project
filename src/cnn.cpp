/*
 * cnn.cpp
 *
 *  Created on: Feb 23, 2020
 *      Author: bingo
 */

#include "cnn.hpp"
#include <cassert>
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

void Cnn::train(DatasetImage& images,
                DatasetLabel& labels) {
    uint32_t nImages = images.getNImages();
    uint32_t nBatches = 1;//nImages/BATCHSIZE;
    double loss = 0.0;
    double lossCumul = 0.0;
    double lGain = 100.0;
    int nlayers = m_layers.size();
    double lrate = LRATE/BATCHSIZE;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        for (int bIndex = 0; bIndex < nBatches; ++bIndex) {
//            arma::vec batch(BATCH_SIZE, arma::fill::randu);
//            batch *= (TRAIN_DATA_SIZE - 1);
            for (int imgIndex = 0; imgIndex < 1; ++imgIndex) {
                for (int lIndex = 0; lIndex < nlayers; ++lIndex) {
                    if (0 == lIndex) {
                        m_layers[lIndex]->forward(images(imgIndex), m_outputs[lIndex]);
                    } else {
                        m_layers[lIndex]->forward(m_outputs[lIndex - 1], m_outputs[lIndex]);
                    }
                    if (LAYERTYPE::FULL == m_layers[lIndex]->getType()) {
                        m_outputs[lIndex] = m_outputs[lIndex]/lGain;
                    }
                    if (LAYERTYPE::LOSS == m_layers[lIndex]->getType()) {
                        std::shared_ptr<LayerLoss> x = std::static_pointer_cast<LayerLoss>(m_layers[lIndex]);
                        loss = x->getLoss();
                        lossCumul += loss;
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
                for (int lIndex = 0; lIndex < nlayers; ++lIndex) {
                    m_layers[lIndex]->update(lrate);
                }
            }
        }
    }

#if 0

          r1.Forward(c1Out, r1Out);
          mp1.Forward(r1Out, mp1Out);
          c2.Forward(mp1Out, c2Out);
          r2.Forward(c2Out, r2Out);
          mp2.Forward(r2Out, mp2Out);
          d.Forward(mp2Out, dOut);
          dOut /= 100;
          s.Forward(dOut, sOut);

          // Compute the loss
          l.Forward(sOut, trainLabels[batch[i]]);
          lossCumul += loss;

          // Backward pass
          l.Backward();
          arma::vec gradWrtPredictedDistribution =
              l.getGradientWrtPredictedDistribution();
          s.Backward(gradWrtPredictedDistribution);
          arma::vec gradWrtSIn = s.getGradientWrtInput();
          d.Backward(gradWrtSIn);
          arma::cube gradWrtDIn = d.getGradientWrtInput();
          mp2.Backward(gradWrtDIn);
          arma::cube gradWrtMP2In = mp2.getGradientWrtInput();
          r2.Backward(gradWrtMP2In);
          arma::cube gradWrtR2In = r2.getGradientWrtInput();
          c2.Backward(gradWrtR2In);
          arma::cube gradWrtC2In = c2.getGradientWrtInput();
          mp1.Backward(gradWrtC2In);
          arma::cube gradWrtMP1In = mp1.getGradientWrtInput();
          r1.Backward(gradWrtMP1In);
          arma::cube gradWrtR1In = r1.getGradientWrtInput();
          c1.Backward(gradWrtR1In);
          arma::cube gradWrtC1In = c1.getGradientWrtInput();
        }

        // Update params
        d.UpdateWeightsAndBiases(BATCH_SIZE, LEARNING_RATE);
        c1.UpdateFilterWeights(BATCH_SIZE, LEARNING_RATE);
        c2.UpdateFilterWeights(BATCH_SIZE, LEARNING_RATE);
      }

  #if DEBUG
      // Output loss on training dataset after each epoch
      std::cout << DEBUG_PREFIX << std::endl;
      std::cout << DEBUG_PREFIX << "Training loss: "
          << lossCumul / (BATCH_SIZE * NUM_BATCHES) << std::endl;
  #endif

      // Compute the training accuracy after epoch
      double correct = 0.0;
      for (size_t i = 0; i < TRAIN_DATA_SIZE; i++)
      {
        // Forward pass
        c1.Forward(trainData[i], c1Out);
        r1.Forward(c1Out, r1Out);
        mp1.Forward(r1Out, mp1Out);
        c2.Forward(mp1Out, c2Out);
        r2.Forward(c2Out, r2Out);
        mp2.Forward(r2Out, mp2Out);
        d.Forward(mp2Out, dOut);
        dOut /= 100;
        s.Forward(dOut, sOut);

        if (trainLabels[i].index_max() == sOut.index_max())
          correct += 1.0;
      }

  #if DEBUG
      // Output accuracy on training dataset after each epoch
      std::cout << DEBUG_PREFIX
          << "Training accuracy: " << correct/TRAIN_DATA_SIZE << std::endl;
  #endif

      // Compute validation accuracy after epoch
      lossCumul = 0.0;
      correct = 0.0;
      for (size_t i = 0; i < VALIDATION_DATA_SIZE; i++)
      {
        // Forward pass
        c1.Forward(validationData[i], c1Out);
        r1.Forward(c1Out, r1Out);
        mp1.Forward(r1Out, mp1Out);
        c2.Forward(mp1Out, c2Out);
        r2.Forward(c2Out, r2Out);
        mp2.Forward(r2Out, mp2Out);
        d.Forward(mp2Out, dOut);
        dOut /= 100;
        s.Forward(dOut, sOut);

        lossCumul += l.Forward(sOut, validationLabels[i]);

        if (validationLabels[i].index_max() == sOut.index_max())
          correct += 1.0;
      }

  #if DEBUG
      // Output validation loss after each epoch
      std::cout << DEBUG_PREFIX
          << "Validation loss: " << lossCumul / (BATCH_SIZE * NUM_BATCHES)
          << std::endl;

      // Output validation accuracy after each epoch
      std::cout << DEBUG_PREFIX
          << "Val accuracy: " << correct / VALIDATION_DATA_SIZE << std::endl;
      std::cout << DEBUG_PREFIX << std::endl;
  #endif

      // Reset cumulative loss and correct count
      lossCumul = 0.0;
      correct = 0.0;

      // Write results on test data to results csv
      std::fstream fout("results_epoch_" + std::to_string(epoch) + ".csv",
                        std::ios::out);
      fout << "ImageId,Label" << std::endl;
      for (size_t i=0; i<TEST_DATA_SIZE; i++)
      {
        // Forward pass
        c1.Forward(testData[i], c1Out);
        r1.Forward(c1Out, r1Out);
        mp1.Forward(r1Out, mp1Out);
        c2.Forward(mp1Out, c2Out);
        r2.Forward(c2Out, r2Out);
        mp2.Forward(r2Out, mp2Out);
        d.Forward(mp2Out, dOut);
        dOut /= 100;
        s.Forward(dOut, sOut);

        fout << std::to_string(i+1) << ","
            << std::to_string(sOut.index_max()) << std::endl;
      }
      fout.close();
    }
#endif
}

void Cnn::test(DatasetImage& images,
               DatasetLabel& labels) {

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
