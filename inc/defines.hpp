/*
 * defines.hpp
 *
 *  Created on: Feb 18, 2020
 *      Author: bingo
 */

#ifndef INC_DEFINES_HPP_
#define INC_DEFINES_HPP_

#include <cassert>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <utility>

namespace capstone {
namespace base {

static const uint32_t CNNLAYERS = 9;
static const uint32_t BATCHSIZE = 10;
static const uint32_t INPUTSIZE = 28;
static const uint32_t INPUTLAYERS = 1;
static const uint32_t FILTERSIZE = 5;
static const uint32_t NFILTERS1 = 6;
static const uint32_t NFILTERS2 = 16;
static const uint32_t POOLSIZE = 2;
static const uint32_t OUTPUTSIZE = 10;
static const double SCALE = 255.0;
static const double LRATE = 0.05;

enum class MTXTYPE {
    ZEROS,
    ONES,
    ID,
    RANDN
};

enum class DATATYPE {
    TRAIN,
    TEST
};

enum class LAYERTYPE {
    CONV,
    RELU,
    POOL,
    FULL,
    SMAX,
    LOSS
};

enum class POOLTYPE {
    AVG,
    MAX
};

typedef struct ImageSize {
    uint32_t size;
    uint32_t npixels;
    ImageSize(const ImageSize& i)
            : size(i.size),
              npixels(size * size) { }
    ImageSize(ImageSize&& i)
            : size(i.size),
              npixels(size * size) { }
    ImageSize(const uint32_t& n)
            : size(n),
              npixels(size * size) { }
    std::string show() {
        std::string x = std::to_string(size) + "X" + std::to_string(size);
        return x;
    }
    const uint32_t& getSize() const {
        return size;
    }
    const uint32_t& nPixels() const {
        return npixels;
    }
    const bool inRange(const uint32_t& i,
                       const uint32_t& j) const {
        return (i >= 0 && i < size && j >= 0 && j < size);
    }
    inline const bool operator==(const ImageSize& rhs) const {
        return (size == rhs.size);
    }
    inline const bool operator<(const ImageSize& rhs) const {
        return (size < rhs.size);
    }
    inline const bool operator>(const ImageSize& rhs) const {
        return (size > rhs.size);
    }
} ImageSize_t;

typedef struct CubeSize {
    uint32_t nlayers;
    ImageSize_t size;
    CubeSize(const CubeSize& c)
            : nlayers(c.nlayers),
              size(c.size) { }
    CubeSize(CubeSize&& c)
            : nlayers(c.nlayers),
              size(c.size) { }
    CubeSize(const uint32_t& n,
             const uint32_t& i)
            : nlayers(n),
              size(ImageSize_t(i)) { }
    std::string show() {
        std::string x = std::to_string(nlayers) + "X" + size.show();
        return x;
    }
    const uint32_t& getNLayers() const {
        return nlayers;
    }
    const ImageSize_t& getImageSize() const {
        return size;
    }
    const uint32_t& getSize() const {
        return size.getSize();
    }
    const bool inRange(const uint32_t& k,
                       const uint32_t& i,
                       const uint32_t& j) const {
        return (k >= 0 && k < nlayers &&
                i >= 0 && i < size.getSize() &&
                j >= 0 && j < size.getSize());
    }
    inline const bool operator==(const CubeSize& rhs) const {
        return ((nlayers == rhs.nlayers) && (size == rhs.size));
    }
    inline const bool operator<(const CubeSize& rhs) const {
        return ((nlayers < rhs.nlayers) && (size < rhs.size));
    }
    inline const bool operator>(const CubeSize& rhs) const {
        return ((nlayers > rhs.nlayers) && (size > rhs.size));
    }
} CubeSize_t;

typedef struct LayerAttr {
    LAYERTYPE type;
    CubeSize_t inputSize;
    CubeSize_t outputSize;
    CubeSize_t weightSize;
    LayerAttr(const LAYERTYPE& lyr,
              const uint32_t& inputLayers,
              const uint32_t& inputImgSize,
              const uint32_t& nfeatures,
              const uint32_t& featureSize)
            : type(lyr),
              inputSize(CubeSize_t(inputLayers, inputImgSize)),
              outputSize(inputSize),
              weightSize(inputSize) {
        switch(type) {
        case LAYERTYPE::CONV:
            outputSize.nlayers = nfeatures;
            outputSize.size.size = inputSize.getSize() - featureSize + 1;
            weightSize.nlayers = inputSize.getNLayers();
            weightSize.size.size = featureSize;
            break;
        case LAYERTYPE::FULL:
            outputSize.nlayers = nfeatures;
            outputSize.size.size = 1;
            break;
        case LAYERTYPE::POOL:
            assert(inputSize.getSize() % featureSize == 0);
            outputSize.nlayers = inputSize.getNLayers();
            outputSize.size.size = inputSize.getSize()/featureSize;
            weightSize.nlayers = inputSize.getNLayers();
            weightSize.size.size = featureSize;
            break;
        case LAYERTYPE::SMAX:
        case LAYERTYPE::LOSS:
            assert(inputSize.getSize() == 1);
            break;
        default:
            break;
        }
    }
} LayerAttr_t;

typedef struct Synch {
    std::mutex mx;
    std::condition_variable cv;
    bool ready = {false};
    void set() {
        std::unique_lock<std::mutex> lock(mx);
        ready = true;
        cv.notify_one();
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mx);
        if (!ready) {
            cv.wait(lock, [this] {return (this->ready);});
        }
    }
} Synch_t;

typedef struct Coords {
    uint32_t i;
    uint32_t j;
    Coords(const uint32_t& ii, const uint32_t& jj)
            : i(ii), j(jj) {
    }
} Coords_t;
} /* base */
} /* capstone */
#endif /* INC_DEFINES_HPP_ */
