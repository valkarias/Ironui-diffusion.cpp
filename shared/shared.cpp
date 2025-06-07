// #include "shared.h"
#include "shared.hpp"
#include "../util.h"

SharedData* SharedData::_instance = nullptr;


sd_image_t* _convertToImage(ggml_context* work_ctx, std::vector<struct ggml_tensor*> _latents, int width, int height, size_t batch) {
    sd_image_t* result_images = (sd_image_t*)calloc(batch, sizeof(sd_image_t));
    if (result_images == NULL) {
        LOG_ERROR("malloc result_images failed");
        return NULL;
    }

    for (size_t i = 0; i < _latents.size(); i++) {
        result_images[i].width   = width;
        result_images[i].height  = height;
        result_images[i].channel = 3;
        result_images[i].data    = sd_tensor_to_image(_latents[i]);
    }

    return result_images;
}

_SHARED sd_image_t* create_images(int width, int height) {
    std::vector<ggml_tensor*> tensors = Shared->getTensors(Shared->getContextkey(), TensorType::IMAGE);
    if (tensors.empty())
        return NULL;
    sd_image_t* results = _convertToImage(Shared->getRawContext(), tensors, width, height, tensors.size());
    
    return results;
}

_SHARED int create_empty_latent(int width, int height, int channels, int batch_count) {
    // Create an empty tensor
    int tensor = Shared->createEmptyTensor(width, height, channels, batch_count);
    if (tensor == EMPTY_INDEX)
        LOG_ERROR("Failed to create empty latent tensor.");

    return tensor;
}

_SHARED void create_context(int width, int height, int batch_count) {
    Shared->createContext(width, height, batch_count);
}

_SHARED int get_shared_context_key() {
   return Shared->getContextkey();
}

_SHARED void set_shared_context_key(int key) {
    Shared->setContextKey(key);
}

_SHARED void clean_tensors(int key, TensorType type) {
    Shared->cleanTensors(key, type);
    LOG_DEBUG("Cleaned tensors of tensor type %d from storage %d", (int)type, key);
}

_SHARED void clean_shared() {
    Shared->cleanInstance();
    LOG_DEBUG("Cleaning shared properties");
}