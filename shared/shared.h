#ifndef __SHARED_H__
#define __SHARED_H__

#include <stddef.h>
#include <stdint.h>

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>

#include "../ggml_extend.hpp"
#include "../stable-diffusion.h"

#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef SD_BUILD_SHARED_LIB
        #define _SHARED __declspec(dllexport)
    #else
        #define _SHARED
    #endif

#else
    #ifdef SD_BUILD_SHARED_LIB
        #if __GNUC__ >= 4
            #define _SHARED __attribute__((visibility("default")))
        #else
            #define _SHARED
        #endif
    #endif
#endif

struct ScopedContext { 
    ggml_context *ctx = NULL; 
    ggml_init_params params = {0}; 

    std::unordered_map<TensorType, std::vector<ggml_tensor*>> tensors;
};

sd_image_t* _convertToImage(
    struct ggml_context* work_ctx, 
    std::vector<struct ggml_tensor*> _latents, 
    int width, 
    int height, 
    size_t batch);
    
//
#ifdef __cplusplus
extern "C" {
#endif

_SHARED void delete_shared();
_SHARED void clean_shared();


_SHARED void clean_tensors(int key, TensorType type);
_SHARED int create_empty_latent( 
                    int width, int height, 
                    int channels, int batch_count);


_SHARED sd_image_t* create_images(int width, int height);
                    
_SHARED void set_shared_context_key(int key);
_SHARED int get_shared_context_key();


#ifdef __cplusplus
}
#endif

#endif  // __SHARED_H__