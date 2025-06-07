#include "ggml_extend.hpp"

#include "stable-diffusion.h"
#include "stable-diffusion.hpp"

#include "model.h"
#include "rng.hpp"
#include "rng_philox.hpp"
#include "util.h"

#include "conditioner.hpp"
#include "control.hpp"
#include "denoiser.hpp"
#include "diffusion_model.hpp"
#include "esrgan.hpp"
#include "lora.hpp"
#include "pmid.hpp"
#include "tae.hpp"
#include "vae.hpp"

#include "shared/shared.h"
#include "shared/shared.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_STATIC
// #include "stb_image_write.h"


/*================================================= SD API ==================================================*/

struct sd_ctx_t {
    StableDiffusionGGML* sd = NULL;
};


sd_ctx_t* new_sd_ctx(
                     const char* model_path_c_str,
                     const char* clip_l_path_c_str,
                     const char* clip_g_path_c_str,
                     const char* t5xxl_path_c_str,
                     const char* diffusion_model_path_c_str,
                     const char* vae_path_c_str,
                     const char* taesd_path_c_str,
                     const char* control_net_path_c_str,
                     const char* lora_model_dir_c_str,
                     const char* embed_dir_c_str,
                     const char* id_embed_dir_c_str,
                     bool vae_decode_only,
                     bool vae_tiling,
                     bool free_params_immediately,
                     int n_threads,
                     enum sd_type_t wtype,
                     enum rng_type_t rng_type,
                     enum schedule_t s,
                     bool keep_clip_on_cpu,
                     bool keep_control_net_cpu,
                     bool keep_vae_on_cpu,
                     bool diffusion_flash_attn) {

    sd_ctx_t* sd_ctx = (sd_ctx_t*)malloc(sizeof(sd_ctx_t));
    if (sd_ctx == NULL) {
        LOG_DEBUG("malloc sd_ctx failed");
        return NULL;
    }

    std::string model_path(model_path_c_str);
    std::string clip_l_path(clip_l_path_c_str);
    std::string clip_g_path(clip_g_path_c_str);
    std::string t5xxl_path(t5xxl_path_c_str);
    std::string diffusion_model_path(diffusion_model_path_c_str);
    std::string vae_path(vae_path_c_str);
    std::string taesd_path(taesd_path_c_str);
    std::string control_net_path(control_net_path_c_str);
    std::string embd_path(embed_dir_c_str);
    std::string id_embd_path(id_embed_dir_c_str);
    std::string lora_model_dir(lora_model_dir_c_str);

    sd_ctx->sd = new StableDiffusionGGML(n_threads,
                                         vae_decode_only,
                                         free_params_immediately,
                                         lora_model_dir,
                                         rng_type);


    if (sd_ctx->sd == NULL) {
        LOG_DEBUG("StableDiffusionGGML failed");
        return NULL;
    }

    if (!sd_ctx->sd->load_from_file(model_path,
                                    clip_l_path,
                                    clip_g_path,
                                    t5xxl_path_c_str,
                                    diffusion_model_path,
                                    vae_path,
                                    control_net_path,
                                    embd_path,
                                    id_embd_path,
                                    taesd_path,
                                    vae_tiling,
                                    (ggml_type)wtype,
                                    s,
                                    keep_clip_on_cpu,
                                    keep_control_net_cpu,
                                    keep_vae_on_cpu,
                                    diffusion_flash_attn)) {
        free_sd_ctx(sd_ctx);
        return NULL;
    }

    // sd_ctx->sd->setModelComponent("first_stage_model", sd_ctx->sd->first_stage_model);
    // sd_ctx->sd->setModelComponent("cond_stage_model", sd_ctx->sd->cond_stage_model);

    return sd_ctx;
}

void free_sd_ctx(sd_ctx_t* sd_ctx) {
    if (sd_ctx == NULL)
        return;

    if (sd_ctx->sd != NULL) {
        LOG_DEBUG("free sd_ctx->sd %p", static_cast<void*>(sd_ctx->sd));
        delete sd_ctx->sd;
        sd_ctx->sd = NULL;
    }
    free(sd_ctx);
}

sd_image_t* generate_image(sd_ctx_t* sd_ctx,
                           struct ggml_context* work_ctx,
                           ggml_tensor* init_latent,
                           std::string prompt,
                           std::string negative_prompt,
                           int clip_skip,
                           float cfg_scale,
                           float guidance,
                           float eta,
                           int width,
                           int height,
                           enum sample_method_t sample_method,
                           const std::vector<float>& sigmas,
                           int64_t seed,
                           int batch_count,
                           const sd_image_t* control_cond,
                           float control_strength,
                           float style_ratio,
                           bool normalize_input,
                           std::string input_id_images_path,
                           std::vector<int> skip_layers = {},
                           float slg_scale              = 0,
                           float skip_layer_start       = 0.01,
                           float skip_layer_end         = 0.2,
                           ggml_tensor* masked_image    = NULL,
                           bool return_latents            = false) {
    if (seed < 0) {
        // Generally, when using the provided command line, the seed is always >0.
        // However, to prevent potential issues if 'stable-diffusion.cpp' is invoked as a library
        // by a third party with a seed <0, let's incorporate randomization here.
        srand((int)time(NULL));
        seed = rand();
    }

    // for (auto v : sigmas) {
    //     std::cout << v << " ";
    // }
    // std::cout << std::endl;

    int sample_steps = sigmas.size() - 1;

    // Apply lora
    auto result_pair                                = extract_and_remove_lora(prompt);
    std::unordered_map<std::string, float> lora_f2m = result_pair.first;  // lora_name -> multiplier

    for (auto& kv : lora_f2m) {
        LOG_DEBUG("lora %s:%.2f", kv.first.c_str(), kv.second);
    }

    prompt = result_pair.second;
    LOG_DEBUG("prompt after extract and remove lora: \"%s\"", prompt.c_str());

    int64_t t0 = ggml_time_ms();
    sd_ctx->sd->apply_loras(lora_f2m);
    int64_t t1 = ggml_time_ms();
    LOG_INFO("apply_loras completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);

    // Photo Maker
    std::string prompt_text_only;
    ggml_tensor* init_img = NULL;
    SDCondition id_cond;
    std::vector<bool> class_tokens_mask;
    if (sd_ctx->sd->stacked_id) {
        if (!sd_ctx->sd->pmid_lora->applied) {
            t0 = ggml_time_ms();
            sd_ctx->sd->pmid_lora->apply(sd_ctx->sd->tensors, sd_ctx->sd->version, sd_ctx->sd->n_threads);
            t1                             = ggml_time_ms();
            sd_ctx->sd->pmid_lora->applied = true;
            LOG_INFO("pmid_lora apply completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->pmid_lora->free_params_buffer();
            }
        }
        // preprocess input id images
        std::vector<sd_image_t*> input_id_images;
        bool pmv2 = sd_ctx->sd->pmid_model->get_version() == PM_VERSION_2;
        if (sd_ctx->sd->pmid_model && input_id_images_path.size() > 0) {
            std::vector<std::string> img_files = get_files_from_dir(input_id_images_path);
            for (std::string img_file : img_files) {
                int c = 0;
                int width, height;
                if (ends_with(img_file, "safetensors")) {
                    continue;
                }
                uint8_t* input_image_buffer = stbi_load(img_file.c_str(), &width, &height, &c, 3);
                if (input_image_buffer == NULL) {
                    LOG_ERROR("PhotoMaker load image from '%s' failed", img_file.c_str());
                    continue;
                } else {
                    LOG_INFO("PhotoMaker loaded image from '%s'", img_file.c_str());
                }
                sd_image_t* input_image = NULL;
                input_image             = new sd_image_t{(uint32_t)width,
                                             (uint32_t)height,
                                             3,
                                             input_image_buffer};
                input_image             = preprocess_id_image(input_image);
                if (input_image == NULL) {
                    LOG_ERROR("preprocess input id image from '%s' failed", img_file.c_str());
                    continue;
                }
                input_id_images.push_back(input_image);
            }
        }
        if (input_id_images.size() > 0) {
            sd_ctx->sd->pmid_model->style_strength = style_ratio;
            int32_t w                              = input_id_images[0]->width;
            int32_t h                              = input_id_images[0]->height;
            int32_t channels                       = input_id_images[0]->channel;
            int32_t num_input_images               = (int32_t)input_id_images.size();
            init_img                               = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, w, h, channels, num_input_images);
            // TODO: move these to somewhere else and be user settable
            float mean[] = {0.48145466f, 0.4578275f, 0.40821073f};
            float std[]  = {0.26862954f, 0.26130258f, 0.27577711f};
            for (int i = 0; i < num_input_images; i++) {
                sd_image_t* init_image = input_id_images[i];
                if (normalize_input)
                    sd_mul_images_to_tensor(init_image->data, init_img, i, mean, std);
                else
                    sd_mul_images_to_tensor(init_image->data, init_img, i, NULL, NULL);
            }
            t0                            = ggml_time_ms();
            auto cond_tup                 = sd_ctx->sd->cond_stage_model->get_learned_condition_with_trigger(work_ctx,
                                                                                                             sd_ctx->sd->n_threads, prompt,
                                                                                                             clip_skip,
                                                                                                             width,
                                                                                                             height,
                                                                                                             num_input_images,
                                                                                                             sd_ctx->sd->diffusion_model->get_adm_in_channels());
            id_cond                       = std::get<0>(cond_tup);
            class_tokens_mask             = std::get<1>(cond_tup);  //
            struct ggml_tensor* id_embeds = NULL;
            if (pmv2) {
                // id_embeds = sd_ctx->sd->pmid_id_embeds->get();
                id_embeds = load_tensor_from_file(work_ctx, path_join(input_id_images_path, "id_embeds.bin"));
                // print_ggml_tensor(id_embeds, true, "id_embeds:");
            }
            id_cond.c_crossattn = sd_ctx->sd->id_encoder(work_ctx, init_img, id_cond.c_crossattn, id_embeds, class_tokens_mask);
            t1                  = ggml_time_ms();
            LOG_INFO("Photomaker ID Stacking, taking %" PRId64 " ms", t1 - t0);
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->pmid_model->free_params_buffer();
            }
            // Encode input prompt without the trigger word for delayed conditioning
            prompt_text_only = sd_ctx->sd->cond_stage_model->remove_trigger_from_prompt(work_ctx, prompt);
            // printf("%s || %s \n", prompt.c_str(), prompt_text_only.c_str());
            prompt = prompt_text_only;  //
            // if (sample_steps < 50) {
            //     LOG_INFO("sampling steps increases from %d to 50 for PHOTOMAKER", sample_steps);
            //     sample_steps = 50;
            // }
        } else {
            LOG_WARN("Provided PhotoMaker model file, but NO input ID images");
            LOG_WARN("Turn off PhotoMaker");
            sd_ctx->sd->stacked_id = false;
        }
        for (sd_image_t* img : input_id_images) {
            free(img->data);
        }
        input_id_images.clear();
    }

    // Get learned condition
    t0               = ggml_time_ms();
    SDCondition cond = sd_ctx->sd->cond_stage_model->get_learned_condition(work_ctx,
                                                                           sd_ctx->sd->n_threads,
                                                                           prompt,
                                                                           clip_skip,
                                                                           width,
                                                                           height,
                                                                           sd_ctx->sd->diffusion_model->get_adm_in_channels());

    SDCondition uncond;
    if (cfg_scale != 1.0) {
        bool force_zero_embeddings = false;
        if (sd_version_is_sdxl(sd_ctx->sd->version) && negative_prompt.size() == 0) {
            force_zero_embeddings = true;
        }
        uncond = sd_ctx->sd->cond_stage_model->get_learned_condition(work_ctx,
                                                                     sd_ctx->sd->n_threads,
                                                                     negative_prompt,
                                                                     clip_skip,
                                                                     width,
                                                                     height,
                                                                     sd_ctx->sd->diffusion_model->get_adm_in_channels(),
                                                                     force_zero_embeddings);
    }
    t1 = ggml_time_ms();
    LOG_DEBUG("get_learned_condition completed, taking %" PRId64 " ms", t1 - t0);

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->cond_stage_model->free_params_buffer();
    }

    // Control net hint
    struct ggml_tensor* image_hint = NULL;
    if (control_cond != NULL) {
        image_hint = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);
        sd_image_to_tensor(control_cond->data, image_hint);
    }

    // Sample
    std::vector<struct ggml_tensor*> final_latents;  // collect latents to decode
    int C = 4;
    if (sd_version_is_sd3(sd_ctx->sd->version)) {
        C = 16;
    } else if (sd_version_is_flux(sd_ctx->sd->version)) {
        C = 16;
    }
    int W = width / 8;
    int H = height / 8;
    LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);
    ggml_tensor* noise_mask = nullptr;
    
    if (sd_version_is_inpaint(sd_ctx->sd->version)) {
        if (masked_image == NULL) {
            int64_t mask_channels = 1;
            if (sd_ctx->sd->version == VERSION_FLUX_FILL) {
                mask_channels = 8 * 8;  // flatten the whole mask
            }
            // no mask, set the whole image as masked
            masked_image = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, init_latent->ne[0], init_latent->ne[1], mask_channels + init_latent->ne[2], 1);
            for (int64_t x = 0; x < masked_image->ne[0]; x++) {
                for (int64_t y = 0; y < masked_image->ne[1]; y++) {
                    if (sd_ctx->sd->version == VERSION_FLUX_FILL) {
                        // TODO: this might be wrong
                        for (int64_t c = 0; c < init_latent->ne[2]; c++) {
                            ggml_tensor_set_f32(masked_image, 0, x, y, c);
                        }
                        for (int64_t c = init_latent->ne[2]; c < masked_image->ne[2]; c++) {
                            ggml_tensor_set_f32(masked_image, 1, x, y, c);
                        }
                    } else {
                        ggml_tensor_set_f32(masked_image, 1, x, y, 0);
                        for (int64_t c = 1; c < masked_image->ne[2]; c++) {
                            ggml_tensor_set_f32(masked_image, 0, x, y, c);
                        }
                    }
                }
            }
        }
        cond.c_concat   = masked_image;
        uncond.c_concat = masked_image;
    } else {
        noise_mask = masked_image;
    }
    for (int b = 0; b < batch_count; b++) {
        int64_t sampling_start = ggml_time_ms();
        int64_t cur_seed       = seed + b;
        LOG_INFO("generating image: %i/%i - seed %" PRId64, b + 1, batch_count, cur_seed);

        sd_ctx->sd->rng->manual_seed(cur_seed);
        struct ggml_tensor* x_t   = init_latent;
        struct ggml_tensor* noise = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1);
        ggml_tensor_set_f32_randn(noise, sd_ctx->sd->rng);

        int start_merge_step = -1;
        if (sd_ctx->sd->stacked_id) {
            start_merge_step = int(sd_ctx->sd->pmid_model->style_strength / 100.f * sample_steps);
            // if (start_merge_step > 30)
            //     start_merge_step = 30;
            LOG_INFO("PHOTOMAKER: start_merge_step: %d", start_merge_step);
        }

        struct ggml_tensor* x_0 = sd_ctx->sd->sample(work_ctx,
                                                     x_t,
                                                     noise,
                                                     cond,
                                                     uncond,
                                                     image_hint,
                                                     control_strength,
                                                     cfg_scale,
                                                     cfg_scale,
                                                     guidance,
                                                     eta,
                                                     sample_method,
                                                     sigmas,
                                                     start_merge_step,
                                                     id_cond,
                                                     skip_layers,
                                                     slg_scale,
                                                     skip_layer_start,
                                                     skip_layer_end,
                                                     noise_mask);

        // struct ggml_tensor* x_0 = load_tensor_from_file(ctx, "samples_ddim.bin");
        // print_ggml_tensor(x_0);
        int64_t sampling_end = ggml_time_ms();
        LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
        final_latents.push_back(x_0);
    }

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }
    int64_t t3 = ggml_time_ms();
    LOG_INFO("generating %" PRId64 " latent images completed, taking %.2fs", final_latents.size(), (t3 - t1) * 1.0f / 1000);

 
    std::vector<struct ggml_tensor*> decoded_images;  // collect decoded images

    //refactor, since we separated decoding from internal generation
    if (!return_latents) {
        // Decode to image
        LOG_INFO("decoding %zu latents", final_latents.size());
        for (size_t i = 0; i < final_latents.size(); i++) {
            t1                      = ggml_time_ms();
            struct ggml_tensor* img = sd_ctx->sd->decode_first_stage(work_ctx, final_latents[i] /* x_0 */);
            // print_ggml_tensor(img);
            if (img != NULL) {
                decoded_images.push_back(img);
            }
            int64_t t2 = ggml_time_ms();
            LOG_INFO("latent %" PRId64 " decoded, taking %.2fs", i + 1, (t2 - t1) * 1.0f / 1000);
        }

        int64_t t4 = ggml_time_ms();
        LOG_INFO("decode_first_stage completed, taking %.2fs", (t4 - t3) * 1.0f / 1000);
        if (sd_ctx->sd->free_params_immediately && !sd_ctx->sd->use_tiny_autoencoder) {
            sd_ctx->sd->first_stage_model->free_params_buffer();
        }
    } else {
        LOG_DEBUG("Returning %zu latents", final_latents.size());
        for (size_t i = 0; i < final_latents.size(); i++) {
            Shared->setTensor(final_latents[i], TensorType::LATENT);
        }

        // so it doesnt get freed pre-maturely
        //if its not set. (in case of separated solo call)
        Shared->setRawContext(work_ctx);
        return NULL;
    }

    return _convertToImage(work_ctx, decoded_images, width, height, batch_count);
}

sd_image_t* txt2img(sd_ctx_t* sd_ctx,
                    const char* prompt_c_str,
                    const char* negative_prompt_c_str,
                    int clip_skip,
                    float cfg_scale,
                    float guidance,
                    float eta,
                    int width,
                    int height,
                    enum sample_method_t sample_method,
                    int sample_steps,
                    int64_t seed,
                    int batch_count,
                    const sd_image_t* control_cond,
                    float control_strength,
                    float style_ratio,
                    bool normalize_input,
                    const char* input_id_images_path_c_str,
                    extra_inference_parameters _extra,
                    int* skip_layers         = NULL,
                    size_t skip_layers_count = 0,
                    float slg_scale          = 0,
                    float skip_layer_start   = 0.01,
                    float skip_layer_end     = 0.2
                ) {
    std::vector<int> skip_layers_vec(skip_layers, skip_layers + skip_layers_count);
    LOG_DEBUG("txt2img %dx%d", width, height);
    if (sd_ctx == NULL) {
        return NULL;
    }

    int C = 4;
    if (sd_version_is_sd3(sd_ctx->sd->version)) {
        C = 16;
    } else if (sd_version_is_flux(sd_ctx->sd->version)) {
        C = 16;
    }
    int W                    = width / 8;
    int H                    = height / 8;

    ggml_tensor* init_latent = NULL;
    int index = _extra.init_latent_index;
    int init_key = _extra.init_key;

    if (index != EMPTY_INDEX) {
        auto tensors = Shared->getTensors(init_key, _extra.init_type);
        if (!tensors.empty())
            init_latent = tensors[index];
    }

    struct ggml_context* work_ctx = Shared->getRawContext();
    int key = Shared->getContextkey();
    if (!work_ctx) {
        Shared->createContext(width, height, batch_count);
        work_ctx = Shared->getRawContext();
    }

    if (!init_latent) {
        LOG_DEBUG("Creating new internal initial latent for ID %d (Not Cached)", key);
        //will get freed with the context nonetheless.
        _create_empty_latent_tensor(work_ctx, width, height, C, &init_latent);
    }

    size_t t0 = ggml_time_ms();
    std::vector<float> sigmas = sd_ctx->sd->denoiser->get_sigmas(sample_steps);

    if (sd_version_is_sd3(sd_ctx->sd->version)) {
        ggml_set_f32(init_latent, 0.0609f);
    } else if (sd_version_is_flux(sd_ctx->sd->version)) {
        ggml_set_f32(init_latent, 0.1159f);
    } else {
        ggml_set_f32(init_latent, 0.f);
    }
    
    if (sd_version_is_inpaint(sd_ctx->sd->version)) {
        LOG_WARN("This is an inpainting model, this should only be used in img2img mode with a mask");
    }

    sd_image_t* result_images = generate_image(sd_ctx,
                                               work_ctx,
                                               init_latent,
                                               prompt_c_str,
                                               negative_prompt_c_str,
                                               clip_skip,
                                               cfg_scale,
                                               guidance,
                                               eta,
                                               width,
                                               height,
                                               sample_method,
                                               sigmas,
                                               seed,
                                               batch_count,
                                               control_cond,
                                               control_strength,
                                               style_ratio,
                                               normalize_input,
                                               input_id_images_path_c_str,
                                               skip_layers_vec,
                                               slg_scale,
                                               skip_layer_start,
                                               skip_layer_end,
                                               NULL,
                                               _extra.return_latents);

    size_t t1 = ggml_time_ms();

    LOG_INFO("txt2img completed in %.2fs", (t1 - t0) * 1.0f / 1000);

    return result_images;
}

sd_image_t* img2img(sd_ctx_t* sd_ctx,
                    sd_image_t init_image,
                    sd_image_t mask,
                    const char* prompt_c_str,
                    const char* negative_prompt_c_str,
                    int clip_skip,
                    float cfg_scale,
                    float guidance,
                    float eta,
                    int width,
                    int height,
                    sample_method_t sample_method,
                    int sample_steps,
                    float strength,
                    int64_t seed,
                    int batch_count,
                    const sd_image_t* control_cond,
                    float control_strength,
                    float style_ratio,
                    bool normalize_input,
                    const char* input_id_images_path_c_str,
                    int* skip_layers         = NULL,
                    size_t skip_layers_count = 0,
                    float slg_scale          = 0,
                    float skip_layer_start   = 0.01,
                    float skip_layer_end     = 0.2) {
    std::vector<int> skip_layers_vec(skip_layers, skip_layers + skip_layers_count);
    LOG_DEBUG("img2img %dx%d", width, height);
    if (sd_ctx == NULL) {
        return NULL;
    }

    struct ggml_init_params params;
    params.mem_size = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
    if (sd_version_is_sd3(sd_ctx->sd->version)) {
        params.mem_size *= 2;
    }
    if (sd_version_is_flux(sd_ctx->sd->version)) {
        params.mem_size *= 3;
    }
    if (sd_ctx->sd->stacked_id) {
        params.mem_size += static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
    }
    params.mem_size += width * height * 3 * sizeof(float) * 3;
    params.mem_size *= batch_count;
    params.mem_buffer = NULL;
    params.no_alloc   = false;
    // LOG_DEBUG("mem_size %u ", params.mem_size);

    struct ggml_context* work_ctx = ggml_init(params);
    if (!work_ctx) {
        LOG_ERROR("ggml_init() failed");
        return NULL;
    }

    size_t t0 = ggml_time_ms();

    if (seed < 0) {
        srand((int)time(NULL));
        seed = rand();
    }
    sd_ctx->sd->rng->manual_seed(seed);

    ggml_tensor* init_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);
    ggml_tensor* mask_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 1, 1);

    sd_mask_to_tensor(mask.data, mask_img);

    sd_image_to_tensor(init_image.data, init_img);

    ggml_tensor* masked_image;

    if (sd_version_is_inpaint(sd_ctx->sd->version)) {
        int64_t mask_channels = 1;
        if (sd_ctx->sd->version == VERSION_FLUX_FILL) {
            mask_channels = 8 * 8;  // flatten the whole mask
        }
        ggml_tensor* masked_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);
        sd_apply_mask(init_img, mask_img, masked_img);
        ggml_tensor* masked_image_0 = NULL;
        if (!sd_ctx->sd->use_tiny_autoencoder) {
            ggml_tensor* moments = sd_ctx->sd->encode_first_stage(work_ctx, masked_img);
            masked_image_0       = sd_ctx->sd->get_first_stage_encoding(work_ctx, moments);
        } else {
            masked_image_0 = sd_ctx->sd->encode_first_stage(work_ctx, masked_img);
        }
        masked_image = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, masked_image_0->ne[0], masked_image_0->ne[1], mask_channels + masked_image_0->ne[2], 1);
        for (int ix = 0; ix < masked_image_0->ne[0]; ix++) {
            for (int iy = 0; iy < masked_image_0->ne[1]; iy++) {
                int mx = ix * 8;
                int my = iy * 8;
                if (sd_ctx->sd->version == VERSION_FLUX_FILL) {
                    for (int k = 0; k < masked_image_0->ne[2]; k++) {
                        float v = ggml_tensor_get_f32(masked_image_0, ix, iy, k);
                        ggml_tensor_set_f32(masked_image, v, ix, iy, k);
                    }
                    // "Encode" 8x8 mask chunks into a flattened 1x64 vector, and concatenate to masked image
                    for (int x = 0; x < 8; x++) {
                        for (int y = 0; y < 8; y++) {
                            float m = ggml_tensor_get_f32(mask_img, mx + x, my + y);
                            // TODO: check if the way the mask is flattened is correct (is it supposed to be x*8+y or x+8*y?)
                            // python code was using "b (h 8) (w 8) -> b (8 8) h w"
                            ggml_tensor_set_f32(masked_image, m, ix, iy, masked_image_0->ne[2] + x * 8 + y);
                        }
                    }
                } else {
                    float m = ggml_tensor_get_f32(mask_img, mx, my);
                    ggml_tensor_set_f32(masked_image, m, ix, iy, 0);
                    for (int k = 0; k < masked_image_0->ne[2]; k++) {
                        float v = ggml_tensor_get_f32(masked_image_0, ix, iy, k);
                        ggml_tensor_set_f32(masked_image, v, ix, iy, k + mask_channels);
                    }
                }
            }
        }
    } else {
        // LOG_WARN("Inpainting with a base model is not great");
        masked_image = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width / 8, height / 8, 1, 1);
        for (int ix = 0; ix < masked_image->ne[0]; ix++) {
            for (int iy = 0; iy < masked_image->ne[1]; iy++) {
                int mx  = ix * 8;
                int my  = iy * 8;
                float m = ggml_tensor_get_f32(mask_img, mx, my);
                ggml_tensor_set_f32(masked_image, m, ix, iy);
            }
        }
    }

    ggml_tensor* init_latent = NULL;
    if (!sd_ctx->sd->use_tiny_autoencoder) {
        ggml_tensor* moments = sd_ctx->sd->encode_first_stage(work_ctx, init_img);
        init_latent          = sd_ctx->sd->get_first_stage_encoding(work_ctx, moments);
    } else {
        init_latent = sd_ctx->sd->encode_first_stage(work_ctx, init_img);
    }

    print_ggml_tensor(init_latent, true);
    size_t t1 = ggml_time_ms();
    LOG_INFO("encode_first_stage completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);

    std::vector<float> sigmas = sd_ctx->sd->denoiser->get_sigmas(sample_steps);
    size_t t_enc              = static_cast<size_t>(sample_steps * strength);
    if (t_enc == sample_steps)
        t_enc--;
    LOG_INFO("target t_enc is %zu steps", t_enc);
    std::vector<float> sigma_sched;
    sigma_sched.assign(sigmas.begin() + sample_steps - t_enc - 1, sigmas.end());

    sd_image_t* result_images = generate_image(sd_ctx,
                                               work_ctx,
                                               init_latent,
                                               prompt_c_str,
                                               negative_prompt_c_str,
                                               clip_skip,
                                               cfg_scale,
                                               guidance,
                                               eta,
                                               width,
                                               height,
                                               sample_method,
                                               sigma_sched,
                                               seed,
                                               batch_count,
                                               control_cond,
                                               control_strength,
                                               style_ratio,
                                               normalize_input,
                                               input_id_images_path_c_str,
                                               skip_layers_vec,
                                               slg_scale,
                                               skip_layer_start,
                                               skip_layer_end,
                                               masked_image);

    size_t t2 = ggml_time_ms();

    LOG_INFO("img2img completed in %.2fs", (t2 - t0) * 1.0f / 1000);

    return result_images;
}

SD_API sd_image_t* img2vid(sd_ctx_t* sd_ctx,
                           sd_image_t init_image,
                           int width,
                           int height,
                           int video_frames,
                           int motion_bucket_id,
                           int fps,
                           float augmentation_level,
                           float min_cfg,
                           float cfg_scale,
                           enum sample_method_t sample_method,
                           int sample_steps,
                           float strength,
                           int64_t seed) {
    if (sd_ctx == NULL) {
        return NULL;
    }

    LOG_INFO("img2vid %dx%d", width, height);

    std::vector<float> sigmas = sd_ctx->sd->denoiser->get_sigmas(sample_steps);

    struct ggml_init_params params;
    params.mem_size = static_cast<size_t>(10 * 1024) * 1024;  // 10 MB
    params.mem_size += width * height * 3 * sizeof(float) * video_frames;
    params.mem_buffer = NULL;
    params.no_alloc   = false;
    // LOG_DEBUG("mem_size %u ", params.mem_size);

    // draft context
    struct ggml_context* work_ctx = ggml_init(params);
    if (!work_ctx) {
        LOG_ERROR("ggml_init() failed");
        return NULL;
    }

    if (seed < 0) {
        seed = (int)time(NULL);
    }

    sd_ctx->sd->rng->manual_seed(seed);

    int64_t t0 = ggml_time_ms();

    SDCondition cond = sd_ctx->sd->get_svd_condition(work_ctx,
                                                     init_image,
                                                     width,
                                                     height,
                                                     fps,
                                                     motion_bucket_id,
                                                     augmentation_level);

    auto uc_crossattn = ggml_dup_tensor(work_ctx, cond.c_crossattn);
    ggml_set_f32(uc_crossattn, 0.f);

    auto uc_concat = ggml_dup_tensor(work_ctx, cond.c_concat);
    ggml_set_f32(uc_concat, 0.f);

    auto uc_vector = ggml_dup_tensor(work_ctx, cond.c_vector);

    SDCondition uncond = SDCondition(uc_crossattn, uc_vector, uc_concat);

    int64_t t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %" PRId64 " ms", t1 - t0);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->clip_vision->free_params_buffer();
    }

    sd_ctx->sd->rng->manual_seed(seed);
    int C                   = 4;
    int W                   = width / 8;
    int H                   = height / 8;
    struct ggml_tensor* x_t = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, video_frames);
    ggml_set_f32(x_t, 0.f);

    struct ggml_tensor* noise = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, video_frames);
    ggml_tensor_set_f32_randn(noise, sd_ctx->sd->rng);

    LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);
    struct ggml_tensor* x_0 = sd_ctx->sd->sample(work_ctx,
                                                 x_t,
                                                 noise,
                                                 cond,
                                                 uncond,
                                                 {},
                                                 0.f,
                                                 min_cfg,
                                                 cfg_scale,
                                                 0.f,
                                                 0.f,
                                                 sample_method,
                                                 sigmas,
                                                 -1,
                                                 SDCondition(NULL, NULL, NULL));

    int64_t t2 = ggml_time_ms();
    LOG_INFO("sampling completed, taking %.2fs", (t2 - t1) * 1.0f / 1000);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }

    struct ggml_tensor* img = sd_ctx->sd->decode_first_stage(work_ctx, x_0);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }
    if (img == NULL) {
        ggml_free(work_ctx);
        return NULL;
    }

    sd_image_t* result_images = (sd_image_t*)calloc(video_frames, sizeof(sd_image_t));
    if (result_images == NULL) {
        ggml_free(work_ctx);
        return NULL;
    }

    for (size_t i = 0; i < video_frames; i++) {
        auto img_i = ggml_view_3d(work_ctx, img, img->ne[0], img->ne[1], img->ne[2], img->nb[1], img->nb[2], img->nb[3] * i);

        result_images[i].width   = width;
        result_images[i].height  = height;
        result_images[i].channel = 3;
        result_images[i].data    = sd_tensor_to_image(img_i);
    }
    ggml_free(work_ctx);

    int64_t t3 = ggml_time_ms();

    LOG_INFO("img2vid completed in %.2fs", (t3 - t0) * 1.0f / 1000);

    return result_images;
}

struct API_ModelComponents {
    AutoEncoderKL* first_stage_model;
    Conditioner* cond_stage_model;
};

//images to tensors
SD_API bool convert_to_tensors(uint8_t** images, int width, int height, int _count) {
    ggml_context* ctx = Shared->getRawContext();
    int cc = Shared->getContextkey();
    if (!ctx || cc == EMPTY_INDEX) {
        return false; 
    }

    size_t count  = static_cast<size_t>(_count);
    for (size_t i = 0; i < count; i++) {
        ggml_tensor* img = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, height, 3, 1);
        sd_image_to_tensor(images[i], img); 
        Shared->setTensor(img, TensorType::IMAGE);
    }
    
    return true;
}

//count = batch count
//why pass key if its in-sync/tracked
SD_API bool vae_process(sd_ctx_t* sd_ctx, int key, bool decode) {
    if (!sd_ctx || !sd_ctx->sd) {
        LOG_ERROR("Invalid SD context.");
        return false;
    }

    auto& sd = sd_ctx->sd;
    if (!sd->first_stage_model) {
        LOG_ERROR("VAE is not initialized.");
        return false;
    }

    ggml_context* ctx = Shared->getRawContext();
    if (!ctx) {
        return false; 
    }

    std::vector<ggml_tensor*> tensors = std::vector<ggml_tensor*>();
    if (decode) {
        tensors = Shared->getTensors(key, TensorType::LATENT);
    } else {
        tensors = Shared->getTensors(key, TensorType::IMAGE);
    }

    if (tensors.empty()) {
        LOG_ERROR("Couldnt retrieve tensors from ID '%d' for decoding.", key);
        return false;
    }

    int count = (int)tensors.size();
    LOG_INFO("Decoding %d latent(s).", count);

    //TODO: Check if the model is a TAE model
    for (int i = 0; i < count; i++) {
        struct ggml_tensor* tensor = NULL;

        if (decode) {
            tensor = sd->decode_first_stage(ctx, tensors[i]);
            Shared->setTensor(tensor, TensorType::IMAGE);
        } else {
            tensor = sd->encode_first_stage(ctx, tensors[i]);
            Shared->setTensor(tensor, TensorType::LATENT);
        }

    }

    return true;
}

SD_API API_ModelComponents* get_model_components(sd_ctx_t* sd_ctx) {
    if (!sd_ctx || !sd_ctx->sd) {
        LOG_ERROR("Invalid Context or SD class.");
        return nullptr;
    }

    auto& sd = sd_ctx->sd;

    // Retrieve components from SharedData
    auto first_stage_model = sd->getModelComponent<AutoEncoderKL>("first_stage_model");
    auto cond_stage_model = sd->getModelComponent<Conditioner>("cond_stage_model");

    if (!first_stage_model || !cond_stage_model) {
        LOG_ERROR("Model components are not fully loaded.");
        return nullptr;
    }

    // Allocate and populate the struct
    API_ModelComponents* components = new API_ModelComponents();
    components->first_stage_model = first_stage_model.get();
    components->cond_stage_model = cond_stage_model.get();

    LOG_DEBUG("First Stage Model: %p", static_cast<void*>(components->first_stage_model));
    LOG_DEBUG("Conditioner Stage Model: %p", static_cast<void*>(components->cond_stage_model));
    return components;
}

SD_API void free_model_components(API_ModelComponents* components) {
    if (components) {
        delete components;
    }
}