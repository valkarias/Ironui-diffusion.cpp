#ifndef SHARED_HPP
#define SHARED_HPP

#include "shared.h"

class ContextsManager {
    // e.g: contexts = {

        //[samplerXYZ] = {
            //entry = {
                //context,
                //tensors = {
                    //[TensorType::LATENT] = [....],
                    //[TensorType::IMAGE] = [....],
                //}
            //}
        //},

        //...
    //}

    std::unordered_map<int, ScopedContext> _contexts;

    public:
    // Constructor
    ContextsManager() = default;

    // Modified destructor
    ~ContextsManager() {
        free_all();
    }

    ScopedContext* create_context(int key, ggml_init_params params) {
        auto& e = _contexts[key];
        if (e.ctx) 
            ggml_free(e.ctx);
        e.params = params;
        e.ctx    = ggml_init(params);

        LOG_DEBUG("Context memory size: %zu MB", params.mem_size / (1024 * 1024));
        return &e;
    }

    ScopedContext* get_context(int key) {
        const auto& it = _contexts.find(key);
        if (it != _contexts.end())
            return &(it->second);

        return NULL;
    }

    void free_context(int key, TensorType type = TensorType::_ALL_) {
        //it -> {key, ScopedContext}
        auto it = _contexts.find(key);
        if (it != _contexts.end()) {
            
            //specific type (DO not free context)
            //since tensors do not get freed, we can maybe reuse them
            if (type != TensorType::_ALL_) {
                auto type_it = it->second.tensors.find(type);
                if (type_it != it->second.tensors.end()) {
                    type_it->second.clear();
                }
                //keep empty section
                // contexts.erase(type_it);
                
            } else {
                // Free all tensor types
                for (auto& [tensor_type, tensor_list] : it->second.tensors) {
                    tensor_list.clear();
                }

                //free ggml context, clearing all tensors
                ggml_free(it->second.ctx);
                it->second.ctx = NULL;
            }

        }
    }

    void free_all() {
        for (auto& ctx : _contexts)
            free_context(ctx.first);
        _contexts.clear();
    }
};


class SharedData {

private:
    //TODO: separate cached context for empty latents.
    ContextsManager* _contexts = new ContextsManager();
    int _currentContextKey = EMPTY_INDEX;
    ScopedContext* _currentContext = NULL;


    static SharedData* _instance;

    void _deleteShared() {
        LOG_DEBUG("Deleting shared data.");
        delete _contexts;
        _currentContext = NULL;
    }

    ggml_init_params _initContextParameters(int width, int height, int batch_count) {
        //support fluxs and sd3
        //multiple empty latents will require tweaking the params.

        ggml_init_params contextParams = {0};

        //determine size dynamically using reallocation.
        contextParams.mem_size = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
        contextParams.mem_buffer = NULL;
        contextParams.no_alloc   = false;

        contextParams.mem_size += width * height * 3 * sizeof(float);
        contextParams.mem_size *= batch_count;

        return contextParams;
    }
    
public:
    // Constructor
    SharedData() = default;

    SharedData(const SharedData&) = delete;
    SharedData& operator=(const SharedData&) = delete;

    static SharedData& instance() {
        if (_instance == nullptr) {
            LOG_DEBUG("Created new shared instance");
            _instance = new SharedData();
        }
        return *_instance;
    }
    
    static void destroyInstance() {
        if (_instance != nullptr) {
            _instance->_deleteShared();
            delete _instance;
            _instance = nullptr;

            LOG_DEBUG("Deleted shared instance");
        }
    }


    static void cleanInstance() {
        _instance->_currentContextKey = EMPTY_INDEX;
        _instance->_currentContext = NULL;
    }

    static size_t getTensorSize(ggml_tensor* tensor) {
        size_t dataSize = 1;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            dataSize *= tensor->ne[i];
        }
        return dataSize * ggml_element_size(tensor);
    }

    //

    void setContextKey(int key) {
        if (_currentContextKey == key) return;

        _currentContextKey = key;
        _currentContext = _contexts->get_context(key);
        if (!_currentContext)
            LOG_WARN("ID %d does not have an initalized context.", key);

    }

    int getContextkey() {
        return _currentContextKey;
    }

    ScopedContext* getCurrentContext() {
        if (!_currentContext) {
            LOG_WARN("'%d' context not set.", _currentContextKey);
            return NULL;
        }
        return _currentContext;
    }

    void createContext(int width, int height, int batch_count) {
        if (_currentContext != NULL)
            LOG_WARN("Overwriting current context.");

        LOG_DEBUG("Creating new context for ID %d", _currentContextKey);
        _currentContext = _contexts->create_context(_currentContextKey, _initContextParameters(width, height, batch_count));
    }

    void setRawContext(ggml_context* _ctx) {
        if (_currentContext->ctx == NULL)
            _currentContext->ctx = _ctx;
        else
            LOG_WARN("Current raw context already set.");
    }

    ggml_context* getRawContext() {
        if (!_currentContext) {
            LOG_WARN("ID %d context is not initialized.", _currentContextKey);
            return NULL;
        }

        if (_currentContext->ctx == NULL)
            LOG_WARN("ID %d raw context is null.", _currentContextKey);
            
        return _currentContext->ctx;
    }

    //cache 1 re-usable empty tensor
    //cache empty tensors in a separate global 'scoped' context.
    int createEmptyTensor(int width, int height, int C, int batch_count) {
        //since we execute dependencies before the root.
        //initial latents are optional, but we still ensure context creation right before inference anyways.
        if (!_currentContext)
            createContext(width, height, batch_count);

        ggml_tensor* init_latent = nullptr;
        _create_empty_latent_tensor(_currentContext->ctx, width, height, C, &init_latent);
        if (init_latent == nullptr || init_latent->data == nullptr) {
            return EMPTY_INDEX;
        }

        return setTensor(init_latent, TensorType::EMPTY);
    }
    
    int setTensor(ggml_tensor* tensor, TensorType type) {        
        LOG_DEBUG("Set tensor in context %d (%zu)", _currentContextKey, getTensorSize(tensor));
        _currentContext->tensors[type].push_back(tensor);

        //index
        return _currentContext->tensors[type].size()-1;
    }

    std::vector<ggml_tensor*> getTensors(int key, TensorType type) {
        const auto _empty = std::vector<ggml_tensor*>();

        const auto& ctx = _contexts->get_context(key);
        if (ctx == NULL)
            return _empty;

        const auto& tensors = ctx->tensors[type];
        if (!tensors.empty())
            return tensors;

        LOG_WARN("Retrieved empty tensor array of type %d from ID %d", (int)type, key);
        return _empty;
    }

    void cleanTensors(int key, TensorType type) {
        _contexts->free_context(key, type);
    }
};

struct shared_data_t {
    SharedData* operator->() { return &SharedData::instance(); }
    SharedData& operator*()  { return SharedData::instance(); }
};

static shared_data_t Shared;

#endif // SHARED_HPP