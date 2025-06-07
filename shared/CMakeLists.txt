# Create a library from the shared sources
add_library(shared
    shared.cpp
    shared.hpp
    shared.h
)

target_include_directories(shared
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/thirdparty
)

target_link_libraries(shared
    PUBLIC
    ggml
)