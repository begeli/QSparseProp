set(
        QSPARSEPROP_SOURCES
        ${QSPARSEPROP_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/quantization_strategy.h
        ${CMAKE_CURRENT_SOURCE_DIR}/quantization8_strategy.h
        ${CMAKE_CURRENT_SOURCE_DIR}/stochastic_quantization.h
        ${CMAKE_CURRENT_SOURCE_DIR}/standard_quantization_strategy.h
        ${CMAKE_CURRENT_SOURCE_DIR}/standard_quantization8_strategy.h
        ${CMAKE_CURRENT_SOURCE_DIR}/sawb_quantization_strategy.h
        ${CMAKE_CURRENT_SOURCE_DIR}/sawb_quantization8_strategy.h
        ${CMAKE_CURRENT_SOURCE_DIR}/luq_strategy.h
        ${CMAKE_CURRENT_SOURCE_DIR}/luq8_strategy.h
        ${CMAKE_CURRENT_SOURCE_DIR}/dithered_quantization_strategy.h
        ${CMAKE_CURRENT_SOURCE_DIR}/dithered_quantization8_strategy.h
        ${CMAKE_CURRENT_SOURCE_DIR}/stochastic_quantization.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/luq8_strategy.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sawb_quantization8_strategy.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/standard_quantization8_strategy.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/dithered_quantization8_strategy.cpp
        PARENT_SCOPE
)