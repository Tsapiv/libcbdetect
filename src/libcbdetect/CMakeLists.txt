include_directories(${PROJECT_SOURCE_DIR}/include/libcbdetect)
include_directories(${PROJECT_SOURCE_DIR}/include/)

aux_source_directory(. SOURCE_LIST)
add_library(cbdetect ${LIB_TYPE} ${SOURCE_LIST})
target_link_libraries(cbdetect ${OpenCV_LIBS})
