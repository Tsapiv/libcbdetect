find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

cmake_policy(SET CMP0095 NEW)

include_directories(${PROJECT_SOURCE_DIR}/include/)


pybind11_add_module(pycbdetect
        pycbdetect.cpp
        WITH_SOABI)

target_link_libraries(pycbdetect PRIVATE ${PROJECT_NAME})
target_link_libraries(pycbdetect PRIVATE pybind11::embed)


target_compile_definitions(pycbdetect PRIVATE VERSION_INFO=${SKBUILD_PROJECT_VERSION})
target_compile_definitions(pycbdetect PRIVATE USE_CAPSULE="1")

add_custom_command(TARGET pycbdetect POST_BUILD
                     COMMAND
                     PYTHONPATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:$ENV{PYTHONPATH} pybind11-stubgen -o ${CMAKE_BINARY_DIR}/stubs/pycbdetect pycbdetect
                     BYPRODUCTS ${CMAKE_BINARY_DIR}/stubs/pycbdetect
                     USES_TERMINAL
                     COMMENT "Generating type stubs")


# set_target_properties(pycbdetect PROPERTIES INSTALL_RPATH "$ORIGIN:$ORIGIN/lib")

install(TARGETS pycbdetect DESTINATION .)

install(DIRECTORY ${CMAKE_BINARY_DIR}/stubs/pycbdetect/ DESTINATION .)