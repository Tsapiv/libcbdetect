cmake_minimum_required(VERSION 3.15)
project(cbdetect LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Define output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/$<CONFIG>)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib/$<CONFIG>)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib/$<CONFIG>)

# Compiler-specific settings
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    add_compile_definitions(UNIX)
elseif (MSVC)
    add_compile_definitions(WINDOWS)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS) # Suppress MSVC warnings
    add_compile_options(/openmp /EHsc)
endif()

# Option for building shared or static library
option(SHARED_LIB "Build shared libraries" OFF)
option(PYTHON_BINDINGS "Build python bindings" OFF)


if (SHARED_LIB)
    set(LIB_TYPE SHARED)
    if (WIN32)
        add_compile_definitions(IS_A_DLL)
    endif()
else()
    set(LIB_TYPE STATIC)
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add subdirectories
add_subdirectory(src/libcbdetect)
add_subdirectory(src/example)
if(PYTHON_BINDINGS)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
    add_subdirectory(external/cvnp)
    add_subdirectory(src/bindings)
endif()



