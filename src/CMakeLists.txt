cmake_minimum_required(VERSION 3.5)

project(MultilayerPerceptron VERSION 0.1 LANGUAGES CXX)

add_compile_options(-O3)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets Charts)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets Charts)

include_directories(
  ${PROJECT_SOURCE_DIR}/model
  ${PROJECT_SOURCE_DIR}/model/graph_mlp
  ${PROJECT_SOURCE_DIR}/model/matrix_mlp
  ${PROJECT_SOURCE_DIR}/model/utility
  ${PROJECT_SOURCE_DIR}/view
  ${PROJECT_SOURCE_DIR}/controller
)

set(HEADERS
  ${PROJECT_SOURCE_DIR}/model/abstract_mlp.h
  ${PROJECT_SOURCE_DIR}/model/config.h
  ${PROJECT_SOURCE_DIR}/model/image.h
  ${PROJECT_SOURCE_DIR}/model/metrics.h
  ${PROJECT_SOURCE_DIR}/model/mlp.h
  ${PROJECT_SOURCE_DIR}/model/graph_mlp/graph_mlp.h
  ${PROJECT_SOURCE_DIR}/model/graph_mlp/layer.h
  ${PROJECT_SOURCE_DIR}/model/graph_mlp/neuron.h
  ${PROJECT_SOURCE_DIR}/model/matrix_mlp/matrix_mlp.h
  ${PROJECT_SOURCE_DIR}/model/utility/activation_functions.h
  ${PROJECT_SOURCE_DIR}/model/utility/io.h
  ${PROJECT_SOURCE_DIR}/model/utility/matrix_operations.h
  ${PROJECT_SOURCE_DIR}/view/mainwindow.h
  ${PROJECT_SOURCE_DIR}/view/mainwindow.h
  ${PROJECT_SOURCE_DIR}/view/painter.h
  ${PROJECT_SOURCE_DIR}/view/graph.h
  ${PROJECT_SOURCE_DIR}/controller/controller.h
)

set(SOURCES
  ${PROJECT_SOURCE_DIR}/model/mlp.cc
  ${PROJECT_SOURCE_DIR}/model/graph_mlp/graph_mlp.cc
  ${PROJECT_SOURCE_DIR}/model/graph_mlp/layer.cc
  ${PROJECT_SOURCE_DIR}/model/graph_mlp/neuron.cc
  ${PROJECT_SOURCE_DIR}/model/matrix_mlp/matrix_mlp.cc
  ${PROJECT_SOURCE_DIR}/model/utility/io.cc
  ${PROJECT_SOURCE_DIR}/model/utility/matrix_operations.cc
  ${PROJECT_SOURCE_DIR}/view/main.cpp
  ${PROJECT_SOURCE_DIR}/view/mainwindow.cpp
  ${PROJECT_SOURCE_DIR}/view/painter.cpp
  ${PROJECT_SOURCE_DIR}/view/graph.cc
  ${PROJECT_SOURCE_DIR}/controller/controller.cc
)

set(UI
  ${PROJECT_SOURCE_DIR}/view/mainwindow.ui
)

set(PROJECT_SOURCES
    ${HEADERS}
    ${SOURCES}
    ${UI}       
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(MultilayerPerceptron
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET MultilayerPerceptron APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
# For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation
else()
    if(ANDROID)
        add_library(MultilayerPerceptron SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(MultilayerPerceptron
            ${PROJECT_SOURCES}
        )
    endif()
endif()

target_link_libraries(MultilayerPerceptron PRIVATE Qt${QT_VERSION_MAJOR}::Widgets Qt${QT_VERSION_MAJOR}::Charts)

# Qt for iOS sets MACOSX_BUNDLE_GUI_IDENTIFIER automatically since Qt 6.1.
# If you are developing for iOS or macOS you should consider setting an
# explicit, fixed bundle identifier manually though.
if(${QT_VERSION} VERSION_LESS 6.1.0)
  set(BUNDLE_ID_OPTION MACOSX_BUNDLE_GUI_IDENTIFIER com.example.MultilayerPerceptron)
endif()
set_target_properties(MultilayerPerceptron PROPERTIES
    ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

include(GNUInstallDirs)
install(TARGETS MultilayerPerceptron
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(MultilayerPerceptron)
endif()


find_program(CPPCHECK cppcheck)

if(CPPCHECK)
    message(STATUS "cppcheck found: ${CPPCHECK}")
    list(
        APPEND CPPCHECK_ARGS
        "--enable=all"
        "--inconclusive"
        "--quiet"
        "--language=c++"
        "--std=c++17"
        "--suppress=missingInclude"
        "--suppress=unusedFunction"
        "--suppress=useStlAlgorithm"
    )
    add_custom_target(
        cppcheck
        COMMAND ${CPPCHECK} ${CPPCHECK_ARGS} ${HEADERS} ${SOURCES}
    )
else()
    message(STATUS "cppcheck not found")
endif()
