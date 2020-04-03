set(LIB_CRCCOSY_SRC
        geometry/segment.cc
        geometry/curvilinear_coordinate_system.cc)

set(LIB_CRCCOSY_INCLUDE ".")

add_library(crccosy ${LIB_CRCCOSY_SRC})

target_compile_features(crccosy
        PUBLIC cxx_lambdas cxx_auto_type
        PRIVATE cxx_lambdas cxx_auto_type)
set_property(TARGET crccosy PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(crccosy PUBLIC ${LIB_CRCCOSY_INCLUDE})

add_executable(test_node main.cc)
target_compile_features(test_node
        PUBLIC cxx_lambdas cxx_auto_type
        PRIVATE cxx_lambdas cxx_auto_type)
set_property(TARGET test_node PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(test_node PUBLIC ${LIB_CRCCOSY_INCLUDE})
target_link_libraries(test_node PUBLIC crccosy)


