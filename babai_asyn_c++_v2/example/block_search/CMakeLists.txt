add_executable(block_search ils_block_search.cpp)
message(${CMAKE_SOURCE_DIR}/cmake-build-debug/lib/libsils.a)
target_link_libraries(block_search ${CMAKE_SOURCE_DIR}/cmake-build-debug/lib/libsils.a)
set_property(TARGET block_search PROPERTY POSITION_INDEPENDENT_CODE ON)