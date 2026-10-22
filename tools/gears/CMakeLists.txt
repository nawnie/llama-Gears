add_library(llama-gears-policy STATIC src/policy.cpp)
target_include_directories(llama-gears-policy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(llama-gears-policy PUBLIC cxx_std_17)

add_executable(llama-gears-inspect src/inspect.cpp)
target_include_directories(llama-gears-inspect PRIVATE ${CMAKE_SOURCE_DIR}/ggml/include)
target_link_libraries(llama-gears-inspect PRIVATE llama-gears-policy ggml)
target_compile_features(llama-gears-inspect PRIVATE cxx_std_17)

if (LLAMA_BUILD_TESTS)
    add_executable(llama-gears-policy-test tests/policy_test.cpp)
    target_link_libraries(llama-gears-policy-test PRIVATE llama-gears-policy)
    add_test(NAME llama-gears-policy COMMAND llama-gears-policy-test)
endif()
