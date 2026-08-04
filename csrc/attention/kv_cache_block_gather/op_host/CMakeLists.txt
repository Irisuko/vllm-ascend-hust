# Register this operator with the custom-op build.  op_host contains the
# CPU-side schema/tiling code; add_modules_sources also connects it to the
# AscendC kernel sources under ../op_kernel.
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        kv_cache_block_gather_def.cpp
    )
endif()

# Produce a kernel binary that CANN can relocate/load at runtime.
add_ops_compile_options(
    OP_NAME KvCacheBlockGather
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            --op_relocatable_kernel_binary=true
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE kv_cache_block_gather ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
    )
endif()
