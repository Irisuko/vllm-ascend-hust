set(_DISPATCH_FFN_INC_OPTS)
if (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include)
elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/arm64-linux/ascendc/include)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/arm64-linux/ascendc/include)
elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include)
endif()
if (EXISTS ${CMAKE_SOURCE_DIR}/third_party/catlass/include)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${CMAKE_SOURCE_DIR}/third_party/catlass/include)
endif()

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        dispatch_ffn_combine_bf16_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME DispatchFFNCombineBF16
    OPTIONS
        --cce-auto-sync=on
        -Wno-deprecated-declarations
        -Werror
        -DHCCL_COMM
        -DCATLASS_ARCH=2201
        ${_DISPATCH_FFN_INC_OPTS}
)

# Ascend 910B/910B2 and 910_93 share the V220 architecture macros, but use
# different full HCCL context layouts. Keep the existing 910_93 window path
# unchanged and select the A2 context layout explicitly for 910B builds.
add_ops_compile_options(
    OP_NAME DispatchFFNCombineBF16
    COMPUTE_UNIT Ascend910B1
    OPTIONS -DDISPATCH_FFN_COMBINE_BF16_A2=1
)
add_ops_compile_options(
    OP_NAME DispatchFFNCombineBF16
    COMPUTE_UNIT Ascend910_9391
    OPTIONS -DDISPATCH_FFN_COMBINE_BF16_A2=0
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE dispatch_ffn_combine_bf16 ACLNNTYPE aclnn_inner)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
    )
endif()
