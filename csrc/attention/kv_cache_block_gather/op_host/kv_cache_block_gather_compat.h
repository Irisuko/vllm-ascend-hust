#pragma once

#include <cstdio>

// Compatibility fallbacks for CANN/open-project build environments that do
// not provide the usual operator logging/checking macros.  This file does not
// define any operator semantics; it only keeps the host sources portable.
#ifndef OP_LOGE
#define OP_LOGE(opname, fmt, ...)                         \
    do {                                                  \
        (void)(opname);                                   \
        std::printf("[ERROR] " fmt "\n", ##__VA_ARGS__); \
    } while (0)
#endif

#ifndef OP_LOGD
#define OP_LOGD(opname, fmt, ...) \
    do {                          \
        (void)(opname);           \
    } while (0)
#endif

#ifndef OP_CHECK_IF
#define OP_CHECK_IF(cond, log_func, expr) \
    do {                                  \
        if (cond) {                       \
            log_func;                     \
            expr;                         \
        }                                 \
    } while (0)
#endif

#ifndef OP_CHECK_NULL_WITH_CONTEXT
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)                  \
    do {                                                          \
        if ((ptr) == nullptr) {                                   \
            OP_LOGE(context, "%s is null", #ptr);                 \
            return ge::GRAPH_FAILED;                              \
        }                                                         \
    } while (0)
#endif
