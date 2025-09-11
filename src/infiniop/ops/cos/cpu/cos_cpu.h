#ifndef __COS_CPU_H__
#define __COS_CPU_H__

#include <cmath>
#include <type_traits>

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../../utils/custom_types.h"

ELEMENTWISE_DESCRIPTOR(cos, cpu)

namespace op::cos::cpu {
typedef struct CosOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return std::cos(x);
    }
    
    bf16_t operator()(const bf16_t &x) const {
        double x_double = static_cast<double>(_bf16_to_f32(x));
        double result = std::cos(x_double);
        return utils::cast<bf16_t>(result);
    }
} CosOp;
} // namespace op::cos::cpu

#endif // __COS_CPU_H__