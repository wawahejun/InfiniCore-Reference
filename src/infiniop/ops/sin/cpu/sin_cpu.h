#ifndef __SIN_CPU_H__
#define __SIN_CPU_H__

#include <cmath>
#include <type_traits>

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../../utils/custom_types.h"

ELEMENTWISE_DESCRIPTOR(sin, cpu)

namespace op::sin::cpu {
typedef struct SinOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return std::sin(x);
    }
    
    bf16_t operator()(const bf16_t &x) const {
        double x_double = static_cast<double>(_bf16_to_f32(x));
        double result = std::sin(x_double);
        return utils::cast<bf16_t>(result);
    }
} SinOp;
} // namespace op::sin::cpu

#endif // __SIN_CPU_H__