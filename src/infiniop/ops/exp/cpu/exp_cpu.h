#ifndef __EXP_CPU_H__
#define __EXP_CPU_H__

#include <cmath>
#include <type_traits>

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../../utils/custom_types.h"

ELEMENTWISE_DESCRIPTOR(exp, cpu)

namespace op::exp::cpu {
typedef struct ExpOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return std::exp(x);
    }
    
    bf16_t operator()(const bf16_t &x) const {
        double x_double = static_cast<double>(_bf16_to_f32(x));
        double result = std::exp(x_double);
        return utils::cast<bf16_t>(result);
    }
} ExpOp;
} // namespace op::exp::cpu

#endif // __EXP_CPU_H__