#ifndef __DIV_CPU_H__
#define __DIV_CPU_H__

#include <cmath>
#include <type_traits>
#include <limits>

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../../utils/custom_types.h"

ELEMENTWISE_DESCRIPTOR(div, cpu)

namespace op::div::cpu {
typedef struct DivOp {
public:
    static constexpr size_t num_inputs = 2;

    template <typename T>
    T operator()(const T &a, const T &b) const {
        if (b == static_cast<T>(0)) {
            if (a > static_cast<T>(0)) {
                return std::numeric_limits<T>::infinity();
            } else if (a < static_cast<T>(0)) {
                return -std::numeric_limits<T>::infinity();
            } else {
                return std::numeric_limits<T>::quiet_NaN();
            }
        }
        return a / b;
    }
    
    bf16_t operator()(const bf16_t &a, const bf16_t &b) const {
        double a_double = static_cast<double>(_bf16_to_f32(a));
        double b_double = static_cast<double>(_bf16_to_f32(b));
        
        if (b_double == 0.0) {
            if (a_double > 0.0) {
                return utils::cast<bf16_t>(std::numeric_limits<double>::infinity());
            } else if (a_double < 0.0) {
                return utils::cast<bf16_t>(-std::numeric_limits<double>::infinity());
            } else {
                return utils::cast<bf16_t>(std::numeric_limits<double>::quiet_NaN());
            }
        }
        
        double result = a_double / b_double;
        return utils::cast<bf16_t>(result);
    }
} DivOp;
} // namespace op::div::cpu

#endif // __DIV_CPU_H__