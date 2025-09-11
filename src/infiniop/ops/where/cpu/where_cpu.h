#ifndef __WHERE_CPU_H__
#define __WHERE_CPU_H__

#include <type_traits>

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../../utils/custom_types.h"

ELEMENTWISE_DESCRIPTOR(where, cpu)

namespace op::where::cpu {
typedef struct WhereOp {
public:
    static constexpr size_t num_inputs = 3;

    // An operator for heterogeneous input types, used to handle cases where condition (bool) and a, b (float, etc.) are of different types
    // Note: According to the elementwise framework
    // the parameter order should be consistent with the order of the inputs vector: inputs[0] = condition, inputs[1] = a, inputs[2] = b
    template <typename Tout, typename Tcond, typename Ta, typename Tb>
    Tout operator()(const Tcond &condition, const Ta &a, const Tb &b) const {
        bool cond_bool;
        if constexpr (std::is_same_v<Tcond, bool>) {
            cond_bool = condition;
        } else {
            // Suppose that the int8 type represents bool
            cond_bool = (condition != 0);
        }
        
        return cond_bool ? static_cast<Tout>(a) : static_cast<Tout>(b);
    }
} WhereOp;
} // namespace op::where::cpu

#endif // __WHERE_CPU_H__