{
    depfiles = "operator.o: src/infiniop/ops/swiglu/operator.cc  src/infiniop/ops/swiglu/../../operator.h  include/infiniop/operator_descriptor.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/tensor_descriptor.h  src/infiniop/ops/swiglu/../../handle.h include/infiniop/handle.h  include/infiniop/ops/swiglu.h  include/infiniop/ops/../operator_descriptor.h  src/infiniop/ops/swiglu/cpu/swiglu_cpu.h  src/infiniop/ops/swiglu/cpu/../../../elementwise/cpu/elementwise_cpu.h  src/infiniop/ops/swiglu/cpu/../../../elementwise/cpu/../../devices/cpu/common_cpu.h  src/infiniop/ops/swiglu/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils.h  src/infiniop/ops/swiglu/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils/custom_types.h  src/infiniop/ops/swiglu/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils/rearrange.h  src/infiniop/ops/swiglu/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils/result.hpp  src/infiniop/ops/swiglu/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils/check.h  include/infinicore.h  src/infiniop/ops/swiglu/cpu/../../../elementwise/cpu/../../devices/cpu/cpu_handle.h  src/infiniop/ops/swiglu/cpu/../../../elementwise/cpu/../../devices/cpu/../../handle.h  src/infiniop/ops/swiglu/cpu/../../../elementwise/cpu/../elementwise.h  src/infiniop/ops/swiglu/cpu/../../../elementwise/cpu/../../../utils.h  src/infiniop/ops/swiglu/cpu/../../../elementwise/cpu/../../operator.h  src/infiniop/ops/swiglu/cpu/../../../elementwise/cpu/../../tensor.h  include/infiniop/tensor_descriptor.h  src/infiniop/ops/swiglu/cpu/../../../elementwise/cpu/../../../utils.h\
",
    depfiles_format = "gcc",
    files = {
        "src/infiniop/ops/swiglu/operator.cc"
    },
    values = {
        "/home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/gcc-11.3.0-7tpmmhoar763gi2qhigyczd2vqqhpgxk/bin/g++",
        {
            "-m64",
            "-fPIC",
            "-g",
            "-O0",
            "-std=c++17",
            "-Iinclude",
            "-DDEBUG_MODE",
            "-DENABLE_CPU_API",
            "-DENABLE_OMP",
            "-DENABLE_CUDNN_API",
            "-finput-charset=UTF-8",
            "-fexec-charset=UTF-8"
        }
    }
}