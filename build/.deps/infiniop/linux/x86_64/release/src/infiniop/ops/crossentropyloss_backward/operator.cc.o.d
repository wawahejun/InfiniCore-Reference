{
    depfiles = "operator.o: src/infiniop/ops/crossentropyloss_backward/operator.cc  src/infiniop/ops/crossentropyloss_backward/../../operator.h  include/infiniop/operator_descriptor.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/tensor_descriptor.h  src/infiniop/ops/crossentropyloss_backward/../../handle.h  include/infiniop/handle.h  include/infiniop/ops/crossentropyloss_backward.h  include/infiniop/ops/../operator_descriptor.h  src/infiniop/ops/crossentropyloss_backward/cpu/crossentropyloss_backward_cpu.h  src/infiniop/ops/crossentropyloss_backward/cpu/../../../elementwise/cpu/elementwise_cpu.h  src/infiniop/ops/crossentropyloss_backward/cpu/../../../elementwise/cpu/../../devices/cpu/common_cpu.h  src/infiniop/ops/crossentropyloss_backward/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils.h  src/infiniop/ops/crossentropyloss_backward/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils/custom_types.h  src/infiniop/ops/crossentropyloss_backward/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils/rearrange.h  src/infiniop/ops/crossentropyloss_backward/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils/result.hpp  src/infiniop/ops/crossentropyloss_backward/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils/check.h  include/infinicore.h  src/infiniop/ops/crossentropyloss_backward/cpu/../../../elementwise/cpu/../../devices/cpu/cpu_handle.h  src/infiniop/ops/crossentropyloss_backward/cpu/../../../elementwise/cpu/../../devices/cpu/../../handle.h  src/infiniop/ops/crossentropyloss_backward/cpu/../../../elementwise/cpu/../elementwise.h  src/infiniop/ops/crossentropyloss_backward/cpu/../../../elementwise/cpu/../../../utils.h  src/infiniop/ops/crossentropyloss_backward/cpu/../../../elementwise/cpu/../../operator.h  src/infiniop/ops/crossentropyloss_backward/cpu/../../../elementwise/cpu/../../tensor.h  include/infiniop/tensor_descriptor.h  src/infiniop/ops/crossentropyloss_backward/cpu/../../../elementwise/cpu/../../../utils.h  src/infiniop/ops/crossentropyloss_backward/cpu/../../../../utils/custom_types.h  src/infiniop/ops/crossentropyloss_backward/nvidia/crossentropyloss_backward_nvidia.cuh  src/infiniop/ops/crossentropyloss_backward/nvidia/../../../elementwise/nvidia/elementwise_nvidia_api.cuh  src/infiniop/ops/crossentropyloss_backward/nvidia/../../../elementwise/nvidia/../elementwise.h\
",
    depfiles_format = "gcc",
    values = {
        "/home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/gcc-11.3.0-7tpmmhoar763gi2qhigyczd2vqqhpgxk/bin/g++",
        {
            "-m64",
            "-fPIC",
            "-O3",
            "-std=c++17",
            "-Iinclude",
            "-DENABLE_CPU_API",
            "-DENABLE_OMP",
            "-DENABLE_NVIDIA_API",
            "-DENABLE_CUDNN_API",
            "-finput-charset=UTF-8",
            "-fexec-charset=UTF-8",
            "-DNDEBUG"
        }
    },
    files = {
        "src/infiniop/ops/crossentropyloss_backward/operator.cc"
    }
}