{
    depfiles = "operator.o: src/infiniop/ops/rearrange/operator.cc  src/infiniop/ops/rearrange/../../operator.h  include/infiniop/operator_descriptor.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/tensor_descriptor.h  src/infiniop/ops/rearrange/../../handle.h include/infiniop/handle.h  include/infiniop/ops/rearrange.h  include/infiniop/ops/../operator_descriptor.h  src/infiniop/ops/rearrange/cpu/rearrange_cpu.h  src/infiniop/ops/rearrange/cpu/../rearrange.h  src/infiniop/ops/rearrange/cpu/../../../../utils.h  src/infiniop/ops/rearrange/cpu/../../../../utils/custom_types.h  src/infiniop/ops/rearrange/cpu/../../../../utils/rearrange.h  src/infiniop/ops/rearrange/cpu/../../../../utils/result.hpp  src/infiniop/ops/rearrange/cpu/../../../../utils/check.h  include/infinicore.h src/infiniop/ops/rearrange/cpu/../../../operator.h  src/infiniop/ops/rearrange/nvidia/rearrange_nvidia.cuh  src/infiniop/ops/rearrange/nvidia/../rearrange.h\
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
        "src/infiniop/ops/rearrange/operator.cc"
    }
}