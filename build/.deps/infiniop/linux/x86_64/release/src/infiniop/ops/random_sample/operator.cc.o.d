{
    depfiles = "operator.o: src/infiniop/ops/random_sample/operator.cc  src/infiniop/ops/random_sample/../../operator.h  include/infiniop/operator_descriptor.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/tensor_descriptor.h  src/infiniop/ops/random_sample/../../handle.h include/infiniop/handle.h  include/infiniop/ops/random_sample.h  include/infiniop/ops/../operator_descriptor.h  src/infiniop/ops/random_sample/cpu/random_sample_cpu.h  src/infiniop/ops/random_sample/cpu/../random_sample.h  src/infiniop/ops/random_sample/cpu/../../../operator.h  src/infiniop/ops/random_sample/cpu/../info.h  src/infiniop/ops/random_sample/cpu/../../../../utils.h  src/infiniop/ops/random_sample/cpu/../../../../utils/custom_types.h  src/infiniop/ops/random_sample/cpu/../../../../utils/rearrange.h  src/infiniop/ops/random_sample/cpu/../../../../utils/result.hpp  src/infiniop/ops/random_sample/cpu/../../../../utils/check.h  include/infinicore.h  src/infiniop/ops/random_sample/cpu/../../../tensor.h  include/infiniop/tensor_descriptor.h  src/infiniop/ops/random_sample/cpu/../../../../utils.h  src/infiniop/ops/random_sample/nvidia/random_sample_nvidia.cuh  src/infiniop/ops/random_sample/nvidia/../random_sample.h\
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
        "src/infiniop/ops/random_sample/operator.cc"
    }
}