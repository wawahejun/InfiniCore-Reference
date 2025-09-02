{
    depfiles = "gather_cpu.o: src/infiniop/ops/gather/cpu/gather_cpu.cc  src/infiniop/ops/gather/cpu/gather_cpu.h  src/infiniop/ops/gather/cpu/../../../operator.h  include/infiniop/operator_descriptor.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/tensor_descriptor.h  src/infiniop/ops/gather/cpu/../../../devices/cpu/cpu_handle.h  src/infiniop/ops/gather/cpu/../../../devices/cpu/../../handle.h  include/infiniop/handle.h  src/infiniop/ops/gather/cpu/../../../../utils.h  src/infiniop/ops/gather/cpu/../../../../utils/custom_types.h  src/infiniop/ops/gather/cpu/../../../../utils/rearrange.h  src/infiniop/ops/gather/cpu/../../../../utils/result.hpp  src/infiniop/ops/gather/cpu/../../../../utils/check.h  include/infinicore.h src/infiniop/ops/gather/cpu/../../../tensor.h  include/infiniop/tensor_descriptor.h  src/infiniop/ops/gather/cpu/../../../../utils.h\
",
    depfiles_format = "gcc",
    values = {
        "/home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/gcc-11.3.0-7tpmmhoar763gi2qhigyczd2vqqhpgxk/bin/g++",
        {
            "-m64",
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
            "-Wall",
            "-Werror",
            "-O3",
            "-std=c++17",
            "-Iinclude",
            "-DENABLE_CPU_API",
            "-DENABLE_OMP",
            "-DENABLE_NVIDIA_API",
            "-DENABLE_CUDNN_API",
            "-finput-charset=UTF-8",
            "-fexec-charset=UTF-8",
            "-fPIC",
            "-Wno-unknown-pragmas",
            "-fopenmp",
            "-DNDEBUG"
        }
    },
    files = {
        "src/infiniop/ops/gather/cpu/gather_cpu.cc"
    }
}