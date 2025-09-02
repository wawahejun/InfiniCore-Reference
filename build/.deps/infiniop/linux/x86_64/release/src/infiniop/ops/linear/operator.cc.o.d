{
    depfiles = "operator.o: src/infiniop/ops/linear/operator.cc  src/infiniop/ops/linear/../../operator.h  include/infiniop/operator_descriptor.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/tensor_descriptor.h  src/infiniop/ops/linear/../../handle.h include/infiniop/handle.h  include/infiniop/ops/linear.h  include/infiniop/ops/../operator_descriptor.h  src/infiniop/ops/linear/cpu/linear_cpu.h  src/infiniop/ops/linear/cpu/../../../operator.h  src/infiniop/ops/linear/cpu/../../../devices/cpu/cpu_handle.h  src/infiniop/ops/linear/cpu/../../../devices/cpu/../../handle.h  src/infiniop/ops/linear/nvidia/linear_nvidia.cuh  src/infiniop/ops/linear/nvidia/../../../operator.h  src/infiniop/ops/linear/nvidia/../../../devices/nvidia/nvidia_handle.h  src/infiniop/ops/linear/nvidia/../../../devices/nvidia/../../handle.h\
",
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
    depfiles_format = "gcc",
    files = {
        "src/infiniop/ops/linear/operator.cc"
    }
}