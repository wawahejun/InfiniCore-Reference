{
    depfiles = "operator.o: src/infiniop/ops/linear_backward/operator.cc  src/infiniop/ops/linear_backward/../../operator.h  include/infiniop/operator_descriptor.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/tensor_descriptor.h  src/infiniop/ops/linear_backward/../../handle.h  include/infiniop/handle.h include/infiniop/ops/linear_backward.h  include/infiniop/ops/../operator_descriptor.h  src/infiniop/ops/linear_backward/cpu/linear_backward_cpu.h  src/infiniop/ops/linear_backward/cpu/../../../operator.h  src/infiniop/ops/linear_backward/cpu/../../../devices/cpu/cpu_handle.h  src/infiniop/ops/linear_backward/cpu/../../../devices/cpu/../../handle.h  src/infiniop/ops/linear_backward/nvidia/linear_backward_nvidia.cuh  src/infiniop/ops/linear_backward/nvidia/../../../operator.h  src/infiniop/ops/linear_backward/nvidia/../../../devices/nvidia/nvidia_handle.h  src/infiniop/ops/linear_backward/nvidia/../../../devices/nvidia/../../handle.h\
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
        "src/infiniop/ops/linear_backward/operator.cc"
    }
}