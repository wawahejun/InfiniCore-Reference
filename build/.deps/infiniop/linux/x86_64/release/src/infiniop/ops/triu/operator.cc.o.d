{
    depfiles = "operator.o: src/infiniop/ops/triu/operator.cc  src/infiniop/ops/triu/../../operator.h  include/infiniop/operator_descriptor.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/tensor_descriptor.h  src/infiniop/ops/triu/../../handle.h include/infiniop/handle.h  include/infiniop/ops/triu.h  include/infiniop/ops/../operator_descriptor.h  src/infiniop/ops/triu/cpu/triu_cpu.h  src/infiniop/ops/triu/cpu/../../../operator.h  src/infiniop/ops/triu/cpu/../../../devices/cpu/cpu_handle.h  src/infiniop/ops/triu/cpu/../../../devices/cpu/../../handle.h  src/infiniop/ops/triu/nvidia/triu_nvidia.cuh  src/infiniop/ops/triu/nvidia/../../../operator.h  src/infiniop/ops/triu/nvidia/../../../devices/nvidia/nvidia_handle.h  src/infiniop/ops/triu/nvidia/../../../devices/nvidia/../../handle.h\
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
        "src/infiniop/ops/triu/operator.cc"
    }
}