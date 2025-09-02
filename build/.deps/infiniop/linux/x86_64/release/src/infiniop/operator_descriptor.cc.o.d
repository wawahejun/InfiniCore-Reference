{
    depfiles = "operator_descriptor.o: src/infiniop/operator_descriptor.cc  src/infiniop/operator.h include/infiniop/operator_descriptor.h  include/infiniop/handle.h include/infiniop/../infinicore.h  include/infiniop/tensor_descriptor.h\
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
        "src/infiniop/operator_descriptor.cc"
    }
}