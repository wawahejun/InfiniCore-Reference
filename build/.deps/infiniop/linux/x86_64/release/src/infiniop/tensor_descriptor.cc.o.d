{
    depfiles = "tensor_descriptor.o: src/infiniop/tensor_descriptor.cc  src/infiniop/../utils.h src/infiniop/../utils/custom_types.h  src/infiniop/../utils/rearrange.h src/infiniop/../utils/result.hpp  src/infiniop/../utils/check.h include/infinicore.h src/infiniop/tensor.h  include/infiniop/tensor_descriptor.h include/infiniop/../infinicore.h\
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
        "src/infiniop/tensor_descriptor.cc"
    }
}