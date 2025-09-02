{
    depfiles = "rearrange.o: src/utils/rearrange.cc src/utils/rearrange.h  src/utils/result.hpp src/utils/check.h include/infinicore.h\
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
        "src/utils/rearrange.cc"
    }
}