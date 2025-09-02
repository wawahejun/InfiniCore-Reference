{
    depfiles = "reduce.o: src/infiniop/reduce/cpu/reduce.cc  src/infiniop/reduce/cpu/reduce.h  src/infiniop/reduce/cpu/../../../utils.h  src/infiniop/reduce/cpu/../../../utils/custom_types.h  src/infiniop/reduce/cpu/../../../utils/rearrange.h  src/infiniop/reduce/cpu/../../../utils/result.hpp  src/infiniop/reduce/cpu/../../../utils/check.h include/infinicore.h\
",
    depfiles_format = "gcc",
    files = {
        "src/infiniop/reduce/cpu/reduce.cc"
    },
    values = {
        "/home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/gcc-11.3.0-7tpmmhoar763gi2qhigyczd2vqqhpgxk/bin/g++",
        {
            "-m64",
            "-g",
            "-Wall",
            "-Werror",
            "-O0",
            "-std=c++17",
            "-Iinclude",
            "-DDEBUG_MODE",
            "-DENABLE_CPU_API",
            "-DENABLE_OMP",
            "-DENABLE_CUDNN_API",
            "-finput-charset=UTF-8",
            "-fexec-charset=UTF-8",
            "-fPIC",
            "-Wno-unknown-pragmas",
            "-fopenmp"
        }
    }
}