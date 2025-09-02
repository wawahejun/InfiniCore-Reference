{
    depfiles = "main.o: src/utils-test/main.cc src/utils-test/utils_test.h  src/utils-test/../utils.h src/utils-test/../utils/custom_types.h  src/utils-test/../utils/rearrange.h src/utils-test/../utils/result.hpp  src/utils-test/../utils/check.h include/infinicore.h\
",
    depfiles_format = "gcc",
    files = {
        "src/utils-test/main.cc"
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
            "-fexec-charset=UTF-8"
        }
    }
}