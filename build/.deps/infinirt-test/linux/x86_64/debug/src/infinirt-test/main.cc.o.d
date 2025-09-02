{
    depfiles = "main.o: src/infinirt-test/main.cc src/infinirt-test/test.h  src/infinirt-test/../utils.h src/infinirt-test/../utils/custom_types.h  src/infinirt-test/../utils/rearrange.h  src/infinirt-test/../utils/result.hpp src/infinirt-test/../utils/check.h  include/infinicore.h include/infinirt.h include/infinicore.h\
",
    depfiles_format = "gcc",
    files = {
        "src/infinirt-test/main.cc"
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