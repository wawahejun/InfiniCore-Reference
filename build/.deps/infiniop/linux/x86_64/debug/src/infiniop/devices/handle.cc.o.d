{
    depfiles = "handle.o: src/infiniop/devices/handle.cc include/infiniop/handle.h  include/infiniop/../infinicore.h src/infiniop/devices/../../utils.h  src/infiniop/devices/../../utils/custom_types.h  src/infiniop/devices/../../utils/rearrange.h  src/infiniop/devices/../../utils/result.hpp  src/infiniop/devices/../../utils/check.h include/infinicore.h  include/infinirt.h include/infinicore.h  src/infiniop/devices/cpu/cpu_handle.h  src/infiniop/devices/cpu/../../handle.h\
",
    depfiles_format = "gcc",
    files = {
        "src/infiniop/devices/handle.cc"
    },
    values = {
        "/home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/gcc-11.3.0-7tpmmhoar763gi2qhigyczd2vqqhpgxk/bin/g++",
        {
            "-m64",
            "-fPIC",
            "-g",
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