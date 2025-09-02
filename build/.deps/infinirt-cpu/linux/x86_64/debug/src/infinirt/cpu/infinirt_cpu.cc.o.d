{
    depfiles = "infinirt_cpu.o: src/infinirt/cpu/infinirt_cpu.cc  src/infinirt/cpu/infinirt_cpu.h src/infinirt/cpu/../infinirt_impl.h  include/infinirt.h include/infinicore.h\
",
    depfiles_format = "gcc",
    files = {
        "src/infinirt/cpu/infinirt_cpu.cc"
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
            "-fopenmp",
            "-fPIC"
        }
    }
}