{
    files = {
        "src/infiniop-test/src/file_mapping.cpp"
    },
    depfiles_format = "gcc",
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
            "-I/home/intern/.infini/include",
            "-I/home/intern/junyi/InfiniCore-Intern/src/infiniop-test/include",
            "-DDEBUG_MODE",
            "-DENABLE_CPU_API",
            "-DENABLE_OMP",
            "-DENABLE_CUDNN_API",
            "-finput-charset=UTF-8",
            "-fexec-charset=UTF-8",
            "-fopenmp"
        }
    },
    depfiles = "file_mapping.o: src/infiniop-test/src/file_mapping.cpp  /home/intern/junyi/InfiniCore-Intern/src/infiniop-test/include/file_mapping.hpp\
"
}