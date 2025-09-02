{
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
            "-I/home/intern/.infini/include",
            "-I/home/intern/junyi/InfiniCore-Intern/src/infiniop-test/include",
            "-DENABLE_CPU_API",
            "-DENABLE_OMP",
            "-DENABLE_NVIDIA_API",
            "-DENABLE_CUDNN_API",
            "-finput-charset=UTF-8",
            "-fexec-charset=UTF-8",
            "-fopenmp",
            "-DNDEBUG"
        }
    },
    depfiles_format = "gcc",
    depfiles = "gguf.o: src/infiniop-test/src/gguf.cpp  /home/intern/junyi/InfiniCore-Intern/src/infiniop-test/include/gguf.hpp  /home/intern/junyi/InfiniCore-Intern/src/infiniop-test/include/file_mapping.hpp\
",
    files = {
        "src/infiniop-test/src/gguf.cpp"
    }
}