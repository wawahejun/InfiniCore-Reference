{
    depfiles = "operator.o: src/infiniop/ops/tril/operator.cc  src/infiniop/ops/tril/../../operator.h  include/infiniop/operator_descriptor.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/tensor_descriptor.h  src/infiniop/ops/tril/../../handle.h include/infiniop/handle.h  include/infiniop/ops/tril.h  include/infiniop/ops/../operator_descriptor.h  src/infiniop/ops/tril/cpu/tril_cpu.h  src/infiniop/ops/tril/cpu/../../../operator.h  src/infiniop/ops/tril/cpu/../../../devices/cpu/cpu_handle.h  src/infiniop/ops/tril/cpu/../../../devices/cpu/../../handle.h\
",
    depfiles_format = "gcc",
    files = {
        "src/infiniop/ops/tril/operator.cc"
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