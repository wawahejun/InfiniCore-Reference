{
    depfiles = "operator.o: src/infiniop/ops/conv/operator.cc  src/infiniop/ops/conv/../../operator.h  include/infiniop/operator_descriptor.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/tensor_descriptor.h  src/infiniop/ops/conv/../../handle.h include/infiniop/handle.h  include/infiniop/ops/conv.h  include/infiniop/ops/../operator_descriptor.h  src/infiniop/ops/conv/cpu/conv_cpu.h src/infiniop/ops/conv/cpu/../conv.h  src/infiniop/ops/conv/cpu/../../../operator.h  src/infiniop/ops/conv/cpu/../info.h  src/infiniop/ops/conv/cpu/../../../../utils.h  src/infiniop/ops/conv/cpu/../../../../utils/custom_types.h  src/infiniop/ops/conv/cpu/../../../../utils/rearrange.h  src/infiniop/ops/conv/cpu/../../../../utils/result.hpp  src/infiniop/ops/conv/cpu/../../../../utils/check.h include/infinicore.h  src/infiniop/ops/conv/cpu/../../../tensor.h  include/infiniop/tensor_descriptor.h  src/infiniop/ops/conv/cpu/../../../../utils.h\
",
    depfiles_format = "gcc",
    files = {
        "src/infiniop/ops/conv/operator.cc"
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