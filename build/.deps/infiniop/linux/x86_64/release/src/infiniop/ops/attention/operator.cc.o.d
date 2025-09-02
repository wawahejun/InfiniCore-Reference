{
    depfiles = "operator.o: src/infiniop/ops/attention/operator.cc  src/infiniop/ops/attention/../../operator.h  include/infiniop/operator_descriptor.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/tensor_descriptor.h  src/infiniop/ops/attention/../../../utils.h  src/infiniop/ops/attention/../../../utils/custom_types.h  src/infiniop/ops/attention/../../../utils/rearrange.h  src/infiniop/ops/attention/../../../utils/result.hpp  src/infiniop/ops/attention/../../../utils/check.h include/infinicore.h  src/infiniop/ops/attention/../../../utils/check.h  src/infiniop/ops/attention/../../handle.h include/infiniop/handle.h  src/infiniop/ops/attention/../../tensor.h  include/infiniop/tensor_descriptor.h  src/infiniop/ops/attention/../../../utils.h  include/infiniop/ops/attention.h  include/infiniop/ops/../operator_descriptor.h  include/infiniop/ops/gemm.h include/infiniop/ops/swiglu.h  include/infiniop/ops/causal_softmax.h include/infiniop/ops/gemm.h  include/infiniop/ops/rearrange.h\
",
    depfiles_format = "gcc",
    values = {
        "/home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/gcc-11.3.0-7tpmmhoar763gi2qhigyczd2vqqhpgxk/bin/g++",
        {
            "-m64",
            "-fPIC",
            "-O3",
            "-std=c++17",
            "-Iinclude",
            "-DENABLE_CPU_API",
            "-DENABLE_OMP",
            "-DENABLE_NVIDIA_API",
            "-DENABLE_CUDNN_API",
            "-finput-charset=UTF-8",
            "-fexec-charset=UTF-8",
            "-DNDEBUG"
        }
    },
    files = {
        "src/infiniop/ops/attention/operator.cc"
    }
}