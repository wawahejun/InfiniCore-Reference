{
    files = {
        "src/infiniop-test/src/ops/rope.cpp"
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
    depfiles = "rope.o: src/infiniop-test/src/ops/rope.cpp  /home/intern/junyi/InfiniCore-Intern/src/infiniop-test/include/ops.hpp  /home/intern/junyi/InfiniCore-Intern/src/infiniop-test/include/test.hpp  /home/intern/junyi/InfiniCore-Intern/src/infiniop-test/include/gguf.hpp  /home/intern/junyi/InfiniCore-Intern/src/infiniop-test/include/file_mapping.hpp  /home/intern/junyi/InfiniCore-Intern/src/infiniop-test/include/tensor.hpp  include/infiniop.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/ops/add.h  include/infiniop/ops/../operator_descriptor.h  include/infiniop/ops/../handle.h  include/infiniop/ops/../tensor_descriptor.h  include/infiniop/ops/../../infinicore.h include/infiniop/ops/and.h  include/infiniop/ops/attention.h include/infiniop/ops/gemm.h  include/infiniop/ops/swiglu.h include/infiniop/ops/cast.h  include/infiniop/ops/causal_softmax.h include/infiniop/ops/clip.h  include/infiniop/ops/conv.h include/infiniop/ops/cos.h  include/infiniop/ops/crossentropyloss_backward.h  include/infiniop/ops/div.h include/infiniop/ops/equal.h  include/infiniop/ops/exp.h include/infiniop/ops/gather.h  include/infiniop/ops/gelu.h include/infiniop/ops/gelu_backward.h  include/infiniop/ops/gemm.h include/infiniop/ops/hardswish.h  include/infiniop/ops/index_copy_inplace.h  include/infiniop/ops/leaky_relu.h include/infiniop/ops/linear.h  include/infiniop/ops/linear_backward.h include/infiniop/ops/mul.h  include/infiniop/ops/or.h include/infiniop/ops/random_sample.h  include/infiniop/ops/rearrange.h include/infiniop/ops/relu.h  include/infiniop/ops/relu_backward.h include/infiniop/ops/rms_norm.h  include/infiniop/ops/rope.h include/infiniop/ops/scatter.h  include/infiniop/ops/sigmoid_backward.h include/infiniop/ops/silu.h  include/infiniop/ops/sin.h include/infiniop/ops/sub.h  include/infiniop/ops/swiglu.h include/infiniop/ops/tanh.h  include/infiniop/ops/tril.h include/infiniop/ops/triu.h  include/infiniop/ops/where.h include/infiniop/tensor_descriptor.h  /home/intern/junyi/InfiniCore-Intern/src/infiniop-test/include/utils.hpp  /home/intern/junyi/InfiniCore-Intern/src/infiniop-test/include/../../utils.h  /home/intern/junyi/InfiniCore-Intern/src/infiniop-test/include/../../utils/custom_types.h  /home/intern/junyi/InfiniCore-Intern/src/infiniop-test/include/../../utils/rearrange.h  /home/intern/junyi/InfiniCore-Intern/src/infiniop-test/include/../../utils/result.hpp  /home/intern/junyi/InfiniCore-Intern/src/infiniop-test/include/../../utils/check.h  include/infinicore.h include/infinirt.h include/infinicore.h\
"
}