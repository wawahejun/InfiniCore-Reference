{
    depfiles = "infiniccl.o: src/infiniccl/infiniccl.cc include/infiniccl.h  include/infinirt.h include/infinicore.h  src/infiniccl/./ascend/infiniccl_ascend.h  src/infiniccl/./ascend/../infiniccl_impl.h  src/infiniccl/./cambricon/infiniccl_cambricon.h  src/infiniccl/./cambricon/../infiniccl_impl.h  src/infiniccl/./cuda/infiniccl_cuda.h  src/infiniccl/./cuda/../infiniccl_impl.h  src/infiniccl/./metax/infiniccl_metax.h  src/infiniccl/./metax/../infiniccl_impl.h\
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
        "src/infiniccl/infiniccl.cc"
    }
}