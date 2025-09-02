{
    values = {
        "/data4/spack/spack/opt/spack/linux-icelake/cuda-12.8.0-dns3hb25ruasvf4rqunuvqefbovgey2m/bin/nvcc",
        {
            "-L/data4/spack/spack/opt/spack/linux-icelake/cuda-12.8.0-dns3hb25ruasvf4rqunuvqefbovgey2m/lib64",
            "-Lbuild/linux/x86_64/release",
            "-lcudart",
            "-linfini-utils",
            "-lcudadevrt",
            "-lrt",
            "-lpthread",
            "-ldl",
            "-Xcompiler=-fPIC",
            "-m64",
            "-ccbin=/home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/gcc-11.3.0-7tpmmhoar763gi2qhigyczd2vqqhpgxk/bin/g++",
            "-dlink"
        }
    },
    files = {
        "build/.objs/infinirt-nvidia/linux/x86_64/release/src/infinirt/cuda/infinirt_cuda.cu.o",
        "build/linux/x86_64/release/libinfini-utils.a"
    }
}