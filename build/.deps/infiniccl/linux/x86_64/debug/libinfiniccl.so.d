{
    files = {
        "build/.objs/infiniccl/linux/x86_64/debug/src/infiniccl/infiniccl.cc.o",
        "build/linux/x86_64/debug/libinfini-utils.a",
        "build/linux/x86_64/debug/libinfinirt-cpu.a"
    },
    values = {
        "/home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/gcc-11.3.0-7tpmmhoar763gi2qhigyczd2vqqhpgxk/bin/g++",
        {
            "-shared",
            "-fPIC",
            "-m64",
            "-Lbuild/linux/x86_64/debug",
            "-linfinirt",
            "-linfinirt-cpu",
            "-linfini-utils",
            "-fopenmp"
        }
    }
}