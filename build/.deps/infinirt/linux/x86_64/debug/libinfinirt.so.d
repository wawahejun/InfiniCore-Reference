{
    files = {
        "build/.objs/infinirt/linux/x86_64/debug/src/infinirt/infinirt.cc.o",
        "build/linux/x86_64/debug/libinfini-utils.a",
        "build/linux/x86_64/debug/libinfinirt-cpu.a"
    },
    values = {
        "/home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/gcc-11.3.0-7tpmmhoar763gi2qhigyczd2vqqhpgxk/bin/g++",
        {
            "-shared",
            "-m64",
            "-fPIC",
            "-Lbuild/linux/x86_64/debug",
            "-linfinirt-cpu",
            "-linfini-utils",
            "-fopenmp"
        }
    }
}