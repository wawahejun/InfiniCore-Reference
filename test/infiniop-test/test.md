#### Exp算子
```bash
# f32精度
srun ../../build/linux/x86_64/release/infiniop-test exp_f32.gguf --metax --warmup 20 --run 1000 2>&1 | tee -a infiniop_test.log

# f16精度
srun ../../build/linux/x86_64/release/infiniop-test exp_f16.gguf --metax --warmup 20 --run 1000 2>&1 | tee -a infiniop_test.log

# bf16精度
srun ../../build/linux/x86_64/release/infiniop-test exp_bf16.gguf --metax --warmup 20 --run 1000 2>&1 | tee -a infiniop_test.log
```

#### Sin算子
```bash
# f32精度
srun ../../build/linux/x86_64/release/infiniop-test sin_f32.gguf --metax --warmup 20 --run 1000 2>&1 | tee -a infiniop_test.log

# f16精度
srun ../../build/linux/x86_64/release/infiniop-test sin_f16.gguf --metax --warmup 20 --run 1000 2>&1 | tee -a infiniop_test.log

# bf16精度
srun ../../build/linux/x86_64/release/infiniop-test sin_bf16.gguf --metax --warmup 20 --run 1000 2>&1 | tee -a infiniop_test.log
```

#### Cos算子
```bash
# f32精度
srun ../../build/linux/x86_64/release/infiniop-test cos_f32.gguf --metax --warmup 20 --run 1000 2>&1 | tee -a infiniop_test.log

# f16精度
srun ../../build/linux/x86_64/release/infiniop-test cos_f16.gguf --metax --warmup 20 --run 1000 2>&1 | tee -a infiniop_test.log

# bf16精度
srun ../../build/linux/x86_64/release/infiniop-test cos_bf16.gguf --metax --warmup 20 --run 1000 2>&1 | tee -a infiniop_test.log
```

#### LeakyReLU算子
```bash
# f32精度
srun ../../build/linux/x86_64/release/infiniop-test leaky_relu_f32.gguf --metax --warmup 20 --run 1000 2>&1 | tee -a infiniop_test.log

# f16精度
srun ../../build/linux/x86_64/release/infiniop-test leaky_relu_f16.gguf --metax --warmup 20 --run 1000 2>&1 | tee -a infiniop_test.log

# bf16精度
srun ../../build/linux/x86_64/release/infiniop-test leaky_relu_bf16.gguf --metax --warmup 20 --run 1000 2>&1 | tee -a infiniop_test.log
```

#### Tanh算子
```bash
# f32精度
srun ../../build/linux/x86_64/release/infiniop-test tanh_f32.gguf --metax --warmup 20 --run 1000 2>&1 | tee -a infiniop_test.log

# f16精度
srun ../../build/linux/x86_64/release/infiniop-test tanh_f16.gguf --metax --warmup 20 --run 1000 2>&1 | tee -a infiniop_test.log

# bf16精度
srun ../../build/linux/x86_64/release/infiniop-test tanh_bf16.gguf --metax --warmup 20 --run 1000 2>&1 | tee -a infiniop_test.log
```

#### Sigmoid Backward算子
```bash
# f32精度
srun ../../build/linux/x86_64/release/infiniop-test sigmoid_backward_f32.gguf --metax --warmup 20 --run 1000 2>&1 | tee -a infiniop_test.log

# f16精度
srun ../../build/linux/x86_64/release/infiniop-test sigmoid_backward_f16.gguf --metax --warmup 20 --run 1000 2>&1 | tee -a infiniop_test.log

# bf16精度
srun ../../build/linux/x86_64/release/infiniop-test sigmoid_backward_bf16.gguf --metax --warmup 20 --run 1000 2>&1 | tee -a infiniop_test.log
```

#### HardSwish算子
```bash
# f32精度
srun ../../build/linux/x86_64/release/infiniop-test hardswish_f32.gguf --metax --warmup 20 --run 1000 2>&1 | tee -a infiniop_test.log

# f16精度
srun ../../build/linux/x86_64/release/infiniop-test hardswish_f16.gguf --metax --warmup 20 --run 1000 2>&1 | tee -a infiniop_test.log

# bf16精度
srun ../../build/linux/x86_64/release/infiniop-test hardswish_bf16.gguf --metax --warmup 20 --run 1000 2>&1 | tee -a infiniop_test.log
```

```bash
srun ../../build/linux/x86_64/release/infiniop-test cast.gguf --metax --warmup 20 --run 1000 2>&1 | tee -a infiniop_test.log


../../build/linux/x86_64/release/infiniop-test where.gguf --nvidia --warmup 20 --run 1000 2>&1 | tee -a infiniop_test.log
```