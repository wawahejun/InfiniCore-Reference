# InfiniCore-Intern

# 项目文档指引
- 算子跨平台测试说明：详见 [算子跨平台测试](./doc/README.md)
- 完成算子总量说明：本次开发共完成 **40个算子** 的开发、测试与整合工作，覆盖easy_ops与mid_ops全部范畴，含正向算子、反向算子及logsoftmax ppl专项算子，具体算子清单对应下述各模块表格。


## 算子模块 easy_ops-1

| 算子名称 | 最低支持类型 | 最低支持要求 |
| --- | --- | --- |
| Exp | f32, f16, bf16 | 单目算子，输入输出类型一致，对齐 torch.exp |
| Sin | f32, f16, bf16 | 单目算子，输入输出类型一致，对齐 torch.sin |
| Cos | f32, f16, bf16 | 单目算子，输入输出类型一致，对齐 torch.cos |
| LeakyReLU | f32, f16, bf16 | 单目算子，输入输出类型一致，对齐 torch.nn.LeakyReLU（negative_slope为构建算子时的float类型常数） |
| Tanh | f32, f16, bf16 | 单目算子，输入输出类型一致，对齐 torch.tanh |
| Sigmoid Backward | f32, f16, bf16 | 输入输出类型一致，为Sigmoid函数的反向算子，对齐 torch.sigmoid 反向逻辑 |
| HardSwish | f32, f16, bf16 | 单目算子，对齐 torch.nn.Hardswish |
| Cast | - | 支持整数类型(int32, int64, uint32, uint64)互转、浮点类型(f64, f32, f16)互转及整数转浮点，无直接对齐torch算子，参考torch类型转换逻辑 |
| Where | a, b, c 所有合法类型（condition为bool） | 三目算子，a、b、c类型一致，对齐 torch.where |


## 算子模块 easy_ops-2

| 算子名称 | 最低支持类型 | 最低支持要求 |
| --- | --- | --- |
| Silu | f32, f16, bf16 | 单目算子，输入输出类型一致，对齐 torch.silu |
| Div | f32, f16, bf16 | 输入输出类型一致，对齐 torch.div |
| And | bool | 双目算子，对齐 torch.logical_and |
| Or | bool | 双目算子，对齐 torch.logical_or |
| ReLU Backward | f32, f16, bf16 | 输入输出类型一致，为ReLU函数的反向算子，对齐 torch.relu 反向逻辑 |
| GeLU | f32, f16, bf16 | 单目算子，使用tanh近似实现，对齐 torch.nn.functional.gelu |
| GeLU Backward | f32, f16, bf16 | GeLU的反向算子，使用tanh近似实现，对齐 torch.nn.functional.gelu 反向逻辑 |
| CrossEntropyLoss Backward | f32, f16, bf16 | CrossEntropyLoss的反向算子，对齐 torch.nn.functional.cross_entropy 反向逻辑（grad_logits = (probs - target) / N，probs为概率，target为与logits同形的one-hot张量） |
| Equal | a,b 所有合法类型（c为bool） | 双目算子，对齐 torch.equal |


## 算子模块 mid_ops-1

| 算子名称 | 最低支持类型 | 最低支持要求 |
| --- | --- | --- |
| ReduceMax | f32, f16, bf16 | 对张量dim维度（dim为size_t类型常数）求最大值，规约维度保留且为1，对齐 torch.max |
| ReduceMean | f32, f16, bf16 | 对张量dim维度（dim为size_t类型常数）求平均值，规约维度保留且为1，对齐 torch.mean |
| BatchNorm | f32, f16, bf16 | 至少支持3维(Batch, Channel, Dim)全连续张量，对齐 torch.nn.BatchNorm1d |
| BatchNorm Backward | f32, f16, bf16 | BatchNorm的反向算子，对齐 torch.nn.BatchNorm1d 反向逻辑（running_mean、running_var为正向计算结果） |
| LayerNorm | f32, f16, bf16 | 至少支持3D输入，对最后一维归一化（最后一维需连续），支持无bias场景，对齐 torch.nn.LayerNorm |
| LayerNorm Backward | f32, f16, bf16 | LayerNorm的反向算子，对齐 torch.nn.LayerNorm 反向逻辑（input_std_deviation、input_standardization为正向计算结果） |
| RMSNorm Backward | f32, f16, bf16 | RMSNorm的反向算子，至少支持3D输入，默认对最后一维归一化（最后一维需连续），对齐正向RMSNorm逻辑（参考torch相关归一化算子设计） |


## 算子模块 mid_ops-2

| 算子名称 | 最低支持类型 | 最低支持要求 |
| --- | --- | --- |
| IndexCopyInplace | 所有合法类型 | 支持任意步长，对齐 torch.Tensor.index_copy_ |
| Gather | 所有合法类型 | 不考虑sparse_grad，对齐 torch.gather |
| Scatter | 所有合法类型 | 不考虑reduce，对齐 torch.Tensor.scatter_ |
| tril | 所有合法类型 | 仅支持2D连续张量，对齐 torch.tril |
| triu | 所有合法类型 | 仅支持2D连续张量，对齐 torch.triu |
| Linear | f32, f16, bf16 | bias为1D、weight为2D，支持无bias场景，对齐 torch.nn.functional.linear |
| Linear Backward | f32, f16, bf16 | Linear的反向算子，支持无bias场景，对齐 torch.nn.functional.linear 反向逻辑 |


## 算子模块 mid_ops-3

| 算子名称 | 最低支持类型 | 最低支持要求 |
| --- | --- | --- |
| CrossEntropyLoss | f32, f16, bf16 | 忽略optional参数，对齐 torch.nn.functional.cross_entropy |
| AveragePool | f32, f16, bf16 | 支持1D-3D场景，仅含kernel_size/stride/padding/ceil_mode参数，对齐 torch.nn.AvgPool1d/2d/3d |
| AveragePool Backward | f32, f16, bf16 | AveragePool的反向算子，支持1D-3D场景，对齐 torch.nn.AvgPool1d/2d/3d 反向逻辑 |
| MaxPool | f32, f16, bf16 | 支持1D-3D场景，仅含kernel_size/stride/padding/ceil_mode参数，对齐 torch.nn.functional.max_pool1d/2d/3d |
| MaxPool Backward | f32, f16, bf16 | MaxPool的反向算子，支持1D-3D场景，对齐 torch.nn.functional.max_pool1d/2d/3d 反向逻辑 |
| InterpolateNearest | f32, f16, bf16, int8 | 对齐 torch.nn.functional.interpolate(..., mode='nearest') |
| Conv Backward | f32, f16, bf16 | 支持1D-3D场景，grad_bias为可选输出（取决于是否含bias），对齐正向Conv算子逻辑（参考torch.nn.Conv1d/2d/3d反向设计） |


## PPL专项算子补充
| 算子名称 | 开发说明 | 完成时间 |
| --- | --- | --- |
| logsoftmax | 对齐torch.log_softmax逻辑，实现细节参考 [项目issue文档](./doc/ISSUE.md) 与 [算子跨平台测试文档](./doc/README.md) | 8月13日 |


## 算子开发时间表
| 日期 | 完成任务 |
| --- | --- |
| 7月17日 | 完成并提交训练营测评工具的文档与代码 |
| 7月21日 | 启动算子开发工作，同步熟悉项目框架 |
| 8月4日 | 完成 easy_ops-1 模块全部9个算子的NVIDIA与MetaX平台开发及测试 |
| 8月6日 | 完成除 BatchNorm、LayerNorm、RMSNorm 三个反向算子外，其余算子的NVIDIA与MetaX平台开发及测试 |
| 8月11日 | 完成 easy_ops-1 模块算子的全平台测试 |
| 8月13日 | 完成 logsoftmax 算子的编写 |
| 8月17日 | 完成 easy_ops-2 模块全部9个算子的全平台开发及测试 |
| 8月20日 | 完成 mid_ops-1 模块全部7个算子的NVIDIA平台开发及测试 |
| 8月21日 | 完成所有简单算子的整合工作，及全平台测试验证 |
| 8月22日 | 完成 mid_ops-2 模块除 IndexCopyInplace、Scatter 外，其余5个算子的NVIDIA与MetaX平台开发及测试 |
| 8月23日 | 完成 mid_ops-3 模块全部7个算子的NVIDIA平台开发及测试 |
| 8月24日 | 完成 mid_ops-2 模块全部7个算子的开发，实现该模块全算子全平台测试通过 |
| 9月2日 | 完成 mid_ops-2 模块算子的GGUF测试文件开发，及全平台测试验证 |
| 9月7日 | 完成 mid_ops-1 模块除 BatchNorm Backward、LayerNorm Backward 外，其余5个算子的GGUF测试文件开发及全平台测试 |
| 9月8日 | 完成 mid_ops-3 模块全部算子的GGUF测试文件在NVIDIA平台的测试验证 |
| 9月9日 | 完成以上所有工作的算子的pytorch profile测试功能的编写与验证测试 |
