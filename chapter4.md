# 第4章：计算核心设计

在上一章中，我们从系统层面了解了NPU的整体架构。现在，让我们深入到NPU的心脏——计算核心。如果说NPU是一座高效运转的工厂，那么计算核心就是工厂里的生产线，而MAC（Multiply-Accumulate）单元则是生产线上的工人。本章将详细探讨如何设计高效的计算核心，从基础的MAC单元开始，逐步构建起能够处理海量神经网络运算的脉动阵列。

我们将重点关注三个关键问题：**如何设计单个MAC单元以实现最高效率？如何将成千上万个MAC单元组织成阵列？如何通过不同的数据流模式（Weight Stationary、Output Stationary、Row Stationary）来优化不同场景下的计算效率？**通过回答这些问题，你将掌握NPU计算核心设计的精髓。

## <a name="41"></a>4.1 MAC阵列设计

### 4.1.1 基础MAC单元

MAC (Multiply-Accumulate) 是NPU的基本计算单元，执行 `C = C + A × B` 运算。

计算核心的演进历程是一部从标量到张量的进化史。就像生物从单细胞进化到多细胞生物，计算单元也经历了类似的演变：

**计算架构的演进历程**

1. **标量处理器时代（1980s-1990s）：** 一次处理一个数据，就像用筷子一粒一粒地夹米饭。
2. **SIMD时代（2000s）：** 引入向量处理，一条指令处理多个数据，就像用勺子一次舀起多粒米饭。代表作：Intel SSE/AVX。
3. **GPU时代（2010s）：** 大规模并行处理，成千上万个核心同时工作，像一个巨大的自助餐厅，数百个人同时进餐。
4. **NPU/TPU时代（2015s-）：** 专为矩阵运算优化，引入脉动阵列和张量核心，就像高度自动化的寿司传送带，每个工位专注完成特定任务。
5. **未来：模拟计算与存内计算（2025+）：** 打破冯·诺依曼架构，计算与存储融合，像大脑神经元般工作。

#### 为什么MAC如此重要？

深度学习的本质是大量的矩阵运算，而矩阵运算可以分解为无数个MAC操作。一个简单的全连接层计算可以表示为：

**全连接层的数学表达式：**

Y = W × X + B

其中分解为MAC操作的计算复杂度为O(M×N×K)。例如，一个1024×1024的矩阵乘法需要约10.7亿次MAC操作。

这就是为什么现代AI芯片都在疯狂堆砌MAC单元的原因。Google TPU v1拥有65,536个MAC单元，而最新的NVIDIA H100则包含了数百万个等效MAC单元。

**优化的流水线MAC单元设计：**

MAC单元采用3级流水线设计：
- 第一级：输入寄存，缓存激活值(a)、权重(b)和部分和(c)
- 第二级：乘法运算，计算a×b
- 第三级：累加运算，执行c + (a×b)

关键设计特点：
- 支持INT8输入，ACC_WIDTH=32防止溢出
- 流水线化提高吞吐率，每周期可处理一个新的MAC操作
- 包含有效信号(valid_out)用于流控制

**Chisel高级综合语言实现：**

使用Chisel可以更简洁地描述MAC单元的流水线结构：
- 利用RegEnable实现条件寄存
- 使用RegNext传递控制信号
- 自动类型推导和位宽管理
- 可生成Verilog代码用于后端实现

### 4.1.2 多精度MAC设计

多精度设计是现代NPU的关键创新。不同的应用场景对精度的需求差异巨大，就像不同的工作需要不同精度的工具——外科手术需要手术刀，而拆墙只需要大锤。

#### 功耗-性能-面积（PPA）权衡的具体数字

以下是基于7nm工艺的实际测量数据（相对于FP32）：

| 数据类型 | 乘法器面积 | 功耗 | 延迟 | 应用场景 |
|---------|-----------|------|------|---------|
| INT4 | 16 gates | 0.1x | 1 cycle | 极低功耗推理 |
| INT8 | 64 gates | 0.25x | 1 cycle | 主流推理 |
| FP16 | ~400 gates | 0.4x | 2 cycles | 训练/高精度推理 |
| FP32 | ~1600 gates | 1.0x | 3 cycles | 科学计算/训练 |

**设计陷阱：精度选择的常见误区**

- **误区1：越低精度越好。** 实际上，过度量化会导致精度崩塌。例如，ResNet-50在INT8下精度损失小于1%，但在INT4下可能损失超过5%。
- **误区2：统一精度设计。** 现代NPU采用混合精度，例如权重用INT4，激活值用INT8，累加器用INT32，这样可以在保持精度的同时最大化效率。
- **误区3：忽视量化友好性。** 不是所有模型都适合量化。Transformer类模型对量化更敏感，需要特殊的量化策略。

#### 真实世界的创新案例

**1. NVIDIA Tensor Core的演进：**
- **第一代（Volta）：** 只支持FP16混合精度，4×4×4矩阵运算
- **第二代（Turing）：** 增加INT8/INT4支持，引入稀疏加速
- **第三代（Ampere）：** 支持TF32（19位精度），结构化稀疏2:4
- **第四代（Hopper）：** 支持FP8（E4M3/E5M2），8×8×16大矩阵

**2. Google TPU的极简主义：**
TPU v1只支持INT8，通过大规模并行（256×256阵列）弥补精度限制。这种"以量取胜"的策略在推理场景下取得了巨大成功，但在训练场景下不得不在TPU v2中加入FP16/BF16支持。

### 4.1.3 MAC阵列组织

#### 数据复用的数学基础

矩阵乘法C = A×B的计算复杂度为O(M×N×K)，但聪明的数据复用可以大幅降低内存访问。以M=N=K=1024为例：

**三种基本复用模式的理论分析：**

| 复用模式 | 内存访问次数 | 复用因子 | 片上存储需求 | 适用场景 |
|---------|------------|---------|------------|----------|
| 无复用 | 3×N³ | 1 | O(1) | 内存带宽无限 |
| 输入复用 | N³ + 2N² | N | O(N) | 批处理推理 |
| 权重复用 | 2N³ + N² | N | O(N) | 单样本推理 |
| 输出复用 | 2N³ + N² | N | O(N) | 训练场景 |

数学证明：对于分块矩阵乘法，最优分块大小Tₘ×Tₙ×Tₖ应满足：
- Tₘ×Tₙ + Tₘ×Tₖ + Tₙ×Tₖ ≤ 片上存储容量
- max(Tₘ×Tₙ×Tₖ) 获得最大计算密度

#### MAC阵列的拓扑选择

**1. 一维阵列：简单但受限**
- 线性排列，适合向量运算
- 典型规模：32-128个MAC
- 代表：早期DSP和ARM NEON

**2. 二维阵列：主流选择**
- 矩形网格，数据流动规则
- 典型规模：16×16到256×256
- 代表：Google TPU、华为达芬奇

**3. 三维阵列：未来趋势**
- 立方体结构，支持张量运算
- 需要3D封装技术支持
- 代表：Cerebras WSE（概念上的3D）

#### 互连网络的关键设计

**全局互连 vs 局部互连的权衡：**

```
全局互连（Crossbar）          局部互连（Mesh）
┌─┬─┬─┬─┐                  ┌─┬─┬─┬─┐
├─┼─┼─┼─┤                  │ │ │ │ │
├─┼─┼─┼─┤                  ├─┼─┼─┼─┤
├─┼─┼─┼─┤                  │ │ │ │ │
└─┴─┴─┴─┘                  └─┴─┴─┴─┘
面积：O(N²)                 面积：O(N)
延迟：O(1)                  延迟：O(√N)
功耗：高                    功耗：低
```

现代NPU通常采用分层互连：
- **第一层：** PE内部的局部连接
- **第二层：** PE簇（如8×8）内的全连接
- **第三层：** 簇间的片上网络（NoC）

**二维MAC阵列组织设计：**

8×8 MAC阵列的关键设计要点：
- 采用行广播模式，每行共享同一个激活值输入
- 权重存储在本地，形成二维数组
- 使用generate语句实现可参数化的阵列规模
- 部分和的连接方式取决于数据流模式（WS/OS/RS）

## <a name="42"></a>4.2 脉动阵列架构

### 4.2.1 脉动阵列原理

脉动阵列通过数据在PE间的有节奏流动，实现高效的数据复用和规则的计算模式。

脉动阵列（Systolic Array）这个名字来源于心脏的脉动（Systole）。就像心脏有节奏地泵血，数据在脉动阵列中也以固定的节奏在处理单元间流动。这个优雅的概念由孔祥重（H.T. Kung）教授在1978年提出，如今已成为AI芯片的核心架构。

#### 脉动阵列的天才之处

想象一个汽车装配线，每个工人负责安装一个部件。传统方法是每个工人都要去仓库取零件，效率低下。脉动阵列的做法是：让零件在传送带上流动，每个工人从传送带取用需要的零件，完成自己的工作后，将半成品继续传递下去。

**核心优势：**
- 数据复用率高：每个数据被多个PE使用
- 通信局部化：只需要邻近PE间通信
- 控制简单：规则的数据流动模式
- 易于扩展：模块化设计便于增加阵列规模

#### 三种经典的脉动阵列变体

| 类型 | 数据流动方式 | 适用场景 | 代表实现 | 优缺点 |
|------|-------------|---------|---------|--------|
| **Weight Stationary (WS)** | 权重固定在PE中，输入和输出流动 | 卷积层（权重复用高） | Google TPU v1 | ✓ 权重只加载一次<br>✗ 输入/输出带宽需求高 |
| **Output Stationary (OS)** | 输出固定在PE中累加，输入和权重流动 | 大矩阵乘法 | NVIDIA CUTLASS | ✓ 减少部分和读写<br>✗ 权重带宽需求高 |
| **Row Stationary (RS)** | 一行数据驻留，其他数据流动 | 通用计算 | MIT Eyeriss | ✓ 灵活性高<br>✗ 控制复杂 |

#### 现代变体和创新

**1. Google TPU的超大规模脉动阵列：**
TPU v1使用256×256的脉动阵列，这在当时是革命性的。为了支撑如此大的阵列，Google设计了独特的"脉动数据调度器"，能够精确控制数据流入的时机，确保计算单元的利用率接近100%。

**2. Groq的时间编排架构（TSP）：**
Groq将脉动阵列的概念推向极致，整个芯片就是一个巨大的脉动系统。他们抛弃了传统的缓存，所有数据移动都是预先编排好的，像一场精心排练的交响乐。

**3. Cerebras的晶圆级脉动：**
Cerebras WSE-2包含850,000个核心，整个晶圆就是一个巨大的2D脉动阵列。数据可以在任意方向流动，突破了传统芯片的边界限制。

### 4.2.2 Weight Stationary脉动阵列实现
**Weight Stationary脉动阵列PE设计：**

WS型PE的核心特点：
- 权重一次加载后固定在PE内部
- 激活值从上向下流动，部分和从左向右流动
- 每个PE执行MAC运算：psum_out = psum_in + (act × weight)
- 适合权重复用率高的场景（如卷积层）

**4×4 Weight Stationary脉动阵列实现：**

设计要点：
- PE阵列采用二维网格连接
- 激活值从顶部输入，逐层向下传递
- 部分和从左侧输入（初始为0），逐列向右累加
- 使用寄存器阵列存储PE间的数据传递
- 最终结果从最右列输出

**优化的流水线PE单元设计：**

流水线化PE的关键特性：
- 两级流水线设计，提高时钟频率
- 权重寄存器在weight_load信号有效时加载，其他时间保持不变
- MAC计算为组合逻辑，结果在下一个时钟周期输出
- 支持有符号运算，使用$signed进行类型转换

**Chisel实现的脉动阵列：**

Chisel版本的优势：
- 面向对象的模块化设计
- 简洁的数组和循环处理
- 自动的类型推导和连接检查
- 可生成高质量Verilog代码

关键实现细节：
- PE单元封装为class SystolicPE
- 使用Array.fill创建二维PE阵列
- for循环实现PE间的规则连接
- 边界处理通过if语句判断

### 4.2.3 脉动阵列数据流动示例

以2×2矩阵乘法为例，展示数据在脉动阵列中的流动过程。

**矩阵定义：**
- A = [[a00, a01], [a10, a11]]
- B = [[b00, b01], [b10, b11]]
- C = A × B

**数据流动过程：**

时刻0：权重加载，将B矩阵元素固定在对应PE中

时刻1-5：数据流动计算
- A矩阵元素从顶部输入，每列错开一个周期
- 数据在PE间向下和向右流动
- 每个PE执行MAC运算，累加结果向右传递
- 最终结果从PE阵列右侧输出

这种数据流动方式确保了每个输入数据被多个PE复用，提高了数据利用率。

### 4.2.4 Output Stationary 脉动阵列实现

Output Stationary（输出固定）是另一种重要的脉动阵列架构，特别适合深度卷积和批处理场景。在这种架构中，每个PE负责计算输出矩阵的一个固定元素，输入数据和权重在PE阵列中流动。

**Output Stationary脉动阵列设计：**

OS架构的关键特性：
- 累加结果固定在每个PE中，避免部分和的频繁移动
- A矩阵从左侧输入，每行错开一个周期（skew）
- B矩阵从顶部输入，每列错开一个周期
- 使用延迟链实现数据错位，确保正确的矩阵乘法计算
- 适合大批量数据处理和深度卷积

**Output Stationary PE单元设计：**

OS PE的工作原理：
- A数据水平流动（从左到右），B数据垂直流动（从上到下）
- 每个PE内部维护一个累加器，固定计算输出矩阵的一个元素
- 乘法和累加操作流水线化
- 支持clear_acc信号清空累加器，开始新的计算

**Chisel实现的Output Stationary PE：**

Chisel版本特点：
- 使用Bundle封装复杂的IO接口
- RegNext简化寄存器链实现
- when/elsewhen结构清晰表达控制逻辑
- 类型安全的有符号数运算

### 4.2.5 Output Stationary vs Weight Stationary 对比

| 特性 | Weight Stationary | Output Stationary | 适用场景 |
|------|------------------|-------------------|----------|
| 数据复用 | 权重驻留在PE中 | 部分和驻留在PE中 | WS: 批量小<br>OS: 批量大 |
| 内存带宽 | 输入/输出带宽高 | 权重/输入带宽高 | WS: 权重复用多<br>OS: 输出通道多 |
| 控制复杂度 | 简单 | 中等 | WS: 资源受限<br>OS: 性能优先 |
| 延迟 | 较低 | 较高（需要数据对齐） | WS: 实时推理<br>OS: 批处理训练 |
| 能效 | 权重读取能耗低 | 部分和读写能耗低 | WS: 边缘设备<br>OS: 数据中心 |

### 4.2.6 脉动阵列的性能建模

#### 理论性能分析

对于N×N的脉动阵列执行M×K×N的矩阵乘法：

**计算吞吐率：**
- 理论峰值：2N² OPS/cycle（每个PE每周期2次操作）
- 实际吞吐率：2N² × 利用率
- 利用率 = min(M/N, K/N, 1)

**带宽需求：**
```
Weight Stationary:
- 输入带宽： N × 数据宽度 bytes/cycle
- 输出带宽： N × 数据宽度 bytes/cycle
- 权重加载： N² × 数据宽度 bytes（一次性）

Output Stationary:
- 输入带宽： N × 数据宽度 bytes/cycle
- 权重带宽： N × 数据宽度 bytes/cycle
- 输出读写： N² × 数据宽度 bytes（最终）
```

#### 真实世界的性能瓶颈

**1. 数据对齐开销（Data Skewing Overhead）**

脉动阵列需要输入数据按特定时序错位输入，这需要额外的缓冲器：
- Skew buffer大小： N × (N-1)/2 个元素
- 对于256×256阵列，需要32K个缓冲单元
- 增加了N-1个周期的启动延迟

**2. 边界效应（Edge Effects）**

当矩阵尺寸不是N的整数倍时：
- 填充开销：(N - M%N) × (N - K%N) 个无效计算
- 利用率下降：对于129×129矩阵在128×128阵列上，利用率仅25%

**3. 流水线泡沫（Pipeline Bubbles）**

切换不同计算任务时的空闲：
- 清空流水线：N个周期
- 重新加载权重：N个周期
- 总开销：2N个周期/任务切换

#### 现代脉动阵列的创新优化

**1. 双缓冲技术（Double Buffering）**
```
时刻1: Buffer A计算，Buffer B加载下一批数据
时刻2: Buffer B计算，Buffer A加载下一批数据
```
隐藏数据加载延迟，提高利用率

**2. 可变维度支持（Variable Dimension Support）**
- Nvidia Ampere：支持8/16/32/64等多种尺寸
- 通过遮罩（masking）和路由重组实现
- 减少填充开销，提高实际利用率

**3. 稀疏支持（Sparsity Support）**
- 跳过零值计算
- 压缩数据表示
- 动态PE分配

### 4.2.7 未来脉动阵列的发展方向

#### 3D脉动阵列

将传统2D脉动阵列扩展到第三维度：
- 支持张量运算的原生加速
- 利用3D封装技术的垂直互连
- 适合Transformer等多维注意力计算

#### 动态可重构脉动阵列

根据工作负载动态调整：
- 大矩阵：使用整个阵列
- 小矩阵：划分成多个小阵列并行处理
- 稀疏矩阵：重组成稀疏处理模式

#### 近数据计算（Near-Data Computing）

将脉动阵列与存储器集成：
- 消除数据移动的能耗
- 提供更高的内部带宽
- 例子：三星的HBM-PIM技术

## <a name="43"></a>4.3 向量处理单元

### 4.3.1 SIMD架构设计

向量处理单元采用SIMD架构，支持非线性激活、池化等操作。

虽然MAC阵列负责了深度学习中90%以上的计算量，但剩下的10%——激活函数、归一化、池化等操作——同样至关重要。这就像做菜，炒菜占了大部分时间，但最后的调味决定了菜品的成败。

#### 为什么需要专门的向量处理单元？

让MAC阵列处理激活函数就像让推土机绣花——不是不能，而是大材小用。向量处理单元（VPU）专门为这些“轻量级但高频”的操作优化：

**VPU vs MAC阵列的设计权衡**

| 特性 | MAC阵列 | 向量处理单元 |
|------|---------|----------------|
| 计算模式 | 固定的乘加运算 | 灵活的算术/逻辑运算 |
| 数据访问 | 规则的矩阵访问 | 灵活的向量访问 |
| 功能单元 | 大量简单MAC | 少量复杂ALU |
| 面积效率 | 高（90%用于计算） | 中（需要多种功能） |

#### 现代VPU的关键创新

**1. 可重构计算管线：**
Intel的Nervana NNP包含了可重构的向量单元，可以根据不同的激活函数动态改变计算管线。例如，计算ReLU时只需要比较器，而计算GELU时需要完整的超越函数单元。

**2. 查找表（LUT）加速：**
对于复杂的激活函数（如Sigmoid、Tanh），许多NPU使用查找表+插值的方法。例如，将函数域分成256段，存储每段的起点值和斜率，通过线性插值获得结果。这种方法可以将计算延迟从20+周期降低到2-3周期。

**3. 融合操作（Fused Operations）：**
现代Transformer大量使用LayerNorm + Activation的组合。高级VPU可以将这些操作融合在一个流水线中完成，避免中间结果写回内存。NVIDIA的Hopper架构可以将整个“Linear-LayerNorm-GeLU”序列融合执行。

**向量处理单元(VPU)设计：**

VPU的SIMD架构特点：
- 16路SIMD通道并行处理
- 每个通道包含：
  - ALU单元：支持加、减、乘、最大/最小值
  - 激活函数单元：支持ReLU、Sigmoid、Tanh
- 操作码解码选择不同功能
- 单周期执行，流水线化设计

### 4.3.2 特殊功能单元

| 功能单元 | 操作 | 实现方式 | 硬件成本 |
|---------|------|---------|----------|
| ReLU单元 | max(0, x) | 比较器+选择器 | 极低 |
| 池化单元 | max/avg pooling | 比较树/加法树 | 低 |
| LUT单元 | sigmoid/tanh | 查找表+插值 | 中等 |
| 归一化单元 | batch/layer norm | 乘法器+移位器 | 高 |

#### 特殊功能单元的设计权衡

**1. 激活函数单元的进化**

从简单到复杂的激活函数硬件实现：

| 激活函数 | 计算复杂度 | 硬件实现 | 精度要求 | 现代NPU支持 |
|----------|----------|----------|----------|-------------|
| ReLU | O(1) | 比较器 | 无损 | ✓ 全部 |
| Leaky ReLU | O(1) | 比较+选择 | 无损 | ✓ 全部 |
| Sigmoid | O(log N) | LUT+插值 | 16-bit | ✓ 部分 |
| Tanh | O(log N) | LUT+插值 | 16-bit | ✓ 部分 |
| GELU | O(N) | 多项式近似 | 16-bit | ✓ 高端 |
| SiLU/Swish | O(N) | Sigmoid×x | 16-bit | ✓ 高端 |

**2. 查找表（LUT）优化技术**

现代NPU使用分段线性插值来平衡精度和存储：

```
分段策略：
- 线性区域：粗粒度分段（如16段）
- 非线性区域：细粒度分段（如128段）
- 饱和区域：直接返回常数

存储需求：
- 均匀分段：256×2×16-bit = 1KB
- 自适应分段：512×3×16-bit = 3KB
```

**3. 归一化单元的挑战**

LayerNorm和BatchNorm的硬件实现难点：
- 需要计算均值和方差（两遍扫描）
- 除法和平方根运算昂贵
- 数值稳定性要求高

解决方案：
- **在线算法：** Welford算法单遍计算均值和方差
- **快速倒数平方根：** Newton-Raphson迭代
- **融合计算：** 与前后层操作合并

#### 现代向量单元的创新设计

**1. 多函数融合单元（Multi-Function Fusion Unit）**

NVIDIA H100的Transformer Engine可以在一个单元中完成：
- MatMul → Add Bias → LayerNorm → Activation
- 减少中间结果的存储
- 提高5倍能效

**2. 可编程向量单元（Programmable Vector Unit）**

Intel Habana Gaudi的TPC（Tensor Processor Core）：
- VLIW架构，支持自定义指令
- 用户可编程新的激活函数
- 适应未来的算法创新

**3. 自适应精度单元（Adaptive Precision Unit）**

Qualcomm Cloud AI 100的动态精度：
- 根据输入范围自动选择精度
- INT8/FP16/FP32自动切换
- 保持精度同时最大化性能

### 4.3.3 TPU Softmax 实现深度解析

Softmax是神经网络中的关键操作，尤其在注意力机制中。TPU采用了软硬件协同的优化策略，实现了极高效的Softmax计算。

#### TPU Softmax 实现架构

**1. 算法优化：数值稳定的 Log-Sum-Exp**

**数值稳定的Softmax算法：**

标准公式：softmax(xᵢ) = exp(xᵢ) / Σ exp(xⱼ)

TPU优化版本：softmax(xᵢ) = exp(xᵢ - max(x)) / Σ exp(xⱼ - max(x))

计算步骤：
1. 并行规约找最大值
2. 广播减法防止溢出
3. 硬件加速指数运算
4. 并行规约求和
5. 逐元素除法（转为乘法）

**2. 硬件实现：VPU（向量处理单元）**

| 计算步骤 | 硬件单元 | 优化技术 | 执行时间 |
|---------|---------|---------|----------|
| 寻找最大值 | VPU并行规约单元 | 树状比较器网络 | O(log N) |
| 广播减法 | VPU SIMD单元 | 标量广播+向量减法 | O(1) |
| 指数运算 | 专用SFU（特殊功能单元） | 硬件LUT+多项式插值 | O(1) |
| 求和操作 | VPU并行规约单元 | 树状加法器网络 | O(log N) |
| 逐元素除法 | VPU乘法单元 | 倒数转乘法 | O(1) |

**3. 关键硬件优化：SFU（特殊功能单元）**

**TPU SFU指数运算实现：**

硬件加速指数运算的关键技术：
1. 范围检测和饱和处理：避免溢出
2. 分解x = n×ln(2) + r：利用exp(x) = 2ⁿ × exp(r)
3. LUT查找表：256项精度表格
4. 二次多项式插值：提高精度
5. 移位重构：利用浮点数特性

**4. 内存优化：算子融合**
- **数据局部性：**Softmax输入通常来自前一层的矩阵乘法（MXU输出），直接流向VPU
- **片上计算：**整个Softmax过程在片上SRAM完成，避免HBM访问
- **流水线优化：**max、exp、sum等操作流水线化，隐藏延迟
- **批处理：**多个序列的Softmax可以共享规约树硬件

**5. XLA编译器优化**
**XLA编译器的Softmax融合优化：**

用户代码：`y = tf.nn.softmax(logits, axis=-1)`

TPU指令序列：
- VPU_MAX: 并行找最大值
- VPU_SUB: 广播减法
- SFU_EXP: 硬件指数运算
- VPU_SUM: 并行规约求和
- VPU_RECIP: 倒数计算
- VPU_MUL: 归一化

所有操作在片上完成，避免HBM访问。

**性能对比：**

| 处理器 | Softmax实现 | 1M元素耗时 | 能效比 |
|--------|-------------|-------------|----------|
| CPU (AVX-512) | 软件循环+数学库 | ~10ms | 基准 |
| GPU (CUDA) | Warp级规约+共享内存 | ~0.5ms | 10x |
| TPU v4 | VPU硬件+SFU+融合 | ~0.05ms | 100x |

## <a name="44"></a>4.4 特殊计算单元

### 4.4.1 Tensor Core设计

Tensor Core是一种执行小矩阵乘法的专用单元，提供更高的计算密度。

Tensor Core代表了计算单元设计的范式转变：从标量运算到矩阵运算。这就像从单兵作战升级到集团军作战——虽然单个士兵的能力没有显著提升，但协同作战的效率呈指数级增长。

#### Tensor Core的革命性创新

**传统MAC vs Tensor Core的根本区别**
- **传统MAC：** C += A × B（标量运算）
- **Tensor Core：** D = A×B + C（矩阵运算，如4×4×4）
- **计算密度提升：** 单个Tensor Core完成64次乘法和48次加法，而占用的面积仅为64个独立MAC的约40%
- **能效提升：** 批量操作减少了控制开销，能效提升2-3倍

#### 实现挑战与解决方案

**挑战1：数据对齐和填充**
Tensor Core要求输入矩阵的维度是特定倍数（如4、8、16）。对于不规则尺寸，需要填充（padding），这会浪费计算资源。解决方案包括：
- 动态尺寸支持：NVIDIA Ampere引入了更灵活的维度支持
- 稀疏加速：利用结构化稀疏跳过填充的零值计算

**挑战2：数据供给带宽**
一个4×4×4的Tensor Core每周期需要32个输入数据。传统的寄存器文件设计无法提供如此高的带宽。创新解决方案：
- 分布式寄存器文件：每个Tensor Core配备专用的局部寄存器
- 寄存器缓存层次：引入L0.5级寄存器缓存
- 数据预取和双缓冲：隐藏数据加载延迟

#### 未来趋势：从数字到模拟

下一代计算单元正在探索模拟计算和存内计算：

**1. 模拟矩阵乘法器（Analog Matrix Multiplier）：**
利用欧姆定律和基尔霍夫定律，在模拟域完成矩阵运算。Mythic和Syntiant等公司已经实现了商用产品，能效提升100倍以上。

**2. 存内计算（Compute-in-Memory）：**
将计算单元直接集成到存储阵列中，彻底消除数据移动。三星的HBM-PIM和SK海力士的AiM都是这个方向的先驱。

**3. 光子计算（Photonic Computing）：**
利用光的干涉实现矩阵乘法，理论上可以达到光速计算。Lightmatter公司的Envise芯片已经展示了这种可能性。

**4×4×4 Tensor Core实现：**

Tensor Core的矩阵运算：D = A×B + C

关键设计特点：
- 16个点积单元并行计算
- 每个点积单元包含4个乘法器
- 加法树实现快速累加
- 总计64次乘法和48次加法在一个周期内完成
- 面积效率是64个独立MAC的40%

### 4.4.2 稀疏计算支持

支持结构化稀疏（如2:4稀疏）可以显著提升有效计算吞吐量。

**2:4结构化稀疏MAC单元：**

稀疏计算优化：
- 每4个权重中只有2个非零值
- 使用索引选择器替代多路选择器
- 减少50%的乘法运算
- 适合经过稀疏优化的模型
- NVIDIA Ampere首次引入此技术

#### 稀疏计算的数学基础与硬件实现

**稀疏性的类型与应用**

| 稀疏类型 | 稀疏模式 | 压缩率 | 硬件复杂度 | 应用场景 |
|---------|---------|--------|----------|----------|
| 非结构化 | 随机分布 | 90%+ | 高 | 科学计算 |
| 结构化 | 2:4, 4:8 | 50% | 中 | 深度学习 |
| 块稀疏 | 16×16块 | 75%+ | 低 | 大模型 |
| 通道稀疏 | 整通道剪枝 | 60%+ | 最低 | 边缘部署 |

**结构化稀疏的优势**
1. **可预测的内存访问：** 固定模式便于硬件优化
2. **简化索引编码：** 2-bit索引足以表示2:4模式
3. **保持对齐：** 不破坏SIMD并行性

**硬件实现的关键技术**

```
2:4稀疏计算流程：
1. 压缩存储：权重(2值) + 索引(2-bit)
2. 索引解码：选择对应的激活值
3. 并行计算：2个MAC单元代替4个
4. 结果累加：与密集计算相同
```

#### 稀疏加速的未来方向

**1. 动态稀疏（Dynamic Sparsity）**
- 根据输入动态跳过零值
- 需要复杂的零值检测和调度
- 适合激活值稀疏

**2. 级联稀疏（Cascading Sparsity）**
- 利用权重和激活值的双重稀疏
- 理论上可达4×加速
- 需要复杂的运行时支持

**3. 学习型稀疏（Learned Sparsity）**
- 训练时学习最优稀疏模式
- 与硬件约束协同设计
- 例如：NVIDIA的结构化稀疏训练

### 4.4.3 新兴计算范式

#### 存内计算（Processing-in-Memory）

**核心思想：** 将计算移到数据所在的地方，而不是将数据移到计算单元。

| 技术方案 | 实现方式 | 优势 | 挑战 |
|---------|---------|------|------|
| DRAM-PIM | HBM内集成ALU | 高带宽 | 工艺复杂 |
| SRAM-CIM | 6T SRAM改造 | 低延迟 | 容量有限 |
| ReRAM-CIM | 电阻式存储 | 非易失 | 精度受限 |
| Flash-CIM | 3D NAND计算 | 大容量 | 速度较慢 |

#### 光子计算（Photonic Computing）

**光学MAC单元原理：**
- 乘法：光强度调制
- 加法：光束合束
- 累加：光电探测器积分

**优势与挑战：**
- ✓ 光速计算，零功耗传输
- ✓ 天然并行，无电磁干扰
- ✗ 难以实现非线性操作
- ✗ 光电转换开销

## 4.5 练习题

### 理论题

**1. MAC阵列设计分析**

对于一个128×128的MAC阵列，执行256×512×128的矩阵乘法：
a) 计算Weight Stationary模式下的数据复用率
b) 分析所需的片上存储容量
c) 估算带宽需求（假设 INT8数据类型）

<details>
<summary>点击查看答案</summary>

a) 数据复用率分析：
- 权重复用次数 = 256/128 = 2次
- 每个权重在加载后被2个不同的输入批次使用
- 总体数据复用率 = 2

b) 片上存储需求：
- 权重存储：128×128 = 16K 个INT8 = 16KB
- 输入缓冲：128×2 = 256 个INT8 = 256B（双缓冲）
- 输出缓冲：128×2 = 256 个INT32 = 1KB（双缓冲）
- 总计约：17.25KB

c) 带宽需求：
- 输入带宽：128 bytes/cycle
- 输出带宽：128×4 = 512 bytes/cycle（INT32）
- 假设1GHz频率，总带宽 = 640 GB/s
</details>

**2. 脉动阵列性能优化**

设计一个64×64的脉动阵列用于Transformer模型的矩阵乘法。如果典型的矩阵尺寸为：
- Q, K, V矩阵：512×64
- 注意力矩阵：512×512

请分析：
a) 哪种数据流模式（WS/OS）更适合？
b) 如何分块以最大化利用率？
c) 估算完成一个注意力层的周期数

<details>
<summary>点击查看答案</summary>

a) 数据流模式选择：
- Q×K^T计算：Output Stationary更优
  - 输出尺寸512×512较大，避免频繁移动部分和
- Attention×V计算：Weight Stationary更优
  - V矩阵较小（512×64），可以完全驻留

b) 分块策略：
- Q, K, V分块：512×64 = 8×1块
- 注意力矩阵分块：512×512 = 8×8 = 64块
- 每块完美匹配64×64阵列，利用率100%

c) 周期数估算：
- Q×K^T: 8×8×1 = 64块 × 64周期 = 4096周期
- Softmax: 512行 × 10周期 = 5120周期
- Attention×V: 8×1×8 = 64块 × 64周期 = 4096周期
- 总计约：13,312周期
</details>

**3. 向量处理单元设计**

设计一个支持GELU激活函数的向量处理单元。GELU(x) = x × Φ(x)，其中Φ是标准正态分布的CDF。

请回答：
a) 如何设计LUT以平衡精度和存储？
b) 估算所需的硬件资源
c) 与ReLU相比，延迟增加多少？

<details>
<summary>点击查看答案</summary>

a) LUT设计：
- 输入范围：[-4, 4]（覆盖99.99%的值）
- 分段策略：
  - [-4, -2]: 32段（梯度较小）
  - [-2, 2]: 192段（梯度较大）
  - [2, 4]: 32段（梯度较小）
- 每段存储：起点值(16-bit) + 斜率(16-bit)
- 总存储：256 × 32-bit = 1KB

b) 硬件资源：
- LUT存储：1KB SRAM
- 地址计算：8-bit比较器 + 编码器
- 插值单元：1个16-bit乘法器 + 1个加法器
- 最终乘法：1个16-bit乘法器（x × Φ(x)）

c) 延迟对比：
- ReLU: 1周期（简单比较）
- GELU: 4周期
  - 地址计算：1周期
  - LUT读取：1周期
  - 插值计算：1周期
  - 最终乘法：1周期
</details>

### 设计题

**4. Tensor Core优化**

为一个边缘AI芯片设计4×4×4的Tensor Core。要求：
- 支持INT8和FP16混合精度
- 面积限制在1mm²以内（7nm工艺）
- 功耗不超过2W

请提供：
a) 详细的微架构设计
b) 面积和功耗估算
c) 与传统MAC阵列的对比

<details>
<summary>点击查看答案</summary>

a) 微架构设计：
```
Tensor Core组成：
- 16个点积单元（DPU）
- 每个DPU包含：
  - 4个INT8乘法器（可组合2个FP16）
  - 3级加法树
  - 累加器（INT32/FP32）
- 共享资源：
  - 3个输入缓冲（4×4×8-bit×3）
  - 1个输出缓冲（4×4×32-bit）
  - 控制逻辑
```

b) PPA估算：
- 面积：
  - DPU阵列：0.6mm²
  - 缓冲器：0.2mm²
  - 控制逻辑：0.1mm²
  - 总计：0.9mm²
- 功耗（@1GHz）：
  - 动态功耗：1.5W
  - 静态功耗：0.3W
  - 总计：1.8W

c) 性能对比：
- Tensor Core: 128 INT8 OPS/cycle
- 16个MAC阵列: 32 INT8 OPS/cycle
- 性能提升：4×
- 能效提升：2.5×
</details>

**5. 稀疏计算加速器**

设计一个支持2:4结构化稀疏的加速器。要求实现：
- 与密集计算相比，提供2×的有效吞吐率
- 支持16×16的矩阵块操作
- 最小化索引存储开销

<details>
<summary>点击查看答案</summary>

设计方案：

1. **索引编码方案：**
   - 每4个元素使用2-bit索引
   - 16种可能的非零位置组合
   - 每个16×16块需要16×4 = 64个索引
   - 索引存储：128 bits/块

2. **计算单元设计：**
   ```
   稀疏PE结构：
   - 2个MAC单元（原有4个）
   - 4:1输入选择器
   - 索引解码器
   - 累加器
   ```

3. **性能分析：**
   - 计算减少：50%
   - 存储减少：~45%（考虑索引开销）
   - 有效吞吐率：1.8-2.0×
</details>

### 编程题

**6. 实现一个简单的4×4脉动阵列仿真器**

使用Python实现一个Weight Stationary脉动阵列仿真器，并验证矩阵乘法的正确性。

<details>
<summary>点击查看参考代码</summary>

```python
import numpy as np

class SystolicArray:
    def __init__(self, size=4):
        self.size = size
        self.weights = np.zeros((size, size))
        self.pe_array = np.zeros((size, size))
        
    def load_weights(self, W):
        """Load weights into the systolic array"""
        assert W.shape == (self.size, self.size)
        self.weights = W.copy()
        
    def compute(self, A, B):
        """Compute C = A @ B using systolic array"""
        assert A.shape[1] == B.shape[0] == self.size
        M, K = A.shape
        K, N = B.shape
        
        # Load weights from B
        self.load_weights(B)
        
        # Initialize output
        C = np.zeros((M, N))
        
        # Systolic computation with data skewing
        for t in range(M + N + K - 2):
            # Input activations (with skewing)
            for i in range(self.size):
                for j in range(self.size):
                    if t - j >= 0 and t - j < M and i < K:
                        act = A[t - j, i]
                    else:
                        act = 0
                    
                    # MAC operation
                    if j == 0:
                        self.pe_array[i, j] = act * self.weights[i, j]
                    else:
                        self.pe_array[i, j] = self.pe_array[i, j-1] + \
                                              act * self.weights[i, j]
            
            # Collect outputs
            for j in range(self.size):
                row = t - j - K + 1
                if 0 <= row < M:
                    C[row, j] = self.pe_array[K-1, j]
                    
        return C

# Test
if __name__ == "__main__":
    sa = SystolicArray(4)
    A = np.random.randint(0, 10, (4, 4))
    B = np.random.randint(0, 10, (4, 4))
    
    C_systolic = sa.compute(A, B)
    C_reference = A @ B
    
    print("Systolic result:")
    print(C_systolic)
    print("\nReference result:")
    print(C_reference)
    print("\nMatch:", np.allclose(C_systolic, C_reference))
```
</details>

**7. 向量处理单元的Softmax实现**

用Verilog实现一个简化的Softmax计算单元，支持8个元素的并行处理。

<details>
<summary>点击查看参考代码框架</summary>

```verilog
module softmax_unit #(
    parameter DATA_WIDTH = 16,
    parameter NUM_ELEMENTS = 8
)(
    input clk,
    input rst_n,
    input valid_in,
    input [DATA_WIDTH*NUM_ELEMENTS-1:0] data_in,
    output reg valid_out,
    output reg [DATA_WIDTH*NUM_ELEMENTS-1:0] data_out
);

    // State machine states
    localparam IDLE = 0, FIND_MAX = 1, SUBTRACT = 2, 
               EXP = 3, SUM = 4, DIVIDE = 5;
    
    reg [2:0] state;
    reg [DATA_WIDTH-1:0] max_val;
    reg [DATA_WIDTH-1:0] sum_exp;
    
    // Pipeline registers
    reg [DATA_WIDTH*NUM_ELEMENTS-1:0] data_reg;
    reg [DATA_WIDTH*NUM_ELEMENTS-1:0] exp_reg;
    
    // Simplified implementation outline
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            valid_out <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (valid_in) begin
                        data_reg <= data_in;
                        state <= FIND_MAX;
                    end
                end
                
                FIND_MAX: begin
                    // Find maximum using parallel comparison tree
                    // ... (implementation details)
                    state <= SUBTRACT;
                end
                
                SUBTRACT: begin
                    // Subtract max from all elements
                    // ... (implementation details)
                    state <= EXP;
                end
                
                EXP: begin
                    // Compute exponentials (using LUT)
                    // ... (implementation details)
                    state <= SUM;
                end
                
                SUM: begin
                    // Sum all exponentials
                    // ... (implementation details)
                    state <= DIVIDE;
                end
                
                DIVIDE: begin
                    // Divide each exp by sum
                    // ... (implementation details)
                    valid_out <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end
    
endmodule
```
</details>

## 本章小结

本章深入探讨了NPU计算核心的设计，从基础的MAC单元到复杂的脉动阵列，再到专门的向量处理单元和特殊计算单元。

**关键要点：**
1. MAC阵列是NPU的核心，通过大规模并行化实现高吞吐率
2. 脉动阵列通过规则的数据流动实现高效的数据复用
3. 向量处理单元补充了非线性操作的支持
4. 现代NPU趋向于支持多精度、稀疏计算和新型计算范式

下一章我们将讨论如何为这些计算核心设计高效的存储系统。