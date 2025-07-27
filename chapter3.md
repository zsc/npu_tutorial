# 第3章：NPU系统架构

在前两章中，我们了解了NPU的基本概念和神经网络计算的基础知识。从本章开始，我们将深入探讨NPU的架构设计。在众多NPU架构中，**脉动阵列（Systolic Array）** 凭借其规则的结构、高效的数据复用和优秀的可扩展性，成为了现代NPU设计的一种流行选择。Google TPU、Tesla FSD芯片等知名NPU都采用了脉动阵列作为其计算核心。

因此，在接下来的几章中，我们将以脉动阵列架构为核心展开讨论。本章将介绍NPU的整体系统架构，第4章将深入脉动阵列的计算核心设计，第5章将探讨如何为脉动阵列设计高效的存储系统。需要强调的是，虽然我们以脉动阵列为例，但**许多设计原则和优化思想同样适用于其他架构**，如Groq的数据流架构（Dataflow Architecture）。两种架构都追求规则的数据流动、高效的并行计算和最小化的内存访问开销，只是在具体实现方式上有所不同。通过深入理解脉动阵列，我们可以掌握NPU设计的核心思想——如何通过规则的数据流动模式实现高效的并行计算。

## 3.1 整体架构设计

### 3.1.1 NPU系统组成

现代NPU系统通常包含以下核心组件：

```
NPU系统架构层次：
┌─────────────────────────────────────────┐
│          Host Interface (PCIe/AXI)       │
├─────────────────────────────────────────┤
│         Command Processor & Scheduler    │
├─────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ Compute │  │ Memory  │  │  DMA    │ │
│  │ Cluster │  │ System  │  │ Engine  │ │
│  └─────────┘  └─────────┘  └─────────┘ │
├─────────────────────────────────────────┤
│         On-chip Interconnect (NoC)      │
├─────────────────────────────────────────┤
│         External Memory Interface        │
└─────────────────────────────────────────┘
```

NPU的整体架构不仅仅是计算、存储和控制单元的简单堆砌，它更像一个为数据流精心设计的"智能工厂"。其核心设计哲学是**最大化数据复用**并**最小化数据搬运**，因为在现代NPU中，数据移动的能耗和时间开销远超计算本身。

#### 核心组件的协同工作流（以一次卷积计算为例）

一个典型的NPU工作流程可以类比为一座高效的**汽车装配厂**：

1. **任务下发 (Instruction)：** CPU（工厂总指挥）向NPU的**主控单元 (Control Unit)** 下达指令："开始生产一批特定型号的汽车（执行一个卷积层计算）"。主控单元是"车间主任"，负责解析蓝图（神经网络指令），协调整个生产流程。

2. **原料入库 (Data Fetch)：** 主控单元命令 **DMA（Direct Memory Access）控制器**——工厂的"智能物流系统"——从外部DRAM（"中央仓库"）中提取所需的**输入特征图 (Input Feature Maps)** 和**权重 (Weights)**。DMA将这些"原材料"高效地运送到位于NPU内部的**片上缓冲 (On-chip Buffer)**，即"车间暂存区"。

   > **设计洞察 (Why DMA?)：** 为什么不让CPU亲自搬运数据？因为CPU是高薪聘请的"总工程师"，让他处理这种重复性的搬运工作是巨大的资源浪费。DMA这个自动化物流系统可以在计算单元工作的同时，并行地准备下一批数据，完美隐藏了数据传输的延迟，确保生产线"永不停工"。

3. **车间生产 (Computation)：** 数据准备就绪后，主控单元激活**计算单元阵列 (PE Array)**——"装配线上的机器人矩阵"。成百上千的PE（Processing Element）就像机器人手臂，每个PE从其旁边的**本地存储 (Local Storage)**（"零件盒"）中取出数据，执行大量的**乘加 (MAC) 运算**。

   > **协同方式 (How they interact?)：** 数据在PE阵列中以一种称为**"脉动阵列 (Systolic Array)"** 的模式高效流动。数据像心跳的脉搏一样，在一个时钟周期内从一个PE传递到下一个PE，并在此过程中完成计算。这种方式最大化了每个数据片段的复用次数，例如，一个权重数据可以与一行输入数据依次进行计算，而无需重复从内存中读取。

4. **成品出库 (Result Write-back)：** 计算完成后，**部分和 (Partial Sums)** 或最终的**输出特征图 (Output Feature Maps)** 被写回到片上缓冲。当一个计算任务块（Tile）完成后，DMA再次启动，将这些"半成品"或"成品"运回DRAM"中央仓库"，或直接送往下一个"生产车间"（下一个神经网络层）。

#### 真实世界案例

- **Google TPU (Tensor Processing Unit):** 其核心就是巨大的脉动阵列。在TPU v1中，拥有一个256x256的MAC阵列，能够在一个时钟周期内完成65,536次运算。它的设计哲学就是"为卷积而生"，通过巨大的片上内存（例如，TPU v2/v3拥有数十MB的HBM）和高效的数据流，确保这个庞大的计算阵列始终"吃饱喝足"。

- **NVIDIA A100 GPU:** 虽然是GPU，但其Tensor Core就是专为AI设计的NPU模块。它采用更灵活的架构，支持不同精度（TF32, FP16, INT8）的计算，并引入了**"结构化稀疏 (Structured Sparsity)"** 支持，这是一种软硬件协同设计，能在不牺牲太多精度的情况下，让计算性能翻倍。

> **常见陷阱与规避：**
> - **陷阱：唯"峰值算力 (Peak TOPs)"论。** 很多NPU宣传极高的理论算力，但如果内存带宽跟不上，计算单元大部分时间都在"挨饿"，实际利用率（Utilization）可能不足20%。
> - **规避：** 评估NPU时，应关注**算力/内存带宽比 (Compute/Memory Ratio)**。一个健康的比例才能确保高效率。对于设计者而言，必须通过精巧的数据流（Dataflow）和缓存（Tiling）策略来弥合计算与访存之间的鸿沟。

### 3.1.2 设计考虑因素

| 设计维度 | 关键指标 | 架构影响 | 优化方向 |
|---------|---------|---------|---------|
| 计算密度 | TOPS/mm² | MAC阵列规模 | 工艺优化、3D集成 |
| 能效比 | TOPS/W | 电源管理设计 | 低功耗设计、DVFS |
| 内存带宽 | GB/s | 存储层次结构 | HBM集成、压缩技术 |
| 灵活性 | 支持的算子种类 | 指令集设计 | 可编程性vs专用化 |
| 扩展性 | 多芯片互联 | NoC架构 | Chiplet、高速互联 |

### 3.1.3 架构演进趋势

NPU架构的演进反映了AI算法和应用需求的变化：

1. **第一代（2015-2017）：** 专注于CNN加速
   - 代表：Google TPU v1
   - 特点：固定功能、高效但灵活性有限

2. **第二代（2018-2020）：** 支持更多网络类型
   - 代表：华为Ascend 310/910
   - 特点：增加了可编程性、支持多种精度

3. **第三代（2021-2023）：** Transformer优化
   - 代表：NVIDIA H100、Google TPU v4
   - 特点：专门的注意力机制加速、超大片上存储

4. **第四代（2024-）：** 生成式AI时代
   - 趋势：支持超长序列、动态稀疏性、推测解码
   - 挑战：内存墙、功耗墙、算法快速迭代

## 3.2 计算单元设计

### 3.2.1 处理单元(PE)架构

处理单元（PE）是NPU的基本计算单元，其设计直接影响整体性能：

```verilog
// 基本PE单元结构
module ProcessingElement #(
    parameter DATA_WIDTH = 16,
    parameter ACC_WIDTH = 32
)(
    input clk, rst_n,
    input [DATA_WIDTH-1:0] a_in,    // 输入激活
    input [DATA_WIDTH-1:0] w_in,    // 权重
    input [ACC_WIDTH-1:0] psum_in,  // 部分和输入
    
    output reg [DATA_WIDTH-1:0] a_out,   // 激活传递
    output reg [DATA_WIDTH-1:0] w_out,   // 权重传递
    output reg [ACC_WIDTH-1:0] psum_out  // 部分和输出
);
    // MAC运算
    wire [2*DATA_WIDTH-1:0] mult_result;
    wire [ACC_WIDTH-1:0] add_result;
    
    assign mult_result = a_in * w_in;
    assign add_result = psum_in + mult_result;
    
    // 寄存器传递
    always @(posedge clk) begin
        if (!rst_n) begin
            a_out <= 0;
            w_out <= 0;
            psum_out <= 0;
        end else begin
            a_out <= a_in;      // 向右传递激活
            w_out <= w_in;      // 向下传递权重
            psum_out <= add_result;  // 输出部分和
        end
    end
endmodule
```

### 3.2.2 MAC阵列组织

MAC阵列的组织方式决定了数据流模式和硬件利用率：

```verilog
// 脉动阵列顶层模块
module SystolicArray #(
    parameter ARRAY_SIZE = 16,
    parameter DATA_WIDTH = 16,
    parameter ACC_WIDTH = 32
)(
    input clk, rst_n,
    input enable,
    
    // 输入接口
    input [DATA_WIDTH-1:0] act_in [0:ARRAY_SIZE-1],
    input [DATA_WIDTH-1:0] weight_in [0:ARRAY_SIZE-1],
    
    // 输出接口
    output [ACC_WIDTH-1:0] result_out [0:ARRAY_SIZE-1]
);
    // PE阵列实例化
    wire [DATA_WIDTH-1:0] act_h [0:ARRAY_SIZE][0:ARRAY_SIZE];
    wire [DATA_WIDTH-1:0] weight_v [0:ARRAY_SIZE][0:ARRAY_SIZE];
    wire [ACC_WIDTH-1:0] psum [0:ARRAY_SIZE][0:ARRAY_SIZE];
    
    genvar i, j;
    generate
        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin : row
            for (j = 0; j < ARRAY_SIZE; j = j + 1) begin : col
                ProcessingElement pe_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .a_in(act_h[i][j]),
                    .w_in(weight_v[i][j]),
                    .psum_in(i == 0 ? 32'h0 : psum[i-1][j]),
                    .a_out(act_h[i][j+1]),
                    .w_out(weight_v[i+1][j]),
                    .psum_out(psum[i][j])
                );
            end
        end
    endgenerate
    
    // 连接输入
    for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
        assign act_h[i][0] = act_in[i];
        assign weight_v[0][i] = weight_in[i];
    end
    
    // 连接输出
    for (j = 0; j < ARRAY_SIZE; j = j + 1) begin
        assign result_out[j] = psum[ARRAY_SIZE-1][j];
    end
endmodule
```

### 3.2.3 数据流模式

脉动阵列支持多种数据流模式，每种都有其优缺点：

#### 1. 权重固定（Weight Stationary, WS）

```
特点：权重预加载到PE中，激活和部分和流动
优点：权重访问能耗最低
缺点：需要较大的PE内部存储
适用：权重复用率高的场景（大batch size）
```

#### 2. 输出固定（Output Stationary, OS）

```
特点：每个PE负责计算一个输出元素
优点：部分和不需要移动，减少累加器位宽
缺点：权重和激活都需要广播
适用：输出数据量较小的场景
```

#### 3. 行固定（Row Stationary, RS）

```
特点：将矩阵运算的一行映射到一行PE
优点：平衡了各种数据的复用
缺点：控制逻辑较复杂
适用：通用场景，如Eyeriss采用此方式
```

### 3.2.4 Transformer加速支持

随着Transformer模型的流行，现代NPU需要支持注意力机制：

```verilog
// 注意力计算单元框架
module AttentionUnit #(
    parameter SEQ_LEN = 512,
    parameter HEAD_DIM = 64,
    parameter DATA_WIDTH = 16
)(
    input clk, rst_n,
    input [DATA_WIDTH-1:0] Q [0:SEQ_LEN-1][0:HEAD_DIM-1],
    input [DATA_WIDTH-1:0] K [0:SEQ_LEN-1][0:HEAD_DIM-1],
    input [DATA_WIDTH-1:0] V [0:SEQ_LEN-1][0:HEAD_DIM-1],
    
    output [DATA_WIDTH-1:0] output [0:SEQ_LEN-1][0:HEAD_DIM-1]
);
    // 1. 计算QK^T
    wire [DATA_WIDTH-1:0] scores [0:SEQ_LEN-1][0:SEQ_LEN-1];
    
    // 2. Softmax（简化版）
    wire [DATA_WIDTH-1:0] attn_weights [0:SEQ_LEN-1][0:SEQ_LEN-1];
    
    // 3. 注意力权重与V相乘
    // 实现细节...
endmodule
```

## 3.3 存储层次结构

### 3.3.1 存储层次设计原则

NPU的存储层次设计遵循以下原则：

1. **局部性原理：** 充分利用时间和空间局部性
2. **带宽匹配：** 各级存储带宽与计算需求匹配
3. **能耗优化：** 数据尽量在低层次存储间移动
4. **容量权衡：** 平衡芯片面积和性能需求

```
存储层次结构：
┌─────────────────────────────┐
│   外部DRAM (8-32GB)         │ ← 大容量，高延迟
├─────────────────────────────┤
│   L2 Cache (4-16MB)         │ ← 中等容量，中等延迟
├─────────────────────────────┤
│   L1 Buffer (128-512KB)     │ ← 小容量，低延迟
├─────────────────────────────┤
│   PE Local Reg (1-4KB)      │ ← 极小容量，零延迟
└─────────────────────────────┘
```

### 3.3.2 片上缓冲设计

片上缓冲是NPU性能的关键：

```verilog
// 可配置的片上缓冲模块
module OnChipBuffer #(
    parameter DEPTH = 1024,
    parameter WIDTH = 256,
    parameter BANKS = 8
)(
    input clk, rst_n,
    
    // 写接口
    input wr_en,
    input [$clog2(DEPTH)-1:0] wr_addr,
    input [WIDTH-1:0] wr_data,
    
    // 读接口（多个读口）
    input rd_en [0:BANKS-1],
    input [$clog2(DEPTH)-1:0] rd_addr [0:BANKS-1],
    output reg [WIDTH-1:0] rd_data [0:BANKS-1]
);
    // 分bank存储减少冲突
    reg [WIDTH/BANKS-1:0] mem [0:BANKS-1][0:DEPTH/BANKS-1];
    
    // 写逻辑
    always @(posedge clk) begin
        if (wr_en) begin
            integer bank_id = wr_addr % BANKS;
            integer bank_addr = wr_addr / BANKS;
            mem[bank_id][bank_addr] <= wr_data[bank_id*WIDTH/BANKS +: WIDTH/BANKS];
        end
    end
    
    // 读逻辑（支持并行读）
    genvar i;
    generate
        for (i = 0; i < BANKS; i = i + 1) begin
            always @(posedge clk) begin
                if (rd_en[i]) begin
                    integer bank_id = rd_addr[i] % BANKS;
                    integer bank_addr = rd_addr[i] / BANKS;
                    rd_data[i] <= {BANKS{mem[bank_id][bank_addr]}};
                end
            end
        end
    endgenerate
endmodule
```

### 3.3.3 数据复用策略

有效的数据复用是NPU高效率的关键：

#### 1. 输入复用（Input Reuse）

```python
# 输入特征图在不同输出通道间复用
for oc in range(output_channels):
    for ic in range(input_channels):
        for y in range(output_height):
            for x in range(output_width):
                # 输入[ic,y,x]被所有输出通道复用
                output[oc,y,x] += input[ic,y,x] * weight[oc,ic]
```

#### 2. 权重复用（Weight Reuse）

```python
# 权重在不同输入位置间复用
for n in range(batch_size):
    for y in range(output_height):
        for x in range(output_width):
            # 权重[oc,ic]被所有空间位置复用
            output[n,oc,y,x] = conv(input[n,:,y:y+k,x:x+k], weight[oc,ic])
```

#### 3. 部分和复用（Partial Sum Reuse）

```python
# 部分和在计算过程中累加
partial_sum = 0
for ic in range(input_channels):
    partial_sum += input[ic] * weight[ic]
output = activation(partial_sum + bias)
```

## 3.4 互连网络设计

### 3.4.1 片上网络(NoC)架构

片上网络负责连接NPU内部的各个组件：

```verilog
// 简化的2D Mesh NoC路由器
module NoCRouter #(
    parameter DATA_WIDTH = 256,
    parameter ADDR_WIDTH = 32,
    parameter X_COORD = 0,
    parameter Y_COORD = 0
)(
    input clk, rst_n,
    
    // 五个方向的输入输出（东南西北+本地）
    input [DATA_WIDTH-1:0] data_in_n, data_in_s, data_in_e, data_in_w, data_in_local,
    input valid_in_n, valid_in_s, valid_in_e, valid_in_w, valid_in_local,
    
    output reg [DATA_WIDTH-1:0] data_out_n, data_out_s, data_out_e, data_out_w, data_out_local,
    output reg valid_out_n, valid_out_s, valid_out_e, valid_out_w, valid_out_local
);
    // XY路由算法实现
    // 先沿X方向路由，再沿Y方向
    
    // 路由逻辑...
endmodule
```

### 3.4.2 数据通路设计

高效的数据通路设计需要考虑：

1. **带宽需求：** 满足峰值计算的数据供给
2. **延迟优化：** 减少数据传输的周期数
3. **拥塞避免：** 防止热点造成的性能下降
4. **功耗控制：** 最小化数据移动的能耗

### 3.4.3 全局同步机制

大规模NPU需要高效的同步机制：

```verilog
// 屏障同步模块
module BarrierSync #(
    parameter NUM_UNITS = 16
)(
    input clk, rst_n,
    input [NUM_UNITS-1:0] sync_req,    // 同步请求
    output reg [NUM_UNITS-1:0] sync_ack // 同步确认
);
    reg [NUM_UNITS-1:0] sync_status;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            sync_status <= 0;
            sync_ack <= 0;
        end else begin
            sync_status <= sync_status | sync_req;
            
            // 所有单元都到达屏障
            if (sync_status == {NUM_UNITS{1'b1}}) begin
                sync_ack <= {NUM_UNITS{1'b1}};
                sync_status <= 0;
            end else begin
                sync_ack <= 0;
            end
        end
    end
endmodule
```

## 习题集 3

### 基础练习题

**题目3.1：** 设计一个4×4的脉动阵列，支持8位整数运算。计算16×16的矩阵乘法需要多少个时钟周期？

<details>
<summary>💡 提示</summary>

考虑数据如何在脉动阵列中流动。对于M×K和K×N的矩阵乘法，使用P×P的脉动阵列，需要考虑：
1. 数据加载延迟
2. 计算时间
3. 结果输出延迟

</details>

<details>
<summary>参考答案</summary>

对于16×16和16×16的矩阵乘法，使用4×4脉动阵列：

1. **分块计算：**
   - M方向：16/4 = 4块
   - N方向：16/4 = 4块
   - K方向：16/4 = 4块

2. **时间计算：**
   - 数据加载延迟：4周期（第一个数据到达最后一个PE）
   - 每个块的计算：4周期
   - K方向4个块累加：4×4 = 16周期
   - 总计算时间：4（加载）+ 16（计算）+ 4（输出）= 24周期

3. **所有块：**
   - 总块数：4×4 = 16个输出块
   - 如果流水线处理：24 + (16-1)×4 = 84周期

</details>

**题目3.2：** 比较Weight Stationary和Output Stationary两种数据流模式的优缺点。给出适用场景。

<details>
<summary>💡 提示</summary>

从以下角度分析：
1. 数据移动量
2. 存储需求
3. 控制复杂度
4. 适用的网络层类型

</details>

<details>
<summary>参考答案</summary>

**Weight Stationary (WS)：**

优点：
- 权重只加载一次，能耗最低
- 适合大batch size，权重复用率高
- 控制逻辑相对简单

缺点：
- PE需要存储完整权重，面积开销大
- 输入和输出数据都需要流动
- 对小batch效率低

适用场景：
- 推理任务（batch size通常较大）
- 全连接层（权重复用率高）
- 功耗敏感的边缘设备

**Output Stationary (OS)：**

优点：
- 部分和不移动，减少了宽位宽累加器的能耗
- 每个PE专注一个输出，易于理解
- 支持灵活的并行策略

缺点：
- 权重和激活都需要广播，带宽需求高
- 难以充分利用数据复用
- 大输出时PE数量需求大

适用场景：
- 1×1卷积（没有空间复用）
- 深度可分离卷积的pointwise部分
- 输出通道数较少的层

</details>

**题目3.3：** 设计一个支持动态精度的PE单元，能够在INT8、INT16和FP16之间切换。

<details>
<summary>💡 提示</summary>

考虑：
1. 不同精度的乘法器如何复用
2. 累加器位宽如何适配
3. 数据对齐和格式转换

</details>

<details>
<summary>参考答案</summary>

```verilog
module FlexiblePE #(
    parameter MAX_WIDTH = 16
)(
    input clk, rst_n,
    input [1:0] precision_mode,  // 00: INT8, 01: INT16, 10: FP16
    input [MAX_WIDTH-1:0] a_in, w_in,
    input [2*MAX_WIDTH-1:0] psum_in,
    
    output reg [MAX_WIDTH-1:0] a_out, w_out,
    output reg [2*MAX_WIDTH-1:0] psum_out
);
    // 内部信号
    wire [2*MAX_WIDTH-1:0] mult_result;
    reg [2*MAX_WIDTH-1:0] mult_result_aligned;
    
    // 可配置乘法器
    FlexibleMultiplier mult_inst (
        .mode(precision_mode),
        .a(a_in),
        .b(w_in),
        .result(mult_result)
    );
    
    // 根据精度模式对齐结果
    always @(*) begin
        case (precision_mode)
            2'b00: begin  // INT8
                // 符号扩展INT8结果
                mult_result_aligned = {{16{mult_result[15]}}, mult_result[15:0]};
            end
            2'b01: begin  // INT16
                mult_result_aligned = mult_result;
            end
            2'b10: begin  // FP16
                mult_result_aligned = mult_result;  // FP16需要特殊处理
            end
            default: mult_result_aligned = 0;
        endcase
    end
    
    // 累加
    wire [2*MAX_WIDTH-1:0] add_result;
    FlexibleAdder add_inst (
        .mode(precision_mode),
        .a(psum_in),
        .b(mult_result_aligned),
        .result(add_result)
    );
    
    // 寄存器输出
    always @(posedge clk) begin
        if (!rst_n) begin
            a_out <= 0;
            w_out <= 0;
            psum_out <= 0;
        end else begin
            a_out <= a_in;
            w_out <= w_in;
            psum_out <= add_result;
        end
    end
endmodule
```

</details>

**题目3.4：** 计算一个NPU执行ResNet-50一个残差块所需的片上存储容量。假设使用16×16的MAC阵列。

<details>
<summary>💡 提示</summary>

ResNet-50的典型残差块包含：
1. 1×1卷积（降维）
2. 3×3卷积
3. 1×1卷积（升维）
需要考虑特征图和权重的存储。

</details>

<details>
<summary>参考答案</summary>

以ResNet-50的典型残差块为例（输入256通道，瓶颈64通道）：

**1. 第一个1×1卷积（256→64）：**
- 输入特征图：56×56×256 = 802KB (FP16)
- 权重：1×1×256×64 = 32KB
- 输出特征图：56×56×64 = 200KB

**2. 3×3卷积（64→64）：**
- 输入特征图：56×56×64 = 200KB
- 权重：3×3×64×64 = 36KB
- 输出特征图：56×56×64 = 200KB

**3. 第三个1×1卷积（64→256）：**
- 输入特征图：56×56×64 = 200KB
- 权重：1×1×64×256 = 32KB
- 输出特征图：56×56×256 = 802KB

**存储需求分析：**
- 最大同时存储：输入(802KB) + 输出(802KB) + 权重(36KB) = 1640KB
- 考虑双缓冲：1640KB × 2 = 3280KB
- 加上中间结果缓存：约4MB

**优化策略：**
- 使用分块（tiling）减少存储需求
- 层融合减少中间结果存储
- 权重压缩技术

</details>

**题目3.5：** 设计一个简单的NoC路由器，支持XY路由算法。

<details>
<summary>💡 提示</summary>

XY路由：先沿X方向路由到目标列，再沿Y方向路由到目标行。需要：
1. 解析目标地址
2. 比较当前坐标
3. 选择输出端口

</details>

<details>
<summary>参考答案</summary>

```verilog
module XYRouter #(
    parameter DATA_WIDTH = 32,
    parameter X_BITS = 4,
    parameter Y_BITS = 4,
    parameter X_COORD = 0,
    parameter Y_COORD = 0
)(
    input clk, rst_n,
    
    // 输入端口 (5个方向: N, S, E, W, Local)
    input [DATA_WIDTH-1:0] data_in [0:4],
    input [4:0] valid_in,
    input [X_BITS-1:0] dest_x [0:4],
    input [Y_BITS-1:0] dest_y [0:4],
    
    // 输出端口
    output reg [DATA_WIDTH-1:0] data_out [0:4],
    output reg [4:0] valid_out
);
    // 方向定义
    localparam NORTH = 0, SOUTH = 1, EAST = 2, WEST = 3, LOCAL = 4;
    
    // 路由决策
    reg [2:0] route_port [0:4];
    
    genvar i;
    generate
        for (i = 0; i < 5; i = i + 1) begin : routing
            always @(*) begin
                if (dest_x[i] == X_COORD && dest_y[i] == Y_COORD) begin
                    route_port[i] = LOCAL;
                end else if (dest_x[i] != X_COORD) begin
                    // 需要X方向路由
                    if (dest_x[i] > X_COORD)
                        route_port[i] = EAST;
                    else
                        route_port[i] = WEST;
                end else begin
                    // 需要Y方向路由
                    if (dest_y[i] > Y_COORD)
                        route_port[i] = SOUTH;
                    else
                        route_port[i] = NORTH;
                end
            end
        end
    endgenerate
    
    // 仲裁和交换逻辑
    reg [4:0] grant [0:4];  // grant[output][input]
    
    // 简化的轮询仲裁
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 0;
        end else begin
            // 对每个输出端口进行仲裁
            for (integer out_port = 0; out_port < 5; out_port++) begin
                valid_out[out_port] <= 0;
                for (integer in_port = 0; in_port < 5; in_port++) begin
                    if (valid_in[in_port] && route_port[in_port] == out_port && !valid_out[out_port]) begin
                        data_out[out_port] <= data_in[in_port];
                        valid_out[out_port] <= 1;
                    end
                end
            end
        end
    end
endmodule
```

</details>

### 高级练习题

**题目3.1：** 设计一个支持稀疏性的脉动阵列架构。要求能够跳过零值计算，提高有效吞吐量。

<details>
<summary>参考答案</summary>

```verilog
module SparseSystolicArray #(
    parameter ARRAY_SIZE = 8,
    parameter DATA_WIDTH = 16
)(
    input clk, rst_n,
    
    // 稠密输入接口
    input [DATA_WIDTH-1:0] act_values [0:ARRAY_SIZE-1],
    input [7:0] act_indices [0:ARRAY_SIZE-1],  // 列索引
    input act_valid [0:ARRAY_SIZE-1],
    
    // 稀疏权重接口 (CSR格式)
    input [DATA_WIDTH-1:0] weight_values [0:ARRAY_SIZE-1],
    input [7:0] weight_col_idx [0:ARRAY_SIZE-1],
    input weight_valid [0:ARRAY_SIZE-1],
    
    // 输出接口
    output reg [DATA_WIDTH+16-1:0] results [0:ARRAY_SIZE-1],
    output reg result_valid [0:ARRAY_SIZE-1]
);
    // 稀疏PE阵列
    wire [DATA_WIDTH-1:0] act_data [0:ARRAY_SIZE][0:ARRAY_SIZE];
    wire [7:0] act_idx [0:ARRAY_SIZE][0:ARRAY_SIZE];
    wire act_v [0:ARRAY_SIZE][0:ARRAY_SIZE];
    
    genvar i, j;
    generate
        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin : row
            for (j = 0; j < ARRAY_SIZE; j = j + 1) begin : col
                SparsePE pe_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    
                    // 激活输入
                    .act_value_in(act_data[i][j]),
                    .act_idx_in(act_idx[i][j]),
                    .act_valid_in(act_v[i][j]),
                    
                    // 权重输入
                    .weight_value(weight_values[j]),
                    .weight_idx(weight_col_idx[j]),
                    .weight_valid(weight_valid[j]),
                    
                    // 传递输出
                    .act_value_out(act_data[i][j+1]),
                    .act_idx_out(act_idx[i][j+1]),
                    .act_valid_out(act_v[i][j+1]),
                    
                    // 结果累加
                    .psum_in(i == 0 ? 0 : results[j]),
                    .psum_out(results[j]),
                    .psum_valid(result_valid[j])
                );
            end
        end
    endgenerate
endmodule

// 稀疏PE单元
module SparsePE #(
    parameter DATA_WIDTH = 16
)(
    input clk, rst_n,
    
    // 激活输入和传递
    input [DATA_WIDTH-1:0] act_value_in,
    input [7:0] act_idx_in,
    input act_valid_in,
    
    output reg [DATA_WIDTH-1:0] act_value_out,
    output reg [7:0] act_idx_out,
    output reg act_valid_out,
    
    // 权重输入
    input [DATA_WIDTH-1:0] weight_value,
    input [7:0] weight_idx,
    input weight_valid,
    
    // 部分和
    input [DATA_WIDTH+16-1:0] psum_in,
    output reg [DATA_WIDTH+16-1:0] psum_out,
    output reg psum_valid
);
    // 索引匹配检测
    wire index_match = (act_idx_in == weight_idx) && act_valid_in && weight_valid;
    
    // MAC运算
    wire [2*DATA_WIDTH-1:0] mult_result = act_value_in * weight_value;
    wire [DATA_WIDTH+16-1:0] add_result = psum_in + mult_result;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            act_value_out <= 0;
            act_idx_out <= 0;
            act_valid_out <= 0;
            psum_out <= 0;
            psum_valid <= 0;
        end else begin
            // 传递激活数据
            act_value_out <= act_value_in;
            act_idx_out <= act_idx_in;
            act_valid_out <= act_valid_in;
            
            // 条件MAC
            if (index_match) begin
                psum_out <= add_result;
                psum_valid <= 1;
            end else begin
                psum_out <= psum_in;
                psum_valid <= psum_valid;
            end
        end
    end
endmodule
```

</details>

**题目3.2：** 分析并优化NPU的功耗。给出至少三种降低功耗的架构级技术。

<details>
<summary>参考答案</summary>

**NPU功耗优化技术：**

**1. 时钟门控（Clock Gating）**
```verilog
// 细粒度时钟门控
module ClockGatedPE (
    input clk, rst_n,
    input enable,
    input [15:0] a_in, w_in,
    output reg [31:0] psum_out
);
    // 局部时钟生成
    wire gated_clk;
    ClockGate cg_inst (
        .clk(clk),
        .enable(enable || (a_in != 0 && w_in != 0)),
        .gated_clk(gated_clk)
    );
    
    // 只在有效数据时计算
    always @(posedge gated_clk) begin
        psum_out <= psum_out + a_in * w_in;
    end
endmodule
```

**2. 数据门控（Data Gating）**
- 零值检测和跳过
- 避免无效计算
- 减少数据翻转

**3. 电压频率调节（DVFS）**
```python
# 根据工作负载动态调整
def adaptive_dvfs(workload_type):
    if workload_type == "compute_bound":
        set_voltage(1.0)  # 高电压
        set_frequency(2.0) # 高频率
    elif workload_type == "memory_bound":
        set_voltage(0.8)  # 低电压
        set_frequency(1.0) # 低频率
    else:  # idle
        set_voltage(0.6)  # 最低电压
        set_frequency(0.5) # 最低频率
```

**4. 分层存储优化**
- 数据尽量在低层次存储间移动
- 减少DRAM访问
- 使用压缩技术

**5. 计算精度优化**
- 动态精度调整
- 混合精度计算
- 量化感知设计

**功耗分析示例：**
```
组件功耗分布（典型NPU）：
- MAC阵列：45%
- 片上存储：25%
- DRAM访问：20%
- 控制逻辑：5%
- 互连网络：5%

优化后：
- 稀疏计算减少MAC功耗：-30%
- 数据复用减少DRAM访问：-50%
- 时钟门控减少静态功耗：-20%
总体功耗降低：40-50%
```

</details>

**题目3.3：** 为NPU设计一个高效的任务调度器，支持多个神经网络模型的并发执行。

<details>
<summary>参考答案</summary>

```python
class NPUTaskScheduler:
    def __init__(self, num_compute_units, memory_size):
        self.compute_units = num_compute_units
        self.memory_size = memory_size
        self.task_queue = PriorityQueue()
        self.resource_map = ResourceMap()
        
    def schedule_task(self, task):
        """任务调度主函数"""
        # 1. 资源检查
        required_compute = task.compute_requirement
        required_memory = task.memory_requirement
        
        if not self.check_resources(required_compute, required_memory):
            # 资源不足，进入等待队列
            self.task_queue.put((task.priority, task))
            return
        
        # 2. 资源分配
        allocated_units = self.allocate_compute_units(required_compute)
        allocated_memory = self.allocate_memory(required_memory)
        
        # 3. 任务映射
        mapping = self.generate_mapping(task, allocated_units)
        
        # 4. 配置硬件
        self.configure_hardware(mapping, allocated_memory)
        
        # 5. 启动执行
        self.execute_task(task, mapping)
    
    def generate_mapping(self, task, compute_units):
        """生成任务到硬件的映射"""
        mapping = {}
        
        if task.type == "convolution":
            # 卷积层映射策略
            mapping = self.map_convolution(task, compute_units)
        elif task.type == "attention":
            # 注意力层映射策略
            mapping = self.map_attention(task, compute_units)
        elif task.type == "fully_connected":
            # 全连接层映射策略
            mapping = self.map_fc(task, compute_units)
            
        return mapping
    
    def map_convolution(self, task, units):
        """卷积层的优化映射"""
        # 考虑因素：
        # 1. 输入/输出通道并行
        # 2. 空间维度分块
        # 3. 数据复用模式
        
        batch_size = task.batch_size
        channels = task.channels
        spatial_size = task.spatial_size
        
        # 动态选择最佳分块策略
        if channels > units:
            # 通道并行
            strategy = "channel_parallel"
            tile_size = channels // units
        else:
            # 空间并行
            strategy = "spatial_parallel"
            tile_size = spatial_size // math.sqrt(units)
            
        return {
            "strategy": strategy,
            "tile_size": tile_size,
            "units": units
        }
    
    def handle_resource_conflict(self):
        """处理资源冲突"""
        # 抢占式调度
        if self.preemption_enabled:
            # 检查是否有低优先级任务可以暂停
            running_tasks = self.get_running_tasks()
            for task in running_tasks:
                if task.priority < self.task_queue.peek().priority:
                    self.preempt_task(task)
                    break
        
        # 任务迁移
        if self.migration_enabled:
            # 将任务迁移到其他可用NPU
            self.migrate_task_to_peer_npu()
```

**硬件支持模块：**

```verilog
module TaskSchedulerHW #(
    parameter MAX_TASKS = 16,
    parameter COMPUTE_UNITS = 64
)(
    input clk, rst_n,
    
    // 任务接口
    input task_valid,
    input [31:0] task_id,
    input [15:0] task_priority,
    input [31:0] task_compute_req,
    input [31:0] task_memory_req,
    
    // 资源状态
    input [COMPUTE_UNITS-1:0] unit_busy,
    input [31:0] free_memory,
    
    // 调度输出
    output reg schedule_valid,
    output reg [31:0] scheduled_task_id,
    output reg [COMPUTE_UNITS-1:0] allocated_units,
    output reg [31:0] memory_base_addr
);
    // 任务队列
    reg [31:0] task_queue [0:MAX_TASKS-1];
    reg [15:0] priority_queue [0:MAX_TASKS-1];
    reg [4:0] queue_head, queue_tail;
    
    // 资源分配器
    wire [COMPUTE_UNITS-1:0] available_units = ~unit_busy;
    wire [6:0] free_unit_count;
    
    // 计算可用单元数
    PopCount #(.WIDTH(COMPUTE_UNITS)) pc_inst (
        .in(available_units),
        .count(free_unit_count)
    );
    
    // 调度决策
    always @(posedge clk) begin
        if (!rst_n) begin
            schedule_valid <= 0;
            queue_head <= 0;
            queue_tail <= 0;
        end else begin
            // 新任务入队
            if (task_valid) begin
                task_queue[queue_tail] <= task_id;
                priority_queue[queue_tail] <= task_priority;
                queue_tail <= queue_tail + 1;
            end
            
            // 调度逻辑
            if (queue_head != queue_tail) begin
                if (free_unit_count >= task_compute_req &&
                    free_memory >= task_memory_req) begin
                    // 可以调度
                    schedule_valid <= 1;
                    scheduled_task_id <= task_queue[queue_head];
                    allocated_units <= allocate_units(task_compute_req);
                    memory_base_addr <= allocate_memory(task_memory_req);
                    queue_head <= queue_head + 1;
                end else begin
                    schedule_valid <= 0;
                end
            end
        end
    end
    
    // 资源分配函数
    function [COMPUTE_UNITS-1:0] allocate_units(input [6:0] count);
        // 实现最适合或首次适合算法
        // ...
    endfunction
endmodule
```

</details>

**题目3.6：** 分析Tensor Core架构相比传统MAC阵列的优势，并计算其理论性能提升。

<details>
<summary>💡 提示</summary>

思考方向：Tensor Core是以矩阵为基本计算单位，而不是标量。比较一次操作完成的计算量、数据复用率、带宽需求。考虑不同精度（FP16、INT8）的影响。

</details>

<details>
<summary>参考答案</summary>

**1. 架构对比：**

| 特性 | 传统MAC阵列 | Tensor Core |
|------|------------|-------------|
| 基本运算 | 标量MAC: c += a × b | 矩阵MAC: D = A×B + C |
| 运算粒度 | 1×1 | 4×4×4 (或更大) |
| 每周期运算量 | 2 ops (乘+加) | 128 ops (4×4×4×2) |
| 数据复用 | 有限 | 矩阵级复用 |

**2. Tensor Core工作原理：**

```
// Tensor Core执行的运算
D[4×4] = A[4×4] × B[4×4] + C[4×4]

// 分解为标量运算：
for i in 0..3:
    for j in 0..3:
        sum = 0
        for k in 0..3:
            sum += A[i][k] * B[k][j]
        D[i][j] = sum + C[i][j]

// 总运算数：4×4×4 = 64次乘法，48次加法，16次加法
// 共128 ops
```

**3. 性能提升计算：**

假设：
- 传统MAC阵列：16×16 = 256个MAC单元
- Tensor Core阵列：4×4 = 16个Tensor Core
- 相同的总硬件面积

性能对比：
- 传统MAC：256 × 2 = 512 ops/cycle
- Tensor Core：16 × 128 = 2048 ops/cycle
- **理论加速比：4×**

**4. 优势分析：**
1. **更高的计算密度：** 相同面积下提供更多运算
2. **更好的数据复用：** 矩阵运算天然具有数据复用
3. **减少控制开销：** 一条指令完成更多运算
4. **更适合深度学习：** 直接匹配GEMM运算模式

**5. 限制条件：**
- 需要对齐到4×4块大小
- 不适合稀疏或不规则运算
- 精度限制（通常是混合精度）
- 编程模型相对复杂

</details>

**题目3.7：** 设计一个简单的NPU指令集架构(ISA)，包含计算、数据传输和控制指令。

<details>
<summary>💡 提示</summary>

思考方向：NPU ISA需要支持矩阵运算、卷积、激活函数等。设计指令格式、寻址模式、寄存器文件。考虑VLIW或SIMD指令集的特点。

</details>

<details>
<summary>参考答案</summary>

**NPU ISA设计：**

**1. 指令格式（32-bit）：**

```
[31:28] | [27:24] | [23:16] | [15:8] | [7:0]
OPCODE  | FLAGS   | DEST    | SRC1   | SRC2/IMM

OPCODE: 4-bit 操作码
FLAGS:  4-bit 标志位（精度、饱和模式等）
DEST:   8-bit 目标寄存器/地址
SRC1:   8-bit 源操作数1
SRC2:   8-bit 源操作数2或立即数
```

**2. 指令集分类：**

**A. 计算指令：**

| 助记符 | 操作码 | 功能 | 示例 |
|-------|--------|------|------|
| MMUL | 0x0 | 矩阵乘法 | MMUL R0, M1, M2 |
| CONV | 0x1 | 卷积运算 | CONV R0, I1, W1 |
| MADD | 0x2 | 矩阵加法 | MADD R0, M1, M2 |
| RELU | 0x3 | ReLU激活 | RELU R0, M1 |
| POOL | 0x4 | 池化操作 | POOL R0, I1, 2x2 |

**B. 数据传输指令：**

| 助记符 | 操作码 | 功能 | 示例 |
|-------|--------|------|------|
| LOAD | 0x8 | 从内存加载 | LOAD M0, [ADDR] |
| STORE | 0x9 | 存储到内存 | STORE [ADDR], M0 |
| MOV | 0xA | 寄存器传输 | MOV R1, R0 |
| BCAST | 0xB | 广播数据 | BCAST M0, R0 |

**C. 控制指令：**

| 助记符 | 操作码 | 功能 | 示例 |
|-------|--------|------|------|
| SYNC | 0xC | 同步屏障 | SYNC |
| LOOP | 0xD | 循环控制 | LOOP 16, LABEL |
| JMP | 0xE | 跳转 | JMP LABEL |
| NOP | 0xF | 空操作 | NOP |

**3. 寄存器文件：**
- 32个矩阵寄存器（M0-M31）：每个可存储16×16矩阵
- 16个标量寄存器（R0-R15）：用于地址和控制
- 特殊寄存器：PC、STATUS、CONFIG

**4. 寻址模式：**
- 直接寻址：LOAD M0, [0x1000]
- 寄存器间接：LOAD M0, [R1]
- 索引寻址：LOAD M0, [R1 + R2]
- 自增寻址：LOAD M0, [R1++]

**5. VLIW打包示例：**

```
// 4个操作并行执行
{
    LOAD M0, [R1++];    // 加载输入
    LOAD M1, [R2++];    // 加载权重
    MMUL M2, M0, M1;    // 矩阵乘法
    RELU M3, M2;        // 激活函数
}
```

</details>

**题目3.8：** 评估不同的功耗优化技术对NPU的影响。给定一个100 TOPS的NPU，分析各种技术的节能潜力。

<details>
<summary>💡 提示</summary>

思考方向：从动态功耗（开关活动）和静态功耗（漏电流）两方面分析。考虑时钟门控、电压频率调节（DVFS）、数据精度优化、稀疏性利用等技术。

</details>

<details>
<summary>参考答案</summary>

**基准NPU规格：**
- 峰值性能：100 TOPS (INT8)
- 功耗：50W (2 TOPS/W)
- 工艺：7nm
- 频率：1GHz

**功耗优化技术分析：**

| 优化技术 | 原理 | 节能潜力 | 实现复杂度 | 性能影响 |
|---------|------|---------|-----------|---------|
| **时钟门控** | 关闭空闲单元时钟 | 15-25% | 低 | 无 |
| **电压频率调节(DVFS)** | 动态调整V/F | 30-40% | 中 | 可能降低 |
| **数据精度优化** | INT8→INT4 | 40-50% | 中 | 精度损失 |
| **稀疏性利用** | 跳过零值计算 | 20-60% | 高 | 依赖稀疏度 |
| **近阈值计算** | 超低电压操作 | 50-70% | 高 | 性能大幅降低 |

**详细分析：**

1. **时钟门控实现：**
```verilog
// 细粒度时钟门控
always @(*) begin
    pe_clk_en = (data_valid && weight_valid) || 
                (pipeline_stage > 0);
end
// 节能：避免无效翻转，减少15-25%动态功耗
```

2. **DVFS策略：**
```python
# 根据负载调节
if utilization < 0.3:
    set_vf_level(0)  # 0.6V, 500MHz
elif utilization < 0.7:
    set_vf_level(1)  # 0.8V, 750MHz
else:
    set_vf_level(2)  # 1.0V, 1GHz
# 节能：P ∝ V²F，可节省30-40%
```

3. **混合精度计算：**
- 第一层：INT8（保持精度）
- 中间层：INT4（容忍精度损失）
- 最后层：INT8（输出质量）
- 节能：减少乘法器面积和功耗40-50%

4. **稀疏性感知：**
```
原始计算：100 TOPS × 50% 稀疏度 = 50 TOPS有效
优化后：仅计算非零值，节省50%功耗
实际节能：考虑检测开销，净节省30-40%
```

**综合优化方案：**
- 基准功耗：50W
- 时钟门控：-10W (20%)
- DVFS（平均）：-10W (20%)
- 混合精度：-8W (16%)
- 稀疏优化：-7W (14%)
- **优化后功耗：15W**
- **能效提升：6.67 TOPS/W（3.3×提升）**

**实施建议：**
1. 优先实施时钟门控（简单有效）
2. 结合DVFS和任务调度
3. 算法层面引入稀疏性
4. 谨慎使用激进的低功耗技术

</details>

## 本章小结

- **NPU系统架构是一个精心设计的数据处理工厂，** 通过计算集群、存储系统、DMA引擎和互连网络的协同工作实现高效的AI计算
- **层次化存储是NPU性能的关键，** 通过L0寄存器、L1 SRAM、L2缓存和外部DRAM的合理设计，最大化数据复用并最小化数据搬运开销
- **数据流管理决定了NPU的效率，** Double Buffering、数据预取、同步机制等技术确保计算单元持续满负荷运行
- **DMA引擎是数据搬运的主力军，** 通过多通道设计、描述符链表、地址生成单元等技术实现高带宽低延迟的数据传输
- **调度器是NPU的大脑，** 负责任务分解、资源分配、依赖管理，确保硬件资源得到充分利用
- **片上互连（NoC）提供灵活的通信架构，** Mesh、Torus等拓扑结构支持大规模并行计算的扩展
- **系统集成需要软硬件协同设计，** ISA定义、编译器优化、运行时管理共同决定了NPU的实际性能表现