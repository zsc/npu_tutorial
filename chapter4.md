# 第4章：计算核心设计

在上一章中，我们从系统层面了解了NPU的整体架构。现在，让我们深入到NPU的心脏——计算核心。如果说NPU是一座高效运转的工厂，那么计算核心就是工厂里的生产线，而MAC（Multiply-Accumulate）单元则是生产线上的工人。本章将详细探讨如何设计高效的计算核心，从基础的MAC单元开始，逐步构建起能够处理海量神经网络运算的脉动阵列。

我们将重点关注三个关键问题：**如何设计单个MAC单元以实现最高效率？如何将成千上万个MAC单元组织成阵列？如何通过不同的数据流模式（Weight Stationary、Output Stationary、Row Stationary）来优化不同场景下的计算效率？**通过回答这些问题，你将掌握NPU计算核心设计的精髓。

## 4.1 MAC阵列设计

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

```python
// 全连接层的数学表达
Y = W × X + B

// 分解为MAC操作
for i in 0..M:
    for j in 0..N:
        Y[i] = 0
        for k in 0..K:
            Y[i] += W[i][k] * X[k][j]  // 这就是MAC操作
        Y[i] += B[i]

// 一个1024×1024的矩阵乘法需要：
// 1024³ = 1,073,741,824 次MAC操作！
```

这就是为什么现代AI芯片都在疯狂堆砌MAC单元的原因。Google TPU v1拥有65,536个MAC单元，而最新的NVIDIA H100则包含了数百万个等效MAC单元。

```verilog
// 优化的流水线MAC单元 - Verilog版本
module MAC_Unit #(
    parameter DATA_WIDTH = 8,      // 输入数据位宽(INT8)
    parameter ACC_WIDTH = 32,      // 累加器位宽(防止溢出)
    parameter PIPELINE_STAGES = 3  // 流水线级数
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    // 输入接口
    input wire signed [DATA_WIDTH-1:0] a_in,      // 激活值
    input wire signed [DATA_WIDTH-1:0] b_in,      // 权重
    input wire signed [ACC_WIDTH-1:0] c_in,       // 部分和输入
    
    // 输出接口
    output reg signed [ACC_WIDTH-1:0] c_out,      // 累加结果
    output reg valid_out
);

    // 流水线寄存器 - 第一级
    reg signed [DATA_WIDTH-1:0] a_reg1, b_reg1;
    reg signed [ACC_WIDTH-1:0] c_reg1;
    reg enable_reg1;
    
    // 流水线寄存器 - 第二级
    reg signed [2*DATA_WIDTH-1:0] mult_reg2;
    reg signed [ACC_WIDTH-1:0] c_reg2;
    reg enable_reg2;
    
    // 乘法结果（第二级计算）
    wire signed [2*DATA_WIDTH-1:0] mult_result;
    assign mult_result = a_reg1 * b_reg1;
    
    // 加法结果（第三级计算）
    wire signed [ACC_WIDTH-1:0] add_result;
    assign add_result = c_reg2 + {{(ACC_WIDTH-2*DATA_WIDTH){mult_reg2[2*DATA_WIDTH-1]}}, mult_reg2};
    
    // 第一级流水线：输入寄存
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_reg1 <= 0;
            b_reg1 <= 0;
            c_reg1 <= 0;
            enable_reg1 <= 0;
        end else begin
            if (enable) begin
                a_reg1 <= a_in;
                b_reg1 <= b_in;
                c_reg1 <= c_in;
            end
            enable_reg1 <= enable;
        end
    end
    
    // 第二级流水线：乘法结果寄存
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mult_reg2 <= 0;
            c_reg2 <= 0;
            enable_reg2 <= 0;
        end else begin
            if (enable_reg1) begin
                mult_reg2 <= mult_result;
                c_reg2 <= c_reg1;
            end
            enable_reg2 <= enable_reg1;
        end
    end
    
    // 第三级流水线：累加结果输出
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            c_out <= 0;
            valid_out <= 0;
        end else begin
            if (enable_reg2) begin
                c_out <= add_result;
            end
            valid_out <= enable_reg2;
        end
    end
endmodule
```

Chisel版本的MAC单元：
```scala
import chisel3._
import chisel3.util._

class MACUnit(val dataWidth: Int = 8, val accWidth: Int = 32) extends Module {
    val io = IO(new Bundle {
        val enable = Input(Bool())
        val a_in = Input(SInt(dataWidth.W))
        val b_in = Input(SInt(dataWidth.W))
        val c_in = Input(SInt(accWidth.W))
        val c_out = Output(SInt(accWidth.W))
        val valid_out = Output(Bool())
    })
    
    // 第一级流水线：输入寄存器
    val a_reg1 = RegEnable(io.a_in, 0.S(dataWidth.W), io.enable)
    val b_reg1 = RegEnable(io.b_in, 0.S(dataWidth.W), io.enable)
    val c_reg1 = RegEnable(io.c_in, 0.S(accWidth.W), io.enable)
    val enable_reg1 = RegNext(io.enable, false.B)
    
    // 第二级流水线：乘法
    val mult_result = a_reg1 * b_reg1
    val mult_reg2 = RegEnable(mult_result, 0.S((2*dataWidth).W), enable_reg1)
    val c_reg2 = RegEnable(c_reg1, 0.S(accWidth.W), enable_reg1)
    val enable_reg2 = RegNext(enable_reg1, false.B)
    
    // 第三级流水线：累加
    val mult_extended = Wire(SInt(accWidth.W))
    mult_extended := mult_reg2.asSInt
    val add_result = c_reg2 + mult_extended
    
    io.c_out := RegEnable(add_result, 0.S(accWidth.W), enable_reg2)
    io.valid_out := RegNext(enable_reg2, false.B)
}

// 生成Verilog
object MACUnitGen extends App {
    (new chisel3.stage.ChiselStage).emitVerilog(
        new MACUnit(),
        Array("--target-dir", "generated")
    )
}
```

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
```verilog
// 二维MAC阵列组织示例 (8x8)
module MAC_Array_8x8 #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32,
    parameter ARRAY_SIZE = 8
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    // 输入数据广播
    input wire [DATA_WIDTH-1:0] act_broadcast [0:ARRAY_SIZE-1],  // 激活值广播
    input wire [DATA_WIDTH-1:0] weight_array [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1],  // 权重
    
    // 部分和累加
    output wire [ACC_WIDTH-1:0] psum_out [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1]
);

    // MAC单元阵列
    genvar i, j;
    generate
        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin : row
            for (j = 0; j < ARRAY_SIZE; j = j + 1) begin : col
                MAC_Unit #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH)
                ) mac_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .enable(enable),
                    .a_in(act_broadcast[i]),              // 行广播
                    .b_in(weight_array[i][j]),            // 本地权重
                    .c_in(/* 根据数据流选择 */),
                    .c_out(psum_out[i][j])
                );
            end
        end
    endgenerate
endmodule
```

## 4.2 脉动阵列架构

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
```verilog
// 权重固定型脉动阵列PE
module SystolicPE_WS #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    
    // 控制信号
    input wire weight_load,      // 权重加载使能
    input wire compute_en,       // 计算使能
    
    // 数据输入
    input wire [DATA_WIDTH-1:0] act_in,      // 激活值输入（从上方）
    input wire [DATA_WIDTH-1:0] weight_in,   // 权重输入（加载时）
    input wire [ACC_WIDTH-1:0] psum_in,      // 部分和输入（从左侧）
    
    // 数据输出
    output reg [DATA_WIDTH-1:0] act_out,     // 激活值输出（向下方）
    output reg [ACC_WIDTH-1:0] psum_out      // 部分和输出（向右侧）
);

    // 内部寄存器
    reg [DATA_WIDTH-1:0] weight_reg;         // 存储的权重
    reg [DATA_WIDTH-1:0] act_reg;            // 激活值寄存器
    
    // MAC运算
    wire [2*DATA_WIDTH-1:0] mult_result;
    wire [ACC_WIDTH-1:0] mac_result;
    
    assign mult_result = act_reg * weight_reg;
    assign mac_result = psum_in + {{(ACC_WIDTH-2*DATA_WIDTH){mult_result[2*DATA_WIDTH-1]}}, mult_result};
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg <= 0;
            act_reg <= 0;
            act_out <= 0;
            psum_out <= 0;
        end else begin
            // 权重加载
            if (weight_load) begin
                weight_reg <= weight_in;
            end
            
            // 计算模式
            if (compute_en) begin
                // 激活值向下传递
                act_reg <= act_in;
                act_out <= act_reg;
                
                // MAC结果向右传递
                psum_out <= mac_result;
            end
        end
    end
endmodule

// 优化的流水线脉动阵列 - Verilog版本
module SystolicArray_4x4_WS #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32,
    parameter ARRAY_DIM = 4
)(
    input wire clk,
    input wire rst_n,
    input wire weight_load,
    input wire compute_en,
    
    // 激活值输入（从顶部进入，已寄存）
    input wire [DATA_WIDTH-1:0] act_in [0:ARRAY_DIM-1],
    
    // 权重加载接口
    input wire [DATA_WIDTH-1:0] weight_in [0:ARRAY_DIM-1][0:ARRAY_DIM-1],
    
    // 结果输出（从右侧输出，已寄存）
    output wire [ACC_WIDTH-1:0] result_out [0:ARRAY_DIM-1]
);

    // PE间的寄存连接
    reg [DATA_WIDTH-1:0] act_reg [0:ARRAY_DIM][0:ARRAY_DIM-1];  // 激活值寄存器
    reg [ACC_WIDTH-1:0] psum_reg [0:ARRAY_DIM-1][0:ARRAY_DIM];  // 部分和寄存器
    reg valid_reg [0:ARRAY_DIM][0:ARRAY_DIM-1];                 // 有效信号寄存器
    
    // 初始化边界条件
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < ARRAY_DIM; i++) begin
                psum_reg[i][0] <= 0;  // 左边界部分和为0
                valid_reg[0][i] <= 0;  // 顶部有效信号初始化
            end
        end else begin
            // 左边界保持为0
            for (int i = 0; i < ARRAY_DIM; i++) begin
                psum_reg[i][0] <= 0;
            end
            // 顶部输入寄存
            for (int j = 0; j < ARRAY_DIM; j++) begin
                act_reg[0][j] <= act_in[j];
                valid_reg[0][j] <= compute_en;
            end
        end
    end
    
    // PE阵列实例化（优化版本）
    genvar i, j;
    generate
        for (i = 0; i < ARRAY_DIM; i = i + 1) begin : pe_row
            for (j = 0; j < ARRAY_DIM; j = j + 1) begin : pe_col
                SystolicPE_WS_Pipelined #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH)
                ) pe_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .weight_load(weight_load),
                    .valid_in(valid_reg[i][j]),
                    .act_in(act_reg[i][j]),
                    .weight_in(weight_in[i][j]),
                    .psum_in(psum_reg[i][j]),
                    .act_out(act_reg[i+1][j]),
                    .psum_out(psum_reg[i][j+1]),
                    .valid_out(valid_reg[i+1][j])
                );
            end
        end
    endgenerate
    
    // 输出连接（已寄存）
    generate
        for (i = 0; i < ARRAY_DIM; i = i + 1) begin
            assign result_out[i] = psum_reg[i][ARRAY_DIM];
        end
    endgenerate
endmodule

// 优化的流水线PE单元
module SystolicPE_WS_Pipelined #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    input wire weight_load,
    input wire valid_in,
    input wire [DATA_WIDTH-1:0] act_in,
    input wire [DATA_WIDTH-1:0] weight_in,
    input wire [ACC_WIDTH-1:0] psum_in,
    output reg [DATA_WIDTH-1:0] act_out,
    output reg [ACC_WIDTH-1:0] psum_out,
    output reg valid_out
);
    // 权重寄存器（保持不变）
    reg [DATA_WIDTH-1:0] weight_reg;
    
    // 流水线寄存器
    reg [DATA_WIDTH-1:0] act_reg;
    reg [ACC_WIDTH-1:0] psum_reg;
    reg valid_reg;
    
    // MAC计算（组合逻辑）
    wire [2*DATA_WIDTH-1:0] mult_result;
    wire [ACC_WIDTH-1:0] mac_result;
    
    assign mult_result = $signed(act_reg) * $signed(weight_reg);
    assign mac_result = psum_reg + {{(ACC_WIDTH-2*DATA_WIDTH){mult_result[2*DATA_WIDTH-1]}}, mult_result};
    
    // 权重加载
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg <= 0;
        end else if (weight_load) begin
            weight_reg <= weight_in;
        end
    end
    
    // 流水线第一级：输入寄存
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            act_reg <= 0;
            psum_reg <= 0;
            valid_reg <= 0;
        end else begin
            act_reg <= act_in;
            psum_reg <= psum_in;
            valid_reg <= valid_in;
        end
    end
    
    // 流水线第二级：输出寄存
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            act_out <= 0;
            psum_out <= 0;
            valid_out <= 0;
        end else begin
            act_out <= act_reg;  // 激活值向下传递
            psum_out <= valid_reg ? mac_result : psum_reg;  // MAC结果向右传递
            valid_out <= valid_reg;
        end
    end
endmodule
```

Chisel版本的脉动阵列：
```scala
import chisel3._
import chisel3.util._

// 脉动PE单元
class SystolicPE(val dataWidth: Int = 8, val accWidth: Int = 32) extends Module {
    val io = IO(new Bundle {
        val weight_load = Input(Bool())
        val valid_in = Input(Bool())
        val act_in = Input(SInt(dataWidth.W))
        val weight_in = Input(SInt(dataWidth.W))
        val psum_in = Input(SInt(accWidth.W))
        val act_out = Output(SInt(dataWidth.W))
        val psum_out = Output(SInt(accWidth.W))
        val valid_out = Output(Bool())
    })
    
    // 权重寄存器
    val weight_reg = RegInit(0.S(dataWidth.W))
    when(io.weight_load) {
        weight_reg := io.weight_in
    }
    
    // 流水线寄存器
    val act_reg = RegNext(io.act_in)
    val psum_reg = RegNext(io.psum_in)
    val valid_reg = RegNext(io.valid_in)
    
    // MAC计算
    val mult_result = act_reg * weight_reg
    val mac_result = psum_reg + mult_result
    
    // 输出寄存器
    io.act_out := RegNext(act_reg)
    io.psum_out := RegNext(Mux(valid_reg, mac_result, psum_reg))
    io.valid_out := RegNext(valid_reg)
}

// 4x4脉动阵列
class SystolicArray4x4(val dataWidth: Int = 8, val accWidth: Int = 32) extends Module {
    val arrayDim = 4
    val io = IO(new Bundle {
        val weight_load = Input(Bool())
        val compute_en = Input(Bool())
        val act_in = Input(Vec(arrayDim, SInt(dataWidth.W)))
        val weight_in = Input(Vec(arrayDim, Vec(arrayDim, SInt(dataWidth.W))))
        val result_out = Output(Vec(arrayDim, SInt(accWidth.W)))
    })
    
    // 创建PE阵列
    val peArray = Array.fill(arrayDim, arrayDim)(Module(new SystolicPE(dataWidth, accWidth)))
    
    // 连接PE阵列
    for (i <- 0 until arrayDim) {
        for (j <- 0 until arrayDim) {
            val pe = peArray(i)(j)
            
            // 权重加载
            pe.io.weight_load := io.weight_load
            pe.io.weight_in := io.weight_in(i)(j)
            
            // 激活值连接（从上到下）
            if (i == 0) {
                pe.io.act_in := RegNext(io.act_in(j))
                pe.io.valid_in := RegNext(io.compute_en)
            } else {
                pe.io.act_in := peArray(i-1)(j).io.act_out
                pe.io.valid_in := peArray(i-1)(j).io.valid_out
            }
            
            // 部分和连接（从左到右）
            if (j == 0) {
                pe.io.psum_in := 0.S
            } else {
                pe.io.psum_in := peArray(i)(j-1).io.psum_out
            }
        }
    }
    
    // 输出连接
    for (i <- 0 until arrayDim) {
        io.result_out(i) := peArray(i)(arrayDim-1).io.psum_out
    }
}
```

### 4.2.3 脉动阵列数据流动示例

以2×2矩阵乘法为例，展示数据在脉动阵列中的流动过程：

```
矩阵A = [a00 a01]    矩阵B = [b00 b01]    结果C = A×B
        [a10 a11]            [b10 b11]

时刻0: 权重加载
PE[0][0] <- b00    PE[0][1] <- b01
PE[1][0] <- b10    PE[1][1] <- b11

时刻1: 
输入: a00, a10 (错开一个周期)
      ↓
    [b00]--[b01]    a00×b00 → PE[0][0]
      ↓
    [b10]--[b11]    

时刻2:
输入: a01, a11
    a00  ↓
    [b00]--[b01]    a00×b01 → PE[0][1], a10×b00 → PE[1][0]
    a10  ↓
    [b10]--[b11]

时刻3:
    a01  a00
    [b00]--[b01]→c00   a01×b10 → PE[0][0], a10×b01 → PE[1][1]
    a11  a10
    [b10]--[b11]

时刻4:
         a01
    [b00]--[b01]→c01   a01×b11 → PE[0][1], a11×b10 → PE[1][0]
         a11
    [b10]--[b11]→c10

时刻5:
    [b00]--[b01]       a11×b11 → PE[1][1]
    [b10]--[b11]→c11
```

### 4.2.4 Output Stationary 脉动阵列实现

Output Stationary（输出固定）是另一种重要的脉动阵列架构，特别适合深度卷积和批处理场景。在这种架构中，每个PE负责计算输出矩阵的一个固定元素，输入数据和权重在PE阵列中流动。

```verilog
// 优化的Output Stationary脉动阵列 - Verilog版本
module OutputStationarySystolicArray #(
    parameter ARRAY_SIZE = 4,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32,
    parameter PIPELINE_STAGES = 3
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire clear_acc,
    
    // 数据输入 - A矩阵从左侧输入，每行错开一个周期
    input wire signed [DATA_WIDTH-1:0] a_data_in [0:ARRAY_SIZE-1],
    input wire a_valid_in [0:ARRAY_SIZE-1],
    
    // 权重输入 - B矩阵从顶部输入，每列错开一个周期
    input wire signed [DATA_WIDTH-1:0] b_data_in [0:ARRAY_SIZE-1],
    input wire b_valid_in [0:ARRAY_SIZE-1],
    
    // 结果输出 - C矩阵
    output reg signed [ACC_WIDTH-1:0] c_data_out [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1],
    output reg c_valid_out [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1]
);

    // 内部信号
    wire signed [DATA_WIDTH-1:0] a_flow [0:ARRAY_SIZE-1][0:ARRAY_SIZE];
    wire signed [DATA_WIDTH-1:0] b_flow [0:ARRAY_SIZE][0:ARRAY_SIZE-1];
    wire a_valid_flow [0:ARRAY_SIZE-1][0:ARRAY_SIZE];
    wire b_valid_flow [0:ARRAY_SIZE][0:ARRAY_SIZE-1];
    wire signed [ACC_WIDTH-1:0] pe_results [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    wire pe_valid [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    
    // 输入延迟寄存器（创建数据错位）
    reg signed [DATA_WIDTH-1:0] a_delay_reg [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    reg signed [DATA_WIDTH-1:0] b_delay_reg [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    reg a_valid_delay [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    reg b_valid_delay [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    
    // 生成输入延迟链
    genvar d, r, c;
    generate
        // A矩阵输入延迟（每行延迟递增）
        for (r = 0; r < ARRAY_SIZE; r = r + 1) begin : a_delay_gen
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int i = 0; i < r; i = i + 1) begin
                        a_delay_reg[r][i] <= 0;
                        a_valid_delay[r][i] <= 0;
                    end
                end else if (enable) begin
                    if (r == 0) begin
                        // 第一行无延迟
                        a_flow[0][0] <= a_data_in[0];
                        a_valid_flow[0][0] <= a_valid_in[0];
                    end else begin
                        // 延迟链
                        a_delay_reg[r][0] <= a_data_in[r];
                        a_valid_delay[r][0] <= a_valid_in[r];
                        for (int i = 1; i < r; i = i + 1) begin
                            a_delay_reg[r][i] <= a_delay_reg[r][i-1];
                            a_valid_delay[r][i] <= a_valid_delay[r][i-1];
                        end
                        a_flow[r][0] <= a_delay_reg[r][r-1];
                        a_valid_flow[r][0] <= a_valid_delay[r][r-1];
                    end
                end
            end
        end
        
        // B矩阵输入延迟（每列延迟递增）
        for (c = 0; c < ARRAY_SIZE; c = c + 1) begin : b_delay_gen
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int i = 0; i < c; i = i + 1) begin
                        b_delay_reg[c][i] <= 0;
                        b_valid_delay[c][i] <= 0;
                    end
                end else if (enable) begin
                    if (c == 0) begin
                        // 第一列无延迟
                        b_flow[0][0] <= b_data_in[0];
                        b_valid_flow[0][0] <= b_valid_in[0];
                    end else begin
                        // 延迟链
                        b_delay_reg[c][0] <= b_data_in[c];
                        b_valid_delay[c][0] <= b_valid_in[c];
                        for (int i = 1; i < c; i = i + 1) begin
                            b_delay_reg[c][i] <= b_delay_reg[c][i-1];
                            b_valid_delay[c][i] <= b_valid_delay[c][i-1];
                        end
                        b_flow[0][c] <= b_delay_reg[c][c-1];
                        b_valid_flow[0][c] <= b_valid_delay[c][c-1];
                    end
                end
            end
        end
    endgenerate
    
    // PE阵列实例化
    generate
        for (r = 0; r < ARRAY_SIZE; r = r + 1) begin : pe_row
            for (c = 0; c < ARRAY_SIZE; c = c + 1) begin : pe_col
                OutputStationaryPE #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH)
                ) pe_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .enable(enable),
                    .clear_acc(clear_acc),
                    
                    // A数据从左向右流动
                    .a_data_in(a_flow[r][c]),
                    .a_valid_in(a_valid_flow[r][c]),
                    .a_data_out(a_flow[r][c+1]),
                    .a_valid_out(a_valid_flow[r][c+1]),
                    
                    // B数据从上向下流动
                    .b_data_in(b_flow[r][c]),
                    .b_valid_in(b_valid_flow[r][c]),
                    .b_data_out(b_flow[r+1][c]),
                    .b_valid_out(b_valid_flow[r+1][c]),
                    
                    // 累加结果
                    .acc_out(pe_results[r][c]),
                    .acc_valid(pe_valid[r][c])
                );
            end
        end
    endgenerate
    
    // 输出寄存
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < ARRAY_SIZE; i = i + 1) begin
                for (int j = 0; j < ARRAY_SIZE; j = j + 1) begin
                    c_data_out[i][j] <= 0;
                    c_valid_out[i][j] <= 0;
                end
            end
        end else begin
            for (int i = 0; i < ARRAY_SIZE; i = i + 1) begin
                for (int j = 0; j < ARRAY_SIZE; j = j + 1) begin
                    c_data_out[i][j] <= pe_results[i][j];
                    c_valid_out[i][j] <= pe_valid[i][j];
                end
            end
        end
    end

endmodule

// Output Stationary PE单元
module OutputStationaryPE #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire clear_acc,
    
    // A数据接口（水平流动）
    input wire signed [DATA_WIDTH-1:0] a_data_in,
    input wire a_valid_in,
    output reg signed [DATA_WIDTH-1:0] a_data_out,
    output reg a_valid_out,
    
    // B数据接口（垂直流动）
    input wire signed [DATA_WIDTH-1:0] b_data_in,
    input wire b_valid_in,
    output reg signed [DATA_WIDTH-1:0] b_data_out,
    output reg b_valid_out,
    
    // 累加结果（固定在PE中）
    output reg signed [ACC_WIDTH-1:0] acc_out,
    output reg acc_valid
);

    // 内部寄存器
    reg signed [2*DATA_WIDTH-1:0] mult_result;
    reg mult_valid;
    reg signed [ACC_WIDTH-1:0] acc_reg;
    
    // 数据传递流水线
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_data_out <= 0;
            a_valid_out <= 0;
            b_data_out <= 0;
            b_valid_out <= 0;
        end else if (enable) begin
            // 数据向右和向下传递
            a_data_out <= a_data_in;
            a_valid_out <= a_valid_in;
            b_data_out <= b_data_in;
            b_valid_out <= b_valid_in;
        end
    end
    
    // 乘法流水线
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mult_result <= 0;
            mult_valid <= 0;
        end else if (enable) begin
            if (a_valid_in && b_valid_in) begin
                mult_result <= a_data_in * b_data_in;
                mult_valid <= 1;
            end else begin
                mult_result <= 0;
                mult_valid <= 0;
            end
        end
    end
    
    // 累加流水线
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_reg <= 0;
            acc_valid <= 0;
        end else if (clear_acc) begin
            acc_reg <= 0;
            acc_valid <= 0;
        end else if (enable && mult_valid) begin
            acc_reg <= acc_reg + mult_result;
            acc_valid <= 1;
        end
    end
    
    // 输出
    assign acc_out = acc_reg;

endmodule
```

```scala
// Chisel版本的Output Stationary脉动阵列
import chisel3._
import chisel3.util._

class OutputStationaryPE(dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  val io = IO(new Bundle {
    val enable = Input(Bool())
    val clearAcc = Input(Bool())
    
    // A数据接口（水平流动）
    val aDataIn = Input(SInt(dataWidth.W))
    val aValidIn = Input(Bool())
    val aDataOut = Output(SInt(dataWidth.W))
    val aValidOut = Output(Bool())
    
    // B数据接口（垂直流动）
    val bDataIn = Input(SInt(dataWidth.W))
    val bValidIn = Input(Bool())
    val bDataOut = Output(SInt(dataWidth.W))
    val bValidOut = Output(Bool())
    
    // 累加结果
    val accOut = Output(SInt(accWidth.W))
    val accValid = Output(Bool())
  })
  
  // 数据传递寄存器
  val aDataReg = RegNext(io.aDataIn)
  val aValidReg = RegNext(io.aValidIn)
  val bDataReg = RegNext(io.bDataIn)
  val bValidReg = RegNext(io.bValidIn)
  
  // 乘法流水线
  val multResult = RegNext(io.aDataIn * io.bDataIn)
  val multValid = RegNext(io.aValidIn && io.bValidIn)
  
  // 累加器
  val accReg = RegInit(0.S(accWidth.W))
  val accValidReg = RegInit(false.B)
  
  when (io.clearAcc) {
    accReg := 0.S
    accValidReg := false.B
  }.elsewhen (io.enable && multValid) {
    accReg := accReg + multResult
    accValidReg := true.B
  }
  
  // 输出连接
  io.aDataOut := aDataReg
  io.aValidOut := aValidReg
  io.bDataOut := bDataReg
  io.bValidOut := bValidReg
  io.accOut := accReg
  io.accValid := accValidReg
}
```

### 4.2.5 Output Stationary vs Weight Stationary 对比

| 特性 | Weight Stationary | Output Stationary | 适用场景 |
|------|------------------|-------------------|----------|
| 数据复用 | 权重驻留在PE中 | 部分和驻留在PE中 | WS: 批量小<br>OS: 批量大 |
| 内存带宽 | 输入/输出带宽高 | 权重/输入带宽高 | WS: 权重复用多<br>OS: 输出通道多 |
| 控制复杂度 | 简单 | 中等 | WS: 资源受限<br>OS: 性能优先 |
| 延迟 | 较低 | 较高（需要数据对齐） | WS: 实时推理<br>OS: 批处理训练 |
| 能效 | 权重读取能耗低 | 部分和读写能耗低 | WS: 边缘设备<br>OS: 数据中心 |

## 4.3 向量处理单元

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

```verilog
module VectorProcessingUnit #(
    parameter VECTOR_WIDTH = 16,    // 向量宽度（并行度）
    parameter DATA_WIDTH = 8,       // 数据位宽
    parameter OPCODE_WIDTH = 5      // 操作码宽度
)(
    input wire clk,
    input wire rst_n,
    
    // 指令接口
    input wire [OPCODE_WIDTH-1:0] opcode,
    input wire execute,
    
    // 向量输入
    input wire [DATA_WIDTH-1:0] vec_a [0:VECTOR_WIDTH-1],
    input wire [DATA_WIDTH-1:0] vec_b [0:VECTOR_WIDTH-1],
    
    // 向量输出
    output reg [DATA_WIDTH-1:0] vec_result [0:VECTOR_WIDTH-1],
    output reg done
);

    // 操作码定义
    localparam OP_ADD  = 5'b00001;
    localparam OP_SUB  = 5'b00010;
    localparam OP_MUL  = 5'b00011;
    localparam OP_MAX  = 5'b00100;
    localparam OP_MIN  = 5'b00101;
    localparam OP_RELU = 5'b00110;
    localparam OP_SIGM = 5'b00111;
    localparam OP_TANH = 5'b01000;
    
    // 功能单元输出
    wire [DATA_WIDTH-1:0] alu_out [0:VECTOR_WIDTH-1];
    wire [DATA_WIDTH-1:0] act_out [0:VECTOR_WIDTH-1];
    
    // SIMD ALU阵列
    genvar i;
    generate
        for (i = 0; i < VECTOR_WIDTH; i = i + 1) begin : simd_lane
            // 算术逻辑单元
            VectorALU #(.DATA_WIDTH(DATA_WIDTH)) alu_inst (
                .a(vec_a[i]),
                .b(vec_b[i]),
                .op(opcode[2:0]),
                .result(alu_out[i])
            );
            
            // 激活函数单元
            ActivationUnit #(.DATA_WIDTH(DATA_WIDTH)) act_inst (
                .data_in(vec_a[i]),
                .func_sel(opcode[4:3]),
                .data_out(act_out[i])
            );
        end
    endgenerate
    
    // 结果选择和流水线控制
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 0;
        end else if (execute) begin
            case (opcode)
                OP_ADD, OP_SUB, OP_MUL, OP_MAX, OP_MIN: begin
                    for (int j = 0; j < VECTOR_WIDTH; j = j + 1) begin
                        vec_result[j] <= alu_out[j];
                    end
                end
                OP_RELU, OP_SIGM, OP_TANH: begin
                    for (int j = 0; j < VECTOR_WIDTH; j = j + 1) begin
                        vec_result[j] <= act_out[j];
                    end
                end
            endcase
            done <= 1;
        end else begin
            done <= 0;
        end
    end
endmodule
```

### 4.3.2 特殊功能单元

| 功能单元 | 操作 | 实现方式 | 硬件成本 |
|---------|------|---------|----------|
| ReLU单元 | max(0, x) | 比较器+选择器 | 极低 |
| 池化单元 | max/avg pooling | 比较树/加法树 | 低 |
| LUT单元 | sigmoid/tanh | 查找表+插值 | 中等 |
| 归一化单元 | batch/layer norm | 乘法器+移位器 | 高 |

### 4.3.3 TPU Softmax 实现深度解析

Softmax是神经网络中的关键操作，尤其在注意力机制中。TPU采用了软硬件协同的优化策略，实现了极高效的Softmax计算。

#### TPU Softmax 实现架构

**1. 算法优化：数值稳定的 Log-Sum-Exp**

```python
// 标准Softmax容易溢出：
// softmax(x_i) = exp(x_i) / Σ exp(x_j)

// TPU采用的数值稳定版本：
// softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))

// 计算步骤：
1. max_val = max(x)           // 并行规约找最大值
2. x_shifted = x - max_val    // 广播减法
3. exp_val = exp(x_shifted)   // 硬件加速指数运算
4. sum_exp = sum(exp_val)     // 并行规约求和
5. result = exp_val / sum_exp // 逐元素除法（转为乘法）
```

**2. 硬件实现：VPU（向量处理单元）**

| 计算步骤 | 硬件单元 | 优化技术 | 执行时间 |
|---------|---------|---------|----------|
| 寻找最大值 | VPU并行规约单元 | 树状比较器网络 | O(log N) |
| 广播减法 | VPU SIMD单元 | 标量广播+向量减法 | O(1) |
| 指数运算 | 专用SFU（特殊功能单元） | 硬件LUT+多项式插值 | O(1) |
| 求和操作 | VPU并行规约单元 | 树状加法器网络 | O(log N) |
| 逐元素除法 | VPU乘法单元 | 倒数转乘法 | O(1) |

**3. 关键硬件优化：SFU（特殊功能单元）**

```verilog
// TPU SFU 指数运算实现（简化示意）
module ExponentialSFU #(
    parameter DATA_WIDTH = 16,  // FP16
    parameter LUT_DEPTH = 256   // 查找表深度
)(
    input wire [DATA_WIDTH-1:0] x,
    output wire [DATA_WIDTH-1:0] exp_x
);
    // 步骤1：范围检测和饱和处理
    wire in_range = (x > -10.0) && (x < 10.0);
    
    // 步骤2：分解 x = n*ln(2) + r，其中 |r| < ln(2)/2
    wire [7:0] n;
    wire [DATA_WIDTH-1:0] r;
    
    // 步骤3：查找表获取 exp(r) 的初值
    wire [DATA_WIDTH-1:0] exp_r_lut;
    LUT_256x16 exp_lut(.addr(r[15:8]), .data(exp_r_lut));
    
    // 步骤4：二次多项式修正
    // exp(r) ≈ exp_r_lut * (1 + r_frac + 0.5*r_frac²)
    wire [DATA_WIDTH-1:0] correction;
    
    // 步骤5：重构结果 exp(x) = 2^n * exp(r)
    wire [DATA_WIDTH-1:0] result = shift_left(exp_r_corrected, n);
    
    assign exp_x = in_range ? result : 
                   (x > 10.0) ? FP16_MAX : FP16_MIN;
endmodule
```

**4. 内存优化：算子融合**
- **数据局部性：**Softmax输入通常来自前一层的矩阵乘法（MXU输出），直接流向VPU
- **片上计算：**整个Softmax过程在片上SRAM完成，避免HBM访问
- **流水线优化：**max、exp、sum等操作流水线化，隐藏延迟
- **批处理：**多个序列的Softmax可以共享规约树硬件

**5. XLA编译器优化**
```python
// XLA识别并融合的Softmax模式
// 输入：用户代码
y = tf.nn.softmax(logits, axis=-1)

// XLA编译后：融合的TPU指令序列
TPU_VPU_MAX      vr1, logits, axis=-1    // 找最大值
TPU_VPU_SUB      vr2, logits, vr1        // 减最大值
TPU_SFU_EXP      vr3, vr2                // 硬件指数
TPU_VPU_SUM      vr4, vr3, axis=-1       // 求和
TPU_VPU_RECIP    vr5, vr4                // 倒数
TPU_VPU_MUL      output, vr3, vr5        // 乘法归一化
```

**性能对比：**

| 处理器 | Softmax实现 | 1M元素耗时 | 能效比 |
|--------|-------------|-------------|----------|
| CPU (AVX-512) | 软件循环+数学库 | ~10ms | 基准 |
| GPU (CUDA) | Warp级规约+共享内存 | ~0.5ms | 10x |
| TPU v4 | VPU硬件+SFU+融合 | ~0.05ms | 100x |

## 4.4 特殊计算单元

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

```verilog
// 4x4x4 Tensor Core实现
// 计算 D = A×B + C，其中A、B、C、D都是4×4矩阵
module TensorCore_4x4x4 #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    // 输入矩阵（扁平化表示）
    input wire [DATA_WIDTH-1:0] mat_a [0:15],  // 4x4矩阵A
    input wire [DATA_WIDTH-1:0] mat_b [0:15],  // 4x4矩阵B
    input wire [ACC_WIDTH-1:0] mat_c [0:15],   // 4x4矩阵C（累加）
    
    // 输出矩阵
    output reg [ACC_WIDTH-1:0] mat_d [0:15],   // 4x4结果矩阵D
    output reg valid
);

    // 内部信号
    wire [ACC_WIDTH-1:0] dot_products [0:15];
    
    // 生成16个点积计算单元
    genvar i, j, k;
    generate
        for (i = 0; i < 4; i = i + 1) begin : row
            for (j = 0; j < 4; j = j + 1) begin : col
                // 计算D[i][j] = sum(A[i][k] * B[k][j]) + C[i][j]
                wire [2*DATA_WIDTH-1:0] products [0:3];
                wire [ACC_WIDTH-1:0] sum;
                
                // 4个并行乘法器
                for (k = 0; k < 4; k = k + 1) begin : mult
                    assign products[k] = mat_a[i*4+k] * mat_b[k*4+j];
                end
                
                // 加法树
                assign sum = mat_c[i*4+j] + 
                           {{(ACC_WIDTH-2*DATA_WIDTH){products[0][2*DATA_WIDTH-1]}}, products[0]} +
                           {{(ACC_WIDTH-2*DATA_WIDTH){products[1][2*DATA_WIDTH-1]}}, products[1]} +
                           {{(ACC_WIDTH-2*DATA_WIDTH){products[2][2*DATA_WIDTH-1]}}, products[2]} +
                           {{(ACC_WIDTH-2*DATA_WIDTH){products[3][2*DATA_WIDTH-1]}}, products[3]};
                
                assign dot_products[i*4+j] = sum;
            end
        end
    endgenerate
    
    // 寄存输出
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid <= 0;
        end else if (enable) begin
            for (int idx = 0; idx < 16; idx = idx + 1) begin
                mat_d[idx] <= dot_products[idx];
            end
            valid <= 1;
        end else begin
            valid <= 0;
        end
    end
endmodule
```

### 4.4.2 稀疏计算支持

支持结构化稀疏（如2:4稀疏）可以显著提升有效计算吞吐量。

```verilog
// 2:4结构化稀疏MAC单元
// 每4个权重中有2个非零值
module SparseMACUnit_2in4 #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    // 稀疏权重输入（2个非零值）
    input wire [DATA_WIDTH-1:0] weight_values [0:1],  // 非零权重值
    input wire [1:0] weight_indices [0:1],            // 权重位置索引(0-3)
    
    // 4个激活值输入
    input wire [DATA_WIDTH-1:0] activations [0:3],
    
    // 累加输入输出
    input wire [ACC_WIDTH-1:0] psum_in,
    output reg [ACC_WIDTH-1:0] psum_out,
    output reg valid
);

    // 选择对应的激活值并计算
    wire [DATA_WIDTH-1:0] selected_acts [0:1];
    wire [2*DATA_WIDTH-1:0] products [0:1];
    wire [ACC_WIDTH-1:0] sum;
    
    // 根据索引选择激活值
    assign selected_acts[0] = activations[weight_indices[0]];
    assign selected_acts[1] = activations[weight_indices[1]];
    
    // 计算两个乘积
    assign products[0] = selected_acts[0] * weight_values[0];
    assign products[1] = selected_acts[1] * weight_values[1];
    
    // 累加
    assign sum = psum_in + 
                {{(ACC_WIDTH-2*DATA_WIDTH){products[0][2*DATA_WIDTH-1]}}, products[0]} +
                {{(ACC_WIDTH-2*DATA_WIDTH){products[1][2*DATA_WIDTH-1]}}, products[1]};
    
    // 寄存输出
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            psum_out <= 0;
            valid <= 0;
        end else if (enable) begin
            psum_out <= sum;
            valid <= 1;
        end else begin
            valid <= 0;
        end
    end
endmodule
```