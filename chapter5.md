# 第5章：存储系统设计

存储系统是NPU性能的关键瓶颈之一。本章深入探讨NPU片上存储系统的架构与设计要点，包括SRAM设计、Memory Banking策略、数据预取机制、缓存一致性、DMA设计以及内存压缩技术。

在现代NPU设计中，有一个残酷的事实：**数据搬运的能耗是计算的10-100倍**。从外部DRAM读取一个32位数据需要约640pJ的能量，而执行一次32位乘法仅需3.1pJ（在45nm工艺下）。这意味着，如果我们不能有效地管理数据移动，即使拥有世界上最快的计算单元也毫无意义。这就是为什么顶级NPU设计团队会花费超过50%的精力在存储系统优化上。

本章将揭示NPU存储系统设计的艺术与科学。我们将从片上SRAM的物理设计开始，探讨如何通过巧妙的Banking策略实现高带宽访问，如何设计智能的预取机制来隐藏访存延迟，以及如何通过压缩技术在有限的片上存储中塞入更多数据。更重要的是，我们将学习如何为脉动阵列这样的规则计算模式设计最优的存储层次结构。通过本章的学习，你将掌握设计高效NPU存储系统的核心技术，理解为什么Google TPU要配备如此巨大的片上存储，以及为什么NVIDIA在每一代GPU中都在不断增加缓存容量。

## <a name="51"></a>5.1 片上SRAM设计

片上SRAM是NPU存储层次结构的核心，为计算单元提供超低延迟、超高带宽的数据访问。

### CPU时代的Scratchpad记忆

在NPU大规模采用Scratchpad内存之前，CPU领域已经有了丰富的探索历史：

> **CPU Scratchpad的历史演进**
> - **IBM Cell BE (2005)：** PlayStation 3的处理器，每个SPE（协处理器）配备256KB的本地存储（Local Store），完全由软件管理。这是最早的大规模商用Scratchpad设计。
> - **TI DSP系列：** 德州仪器的DSP从C6000系列开始就采用了L1P（程序）和L1D（数据）分离的Scratchpad设计，专门用于信号处理的确定性延迟。
> - **ARM TCM (Tightly Coupled Memory)：** ARM Cortex-R系列处理器的标配，提供单周期访问延迟，广泛用于汽车电子和实时控制。
> - **Intel Xeon Phi：** 每个核心配备512KB的本地内存，可配置为缓存或Scratchpad模式。

NPU从这些先驱中学到的关键经验：
- **软件管理的优势：** 虽然增加了编程复杂度，但能实现近乎完美的数据局部性
- **分层设计的必要性：** 单一级别的Scratchpad无法满足所有需求
- **DMA的关键作用：** 高效的DMA引擎是Scratchpad成功的前提

### 5.1.1 SRAM设计权衡

SRAM设计就像建造一个高效的仓库系统——容量、速度、成本之间存在着根本性的权衡。理解这些权衡是设计高效存储系统的基础。

```
SRAM设计的关键权衡：

1. 容量 vs. 面积/功耗
   - SRAM面积密度：~0.2 MB/mm² (7nm工艺)
   - 静态功耗：~1mW/MB
   - 动态功耗：与访问频率成正比

2. 端口设计
   - 单端口：面积最小，但限制并行访问
   - 真双端口(1R1W)：面积增加~70%，支持一个读操作和一个写操作同时进行
   - 多端口(nRmW)：面积随端口数超线性增长

3. 访问延迟
   - 容量增大 → 延迟增加（解码器、字线、位线延迟）
   - 典型延迟：32KB ~1 cycle, 256KB ~2-3 cycles
```

#### 深入理解SRAM的物理限制

**1. 为什么SRAM这么"贵"？**

一个SRAM单元需要6个晶体管（6T），而DRAM只需要1个晶体管+1个电容（1T1C）。这意味着：
- 相同面积下，DRAM容量是SRAM的6-8倍
- 但SRAM速度快10-100倍，且不需要刷新
- 这就像跑车vs货车的权衡——速度与容量不可兼得

**2. 多端口的代价**

每增加一个端口，需要：
- 额外的字线（Word Line）和位线（Bit Line）
- 更复杂的仲裁逻辑
- 更大的单元面积（每个端口需要2个额外晶体管）

> **设计陷阱：盲目追求多端口**
> 
> 很多设计师认为端口越多越好，但实际上：
> - 4端口SRAM的面积是单端口的3-4倍
> - 大多数访问模式并不需要真正的多端口
> - 通过Banking和时分复用，往往能达到类似效果

#### 现代SRAM创新技术

**1. 近阈值电压（Near-Threshold）SRAM：**
通过降低工作电压接近晶体管阈值电压，可以大幅降低功耗（减少50-70%），但代价是速度降低和稳定性挑战。适用于边缘AI设备。

**2. 基于MRAM/ReRAM的"类SRAM"：**
新型非易失性存储器正在模糊SRAM和存储的界限。例如，台积电的22nm eMRAM已经可以提供接近SRAM的速度，但密度提升3-4倍。

**3. 3D SRAM：**
将SRAM堆叠在逻辑电路上方，通过TSV（硅通孔）连接。这种技术已经在某些高端GPU中使用，可以将存储密度提升2-3倍。

### 5.1.2 多级存储层次

**典型的三级存储层次设计：**
- **L0级：PE本地寄存器文件** - 16个256位寄存器，单周期访问
- **L1级：PE集群共享缓存** - 64KB容量，2-3周期延迟
- **L2级：全局共享缓存** - MB级容量，5-10周期延迟

**L0寄存器文件的关键设计要点：**
1. **流水线设计**：写操作流水线化，减少关键路径延迟
2. **旁路逻辑**：当读写地址相同时，直接转发写入数据，避免数据冒险
3. **参数化设计**：深度和宽度可配置，适应不同PE架构需求
4. **同步复位**：确保初始状态可控

**L0寄存器文件的Chisel实现特点：**
- 使用`RegNext`实现流水线寄存器，自动处理时序
- `Mux`选择器实现旁路逻辑，简洁高效
- `VecInit`创建寄存器数组，支持参数化深度
- Chisel的类型推断减少了代码量，提高可读性

**L1集群缓存设计要点：**
- **容量**：64KB，平衡面积与性能
- **端口数**：4个独立端口，支持多PE并行访问
- **数据宽度**：256位，匹配矩阵运算的数据粒度
- **访问仲裁**：需要处理多端口冲突，保证公平性

### 5.1.3 特殊SRAM结构

**转置SRAM的设计动机与实现：**

转置SRAM是NPU中的特殊存储结构，专门优化矩阵转置操作：

1. **双模式访问**：
   - 行模式：按行读写，适合正常的矩阵运算
   - 列模式：按列读写，实现零开销转置

2. **硬件实现要点**：
   - 使用二维存储阵列 `mem[ROWS][COLS]`
   - 通过`row_mode`信号切换访问模式
   - `generate`块并行处理所有列的读写

3. **性能优势**：
   - 转置操作从O(n²)降至O(1)
   - 消除了数据搬运的功耗开销
   - 特别适合Transformer中的注意力计算

## <a name="52"></a>5.2 Memory Banking策略

Memory Banking通过将SRAM划分为多个独立的Bank，实现并行访问，成倍提升有效带宽。

Banking的本质是"分而治之"——将一个大的存储器拆分成多个可以独立访问的小存储器。这就像将一个大超市拆分成多个专柜，顾客可以同时在不同专柜购物，而不会相互阻塞。

### Banking的数学原理

理想情况下，N个Bank可以提供N倍的带宽。但现实中，由于访问模式的不均匀性，会产生Bank冲突。关键是找到最优的Bank数量和地址映射策略。

> **Bank数量选择的黄金法则**
> - **质数法则：** Bank数量选择质数（如7、11、13），可以减少规则访问模式的冲突
> - **2的幂次：** 便于硬件实现（简单的位操作），但容易产生冲突
> - **混合策略：** 2的幂次×质数（如8×3=24），平衡实现复杂度和冲突率

### 5.2.1 Bank冲突分析

```
Bank冲突的主要场景：

1. 卷积中的步长访问
   - 3×3卷积，stride=2时的访问模式
   - Bank数量需要考虑GCD(stride, bank_num)

2. 矩阵转置访问
   - 行访问：连续地址
   - 列访问：地址间隔为矩阵宽度

3. 稀疏访问模式
   - 不规则的访问地址
   - 需要动态仲裁机制
```

#### 真实案例：卷积访问的Bank冲突

让我们分析一个3×3卷积在8-Bank系统中的访问模式：

```
假设输入特征图宽度W=64，使用简单的模8映射：
Bank_ID = Address % 8

第一行访问地址：[0, 1, 2]  → Banks: [0, 1, 2]  ✓ 无冲突
第二行访问地址：[64, 65, 66] → Banks: [0, 1, 2]  ✗ 全部冲突！
第三行访问地址：[128, 129, 130] → Banks: [0, 1, 2]  ✗ 全部冲突！

问题：所有访问都集中在前3个Bank，其他5个Bank完全空闲！
```

**解决方案：XOR-Based Banking**

```
// 改进的地址映射
Bank_ID = (Address[2:0]) XOR (Address[8:6])

这种映射将高位地址位混入Bank选择，打散了规则的访问模式
```

> **实战经验：Banking不是越多越好**
> - Bank数量翻倍，仲裁器复杂度增加4倍
> - 过多的Bank会导致每个Bank容量太小，增加miss率
> - 典型的sweet spot：8-16个Bank

### 5.2.2 地址映射策略

```verilog
// 优化的多Bank SRAM控制器 - SystemVerilog版本（带流水线和仲裁）
module MultiBank_SRAM #(
    parameter NUM_BANKS = 8,
    parameter BANK_SIZE = 8192,     // 每个Bank 8KB
    parameter DATA_WIDTH = 256,
    parameter ADDR_WIDTH = 16
)(
    input wire clk,
    input wire rst_n,
    
    // 请求接口（支持多个并行请求）
    input wire [3:0] req_valid,
    input wire [ADDR_WIDTH-1:0] req_addr [3:0],
    input wire [3:0] req_wr,
    input wire [DATA_WIDTH-1:0] req_wdata [3:0],
    output reg [3:0] req_ready,
    output reg [DATA_WIDTH-1:0] resp_data [3:0],
    output reg [3:0] resp_valid
);

    // 第一级流水线：地址解码
    reg [3:0] req_valid_r1;
    reg [2:0] bank_id_r1 [3:0];
    reg [12:0] bank_addr_r1 [3:0];
    reg [3:0] req_wr_r1;
    reg [DATA_WIDTH-1:0] req_wdata_r1 [3:0];
    reg [3:0] req_id_r1;  // 请求者ID
    
    // 地址解码逻辑
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            req_valid_r1 <= 0;
            for (int i = 0; i < 4; i++) begin
                bank_id_r1[i] <= 0;
                bank_addr_r1[i] <= 0;
                req_wr_r1[i] <= 0;
                req_wdata_r1[i] <= 0;
                req_id_r1[i] <= i;
            end
        end else begin
            req_valid_r1 <= req_valid;
            for (int i = 0; i < 4; i++) begin
                // 交织映射：低位作为bank索引
                bank_id_r1[i] <= req_addr[i][2:0];
                bank_addr_r1[i] <= req_addr[i][ADDR_WIDTH-1:3];
                req_wr_r1[i] <= req_wr[i];
                req_wdata_r1[i] <= req_wdata[i];
            end
        end
    end
    
    // 第二级流水线：Bank仲裁
    reg [3:0] bank_grant_r2 [NUM_BANKS-1:0];
    reg [3:0] grant_id_r2 [NUM_BANKS-1:0];  // 被授权的请求者ID
    
    // 改进的仲裁逻辑（轮询优先级）
    reg [1:0] priority_ptr [NUM_BANKS-1:0];  // 每个Bank的优先级指针
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int j = 0; j < NUM_BANKS; j++) begin
                bank_grant_r2[j] <= 0;
                grant_id_r2[j] <= 0;
                priority_ptr[j] <= 0;
            end
        end else begin
            // 对每个Bank进行仲裁
            for (int j = 0; j < NUM_BANKS; j++) begin
                bank_grant_r2[j] <= 0;
                
                // 从优先级指针开始轮询
                for (int k = 0; k < 4; k++) begin
                    int req_idx = (priority_ptr[j] + k) % 4;
                    if (req_valid_r1[req_idx] && bank_id_r1[req_idx] == j && |bank_grant_r2[j] == 0) begin
                        bank_grant_r2[j][req_idx] <= 1;
                        grant_id_r2[j] <= req_idx;
                        priority_ptr[j] <= (req_idx + 1) % 4;  // 更新优先级
                    end
                end
            end
        end
    end
    
    // Bank SRAM实例和第三级流水线
    wire [DATA_WIDTH-1:0] bank_rdata [NUM_BANKS-1:0];
    reg [3:0] resp_valid_r3;
    reg [3:0] resp_id_r3 [NUM_BANKS-1:0];
    
    genvar i;
    generate
        for (i = 0; i < NUM_BANKS; i = i + 1) begin : bank_gen
            // 选择授权的请求
            wire bank_en = |bank_grant_r2[i];
            wire [3:0] grant_onehot = bank_grant_r2[i];
            wire [1:0] grant_idx = grant_id_r2[i];
            
            // Mux选择授权请求的信号
            wire bank_wr = req_wr_r1[grant_idx] & bank_en;
            wire [12:0] bank_addr = bank_addr_r1[grant_idx];
            wire [DATA_WIDTH-1:0] bank_wdata = req_wdata_r1[grant_idx];
            
            // Bank SRAM实例
            BankSRAM #(
                .SIZE(BANK_SIZE),
                .WIDTH(DATA_WIDTH)
            ) bank_inst (
                .clk(clk),
                .en(bank_en),
                .wr(bank_wr),
                .addr(bank_addr),
                .wdata(bank_wdata),
                .rdata(bank_rdata[i])
            );
            
            // 响应ID寄存
            always @(posedge clk) begin
                resp_id_r3[i] <= grant_id_r2[i];
            end
        end
    endgenerate
    
    // 第四级流水线：响应汇集
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            resp_valid <= 0;
            for (int i = 0; i < 4; i++)
                resp_data[i] <= 0;
        end else begin
            resp_valid <= 0;
            
            // 将Bank响应路由回请求者
            for (int j = 0; j < NUM_BANKS; j++) begin
                if (|bank_grant_r2[j]) begin
                    int resp_idx = resp_id_r3[j];
                    resp_data[resp_idx] <= bank_rdata[j];
                    resp_valid[resp_idx] <= 1;
                end
            end
        end
    end
    
    // Ready信号（考虑仲裁结果）
    always @(*) begin
        req_ready = 4'b1111;  // 默认都ready，实际使用时可根据Bank忙碌状态调整
    end
endmodule

// Bank SRAM模块
module BankSRAM #(
    parameter SIZE = 8192,
    parameter WIDTH = 256,
    parameter ADDR_WIDTH = 13
)(
    input wire clk,
    input wire en,
    input wire wr,
    input wire [ADDR_WIDTH-1:0] addr,
    input wire [WIDTH-1:0] wdata,
    output reg [WIDTH-1:0] rdata
);
    reg [WIDTH-1:0] mem [0:SIZE-1];
    
    always @(posedge clk) begin
        if (en) begin
            if (wr)
                mem[addr] <= wdata;
            else
                rdata <= mem[addr];
        end
    end
endmodule
```

Chisel版本的多Bank SRAM控制器：

```scala
import chisel3._
import chisel3.util._

class MultiBankSRAM(numBanks: Int = 8, bankSize: Int = 8192, 
                    dataWidth: Int = 256, numPorts: Int = 4) extends Module {
    val addrWidth = log2Ceil(numBanks * bankSize)
    val bankAddrWidth = log2Ceil(bankSize)
    
    val io = IO(new Bundle {
        val req = Vec(numPorts, new Bundle {
            val valid = Input(Bool())
            val addr = Input(UInt(addrWidth.W))
            val wr = Input(Bool())
            val wdata = Input(UInt(dataWidth.W))
            val ready = Output(Bool())
        })
        val resp = Vec(numPorts, new Bundle {
            val data = Output(UInt(dataWidth.W))
            val valid = Output(Bool())
        })
    })
    
    // 第一级流水线：地址解码
    val reqValidR1 = RegNext(VecInit(io.req.map(_.valid)))
    val bankIdR1 = io.req.map(r => RegNext(r.addr(log2Ceil(numBanks)-1, 0)))
    val bankAddrR1 = io.req.map(r => RegNext(r.addr >> log2Ceil(numBanks)))
    val reqWrR1 = RegNext(VecInit(io.req.map(_.wr)))
    val reqWdataR1 = RegNext(VecInit(io.req.map(_.wdata)))
    
    // 仲裁器（每个Bank一个）
    val arbiters = Seq.fill(numBanks)(Module(new RRArbiter(numPorts)))
    val banks = Seq.fill(numBanks)(Module(new BankSRAM(bankSize, dataWidth)))
    
    // 连接请求到仲裁器
    for (i <- 0 until numPorts) {
        for (j <- 0 until numBanks) {
            arbiters(j).io.req(i).valid := reqValidR1(i) && (bankIdR1(i) === j.U)
            arbiters(j).io.req(i).bits := Cat(reqWdataR1(i), bankAddrR1(i), reqWrR1(i))
        }
    }
    
    // 连接仲裁器到Bank
    for (j <- 0 until numBanks) {
        banks(j).io.en := arbiters(j).io.chosen.valid
        banks(j).io.wr := arbiters(j).io.chosen.bits(0)
        banks(j).io.addr := arbiters(j).io.chosen.bits(bankAddrWidth, 1)
        banks(j).io.wdata := arbiters(j).io.chosen.bits >> (bankAddrWidth + 1)
    }
    
    // 响应路由
    for (i <- 0 until numPorts) {
        io.resp(i).valid := RegNext(arbiters.map(a => a.io.grant(i)).reduce(_ || _))
        io.resp(i).data := RegNext(MuxCase(0.U, 
            banks.zipWithIndex.map { case (bank, j) => 
                (arbiters(j).io.grant(i) -> bank.io.rdata)
            }
        ))
        io.req(i).ready := true.B  // 简化：始终ready
    }
}

// 轮询仲裁器
class RRArbiter(n: Int) extends Module {
    val io = IO(new Bundle {
        val req = Vec(n, Flipped(Valid(UInt())))
        val chosen = Valid(UInt())
        val grant = Vec(n, Output(Bool()))
    })
    
    val priority = RegInit(0.U(log2Ceil(n).W))
    
    // 轮询逻辑
    val reqVec = VecInit(io.req.map(_.valid))
    val shiftReq = VecInit((0 until n).map(i => reqVec((i + priority) % n)))
    val shiftGrant = PriorityEncoderOH(shiftReq)
    
    // 输出
    io.grant := VecInit((0 until n).map(i => shiftGrant((i - priority + n) % n)))
    io.chosen.valid := reqVec.reduce(_ || _)
    io.chosen.bits := Mux1H(io.grant, io.req.map(_.bits))
    
    // 更新优先级
    when(io.chosen.valid) {
        priority := (priority + PriorityEncoder(io.grant) + 1.U) % n.U
    }
}

// 专用于卷积的Bank映射
module ConvBankMapping #(
    parameter BANK_BITS = 3,        // 8个Bank
    parameter CHANNEL_BITS = 6      // 64个通道
)(
    input wire [15:0] h_idx,        // Height坐标
    input wire [15:0] w_idx,        // Width坐标
    input wire [CHANNEL_BITS-1:0] c_idx,  // Channel坐标
    output wire [BANK_BITS-1:0] bank_id,
    output wire [15:0] bank_offset
);
    // 斜对角映射，避免3×3卷积的Bank冲突
    wire [BANK_BITS-1:0] skew;
    assign skew = (h_idx + w_idx) & ((1 << BANK_BITS) - 1);
    assign bank_id = (c_idx[BANK_BITS-1:0] + skew) & ((1 << BANK_BITS) - 1);
    
    // Bank内偏移地址
    assign bank_offset = {c_idx[CHANNEL_BITS-1:BANK_BITS], h_idx[7:0], w_idx[7:0]};
endmodule
```

### 5.2.3 Bank冲突解决

**Bank冲突解决的硬件实现：**

带冲突缓冲的Bank访问调度器是处理多请求者竞争的关键组件：

1. **请求队列管理**：
   - 每个请求者配备4深度的FIFO队列
   - 支持16个并发请求者
   - 使用头尾指针管理循环队列

2. **冲突检测算法**：
   - 每周期扫描所有队列头部
   - 检测目标Bank冲突
   - 选择无冲突的请求发送

3. **调度策略**：
   - 优先级基于队列深度
   - 避免饥饿：长时间等待的请求提升优先级
   - 支持QoS（服务质量）级别

## <a name="53"></a>5.3 数据预取机制

数据预取通过提前将数据从DRAM加载到片上SRAM，隐藏内存访问延迟，是提升NPU性能的关键技术。

预取的艺术在于"料敌于先"——在计算单元需要数据之前，就已经将数据准备好。这就像一个优秀的助手，总能在老板需要文件之前就放在桌上。

### 预取的三大挑战

> **挑战与解决方案**
> - **挑战1：预取太早** → 数据被驱逐出缓存 → 解决：基于计算进度的动态预取
> - **挑战2：预取太晚** → 无法隐藏延迟 → 解决：多级预取队列
> - **挑战3：预取错误数据** → 浪费带宽和功耗 → 解决：基于编译器提示的确定性预取

### NPU预取 vs CPU预取

NPU的预取相比CPU有独特优势：
- **确定性访问模式：** 神经网络的数据访问是完全可预测的，不像CPU的通用程序
- **编译时优化：** 可以在编译时生成精确的预取指令
- **双缓冲友好：** 自然适合乒乓缓冲的设计模式

### 5.3.1 预取策略

```
预取机制的核心要素：

1. 预取时机
   - 基于计算进度的预取
   - 基于地址模式的预取
   - 软件控制的显式预取

2. 预取粒度
   - 细粒度：单个Tile (如16×16)
   - 粗粒度：整个Feature Map
   - 自适应粒度：根据可用空间动态调整

3. 预取深度
   - Double Buffering: 计算当前数据时预取下一批
   - Triple Buffering: 更深的流水线，容忍更大延迟
```

### 5.3.2 硬件预取引擎

**智能预取引擎的设计要点：**

该预取引擎采用4级状态机和自适应预取算法：

1. **状态机设计**：
   - **IDLE**：等待预取启动
   - **MONITOR**：监控计算进度，决策预取时机
   - **ISSUE_REQ**：发出DRAM读取请求
   - **WAIT_RESP**：等待DRAM响应

2. **预取距离算法**：
   - 基于计算进度动态调整
   - 公式：`prefetch_distance = queue_count × 16`
   - 确保数据在需要前到达

3. **队列管理**：
   - 4深度预取队列
   - 循环缓冲区设计
   - 避免过度预取造成内存浪费

4. **地址生成**：
   - 基于stride的规则访问模式
   - 支持2D/3D数据结构
   - 编译器指导的预取提示

**双缓冲预取控制器设计：**

双缓冲是NPU中最常用的隐藏访存延迟技术：

1. **缓冲区配置**：
   - 两个16KB缓冲区（A和B）
   - 256位数据宽度，匹配计算单元
   - 乒乓切换机制

2. **工作流程**：
   - 缓冲区A供计算时，缓冲区B从DRAM预取
   - 计算完成后切换角色
   - 通过`buffer_sel`信号控制当前活跃缓冲区

3. **同步机制**：
   - `buffer_ready`标志数据就绪
   - 确保计算不会访问未完成的缓冲区
   - 支持变长预取（prefetch_length）
    
    // 缓冲区实例
    wire [DATA_WIDTH-1:0] buffer_rdata [1:0];
    
    genvar i;
    generate
        for (i = 0; i < 2; i = i + 1) begin
            SimpleDualPortRAM #(
                .DEPTH(BUFFER_SIZE/32),
                .WIDTH(DATA_WIDTH)
            ) buffer (
                .clk(clk),
                .wr_en(dram_resp_valid && (buffer_sel != i)),
                .wr_addr(buffer_write_addr[i]),
                .wr_data(dram_resp_data),
                .rd_addr(compute_addr),
                .rd_data(buffer_rdata[i])
            );
        end
    endgenerate
    
    // 计算接口
    assign compute_data = buffer_rdata[buffer_sel];
    assign compute_ready = buffer_ready[buffer_sel];
    
    // 缓冲切换逻辑
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            buffer_sel <= 0;
        end else if (/* 当前buffer计算完成 && 另一个buffer预取完成 */) begin
            buffer_sel <= ~buffer_sel;
        end
    end
endmodule
```

### 5.3.3 软件控制预取

**软件控制预取的关键设计：**

1. **预取指令格式**：
   - 8位操作码指定PREFETCH指令
   - 4位缓冲区ID，支持16个独立缓冲区
   - 访问模式：线性、2D、3D数据布局
   - 优先级控制，确保关键数据优先

2. **预取描述符结构**：
   - 源/目标地址对
   - 三维数据描述：size和stride
   - 支持复杂的张量访问模式
   - 适合神经网络的NHWC/NCHW布局

3. **卷积层预取策略**：
   - **权重预取**：一次性加载全部权重（复用率高）
   - **输入Tile化**：按Tile大小分块预取
   - **双缓冲调度**：计算当前Tile时预取下一Tile
   - **同步机制**：`wait_prefetch_complete`确保数据就绪

4. **性能优化要点**：
   - 预取粒度与Tile大小匹配
   - 预取时机基于计算进度
   - 多级预取引擎并行工作

## <a name="54"></a>5.4 缓存一致性

在多核NPU系统中，缓存一致性确保不同核心看到的数据是一致的，这对正确性至关重要。

### 5.4.1 NPU缓存一致性挑战

```
NPU缓存一致性的特点：

1. 软件管理为主
   - 神经网络计算流程确定
   - 编译器可以精确分析数据依赖
   - 显式同步点插入

2. 简化的硬件支持
   - 基本的Cache刷新/失效指令
   - DMA与Cache的协同
   - 全局同步屏障

3. 常见场景
   - 多核协同计算大矩阵乘法
   - Pipeline并行中的数据传递
   - 模型参数的广播更新
```

### 5.4.2 软件管理的缓存一致性

**缓存控制单元的硬件实现：**

NPU缓存控制单元主要提供软件管理的一致性支持：

1. **核心操作**：
   - **Flush**：将脏数据写回内存并失效
   - **Invalidate**：失效缓存行，不写回
   - **Clean**：写回脏数据但保留在缓存中

2. **全局屏障同步机制**：
   - 4状态机：IDLE → WAIT_ALL → SYNC → RELEASE
   - 等待所有核心到达屏障点
   - 统一执行缓存同步操作
   - 同时释放所有等待核心

3. **多核协调**：
   - 每个核心独立的命令端口
   - 支持地址范围操作
   - 并行处理多个核心请求

4. **性能优化**：
   - 流水线化命令处理
   - 批量缓存操作合并
   - 避免不必要的全局同步

**软件管理一致性协议的设计要点：**

1. **目录结构**：
   - 1024个目录项，每项覆癖1KB地址空间
   - `owner_vector`：记录数据的独占拥有者
   - `sharer_vector`：记录所有共享者
   - 支持4核心系统

2. **操作类型**：
   - **Read**：共享读取，加入sharer_vector
   - **Write**：写操作，需要失效其他共享者
   - **Exclusive**：独占访问，获得所有权

3. **地址映射策略**：
   - 使用地址位[19:10]作为目录索引
   - 1KB粒度的跟踪
   - 平衡目录开销和精度
    
    // 处理核心请求
    always @(posedge clk) begin
        for (int i = 0; i < 4; i++) begin
            if (core_req[i]) begin
                reg [9:0] idx = addr_to_dir_idx(core_addr[i]);
                
                case (core_op[i])
                    2'b00: begin // Read
                        // 添加到共享者列表
                        sharer_vector[idx][i] <= 1;
                    end
                    
                    2'b01: begin // Write
                        // 需要独占访问
                        if (sharer_vector[idx] != 0 && sharer_vector[idx] != (1 << i)) begin
                            // 失效其他共享者
                            invalidate_req <= sharer_vector[idx] & ~(1 << i);
                        end
                        owner_vector[idx] <= (1 << i);
                        sharer_vector[idx] <= (1 << i);
                    end
                    
                    2'b10: begin // Exclusive (for RMW)
                        // 获取独占权限
                        invalidate_req <= sharer_vector[idx] & ~(1 << i);
                        if (owner_vector[idx] != 0 && owner_vector[idx] != (1 << i)) begin
                            writeback_req <= owner_vector[idx];
                        end
                        owner_vector[idx] <= (1 << i);
                        sharer_vector[idx] <= (1 << i);
                    end
                endcase
            end
        end
    end
endmodule
```

## <a name="55"></a>5.5 DMA设计

DMA（Direct Memory Access）是NPU中实现高效数据传输的关键组件，它允许数据在不占用处理器的情况下在内存之间移动。

### 5.5.1 多通道DMA架构

**高性能DMA引擎的架构设计：**

现代NPU的DMA引擎通常采用8通道并行架构：

1. **多通道设计优势**：
   - 8个独立通道，支持并发传输
   - 每通道512位数据宽度
   - 最大突发64传输，优化DDR带宽利用
   - 40位地址支持大内存系统

2. **AXI接口特性**：
   - 读写通道分离，全双工传输
   - 支持Outstanding事务
   - 可配置突发长度和大小
   - 内置流量控制

3. **描述符管理**：
   - 描述符格式包含：源/目标地址、字节数、步长等
   - 支持2D传输模式（行列步长）
   - 链式描述符支持复杂传输序列
   - 标志位控制传输模式

4. **通道仲裁机制**：
   - 优先级仲裁器处理多通道竞争
   - 防止单通道垄断带宽
   - 支持QoS保证关键传输

5. **性能优化特性**：
   - 数据预取缓冲
   - 写合并优化
   - 地址对齐检查
   - 错误处理机制

### 5.5.2 2D/3D传输模式

**支持复杂数据布局的DMA通道设计：**

现代NPU的DMA需要支持多种数据布局传输模式：

1. **传输模式类型**：
   - **1D模式**：连续内存块传输
   - **2D模式**：矩阵数据，支持行步长（pitch）
   - **3D模式**：三维张量，支持片步长（slice pitch）
   - **Gather模式**：根据索引数组收集离散数据
   - **Scatter模式**：根据索引数组分散写入数据

2. **地址计算策略**：
   - 1D：`addr = base + offset`
   - 2D：`addr = base + y*pitch + x*width`
   - 3D：`addr = base + z*slice_pitch + y*pitch + x*width`
   - Gather/Scatter：`addr = base + index[i]*element_size`

3. **状态机设计**：
   - 9状态机：IDLE → PARSE_DESC → CALC_ADDR → ISSUE_READ → WAIT_READ → ISSUE_WRITE → WAIT_WRITE → UPDATE_ADDR → COMPLETE
   - 支持流水线操作
   - 错误处理和重试机制

4. **计数器管理**：
   - 三维计数器（x_count, y_count, z_count）
   - 自动换行换片逻辑
   - 边界检查和溢出保护

5. **性能优化**：
   - 地址预计算减少关键路径
   - 突发传输合并小请求
   - 支持地址对齐优化

**张量布局转换DMA的设计：**

张量布局转换是NPU中的重要功能，支持NCHW、NHWC等不同数据格式间的高效转换：

1. **布局转换原理**：
   - 支持最多4维张量（批次、通道、高度、宽度）
   - 维度排列可配置：src_layout和dst_layout定义维度顺序
   - 自动计算各维度的stride（步长）

2. **Stride计算算法**：
   - 对于每个维度，stride等于其后所有维度大小的乘积
   - 例如NCHW布局：C的stride = H×W，N的stride = C×H×W
   - 支持任意维度排列组合

3. **地址映射公式**：
   ```
   linear_addr = base + Σ(index[d] × stride[d] × element_size)
   ```
   - 多维坐标转换为线性地址
   - 支持不同数据类型（float32、int8等）

4. **迭代器设计**：
   - 多维索引自动递增
   - 支持边界检查和自动换行
   - 完成检测基于最高维度

5. **性能优化策略**：
   - 批量传输相邻元素
   - 缓存友好的访问模式
   - 支持并行多通道传输

## <a name="56"></a>5.6 内存压缩技术

内存压缩可以显著提高NPU的有效带宽，特别是对于稀疏或冗余的数据。

### 5.6.1 权重压缩

**结构化稀疏权重压缩器设计：**

结构化稀疏是现代NPU中常用的压缩技术，特别是2:4稀疏模式：

1. **2:4稀疏模式**：
   - 每4个权重中保留2个非零值
   - 压缩率50%，精度损失最小
   - 硬件加速器可直接支持

2. **压缩算法**：
   - 使用排序网络找出最大的N个值
   - 生成位掩码指示非零位置
   - 将零值剪枝，只存储非零值

3. **硬件实现要点**：
   - 并行比较器网络
   - 流水线化处理
   - 低延迟输出

4. **存储格式**：
   - 值数组：存储非零权重
   - 掩码数组：指示非零位置
   - 元数据：压缩比等信息

**权重量化压缩器设计：**

量化是将高精度权重转换为低位宽表示的技术：

1. **动态量化算法**：
   - 输入：16位浮点或定点数
   - 输出：4位整数（INT4）
   - 压缩率75%，显著减少存储和带宽

2. **量化参数计算**：
   - 找出权重的最大最小值
   - 计算scale：`(max-min)/(2^bits-1)`
   - 计算zero_point：`round(-min/scale)`

3. **量化公式**：
   ```
   quantized = round((weight - zero_point) / scale)
   dequantized = quantized * scale + zero_point
   ```

4. **硬件优化**：
   - 并行最大最小值查找
   - 流水线化量化计算
   - 支持批量处理
            
            // 计算量化参数
            scale = (max_val - min_val) >> OUT_WIDTH;
            zero_point = min_val;
            
            // 量化
            for (int i = 0; i < NUM_WEIGHTS; i++) begin
                reg signed [IN_WIDTH-1:0] shifted = weights_in[i] - zero_point;
                weights_out[i] = shifted / scale;
            end
            
            valid_out <= 1;
        end else begin
            valid_out <= 0;
        end
    end
endmodule
```

### 5.6.2 激活值压缩

**动态激活值压缩器设计：**

激活值压缩针对神经网络推理中的中间结果，采用多种压缩策略：

1. **支持的压缩模式**：
   - **无压缩**：原始数据直接传输
   - **RLE（游程编码）**：适合大量重复值（如ReLU后的零值）
   - **Bit-packing**：去除高位零值
   - **Delta编码**：存储相邻值的差异
   - **稀疏压缩**：只存储非零值及其索引

2. **RLE压缩算法**：
   - 格式：[长度(8bit)][值(16bit)]
   - 最大游程长度：255
   - 适合连续相同值的场景
   - 压缩比取决于数据分布

3. **稀疏压缩算法**：
   - 格式：[非零数(8bit)][索引1,值1][索引2,值2]...
   - 索引占8位，支持256个元素
   - 适合稀疏度>50%的数据
   - 压缩比：稀疏度/(1+索引开销)

4. **硬件实现特点**：
   - 单周期压缩决策
   - 固定大小输出缓冲（512位）
   - 元数据记录压缩类型
   - 支持流水线操作

5. **选择策略**：
   - ReLU层输出：优先RLE（大量零值）
   - 全连接层：稀疏压缩
   - 卷积层：根据稀疏度动态选择

### 5.6.3 自适应压缩策略

**自适应压缩选择器设计：**

自适应压缩根据数据特征动态选择最优压缩算法：

1. **数据特征分析**：
   - **稀疏度**：零值占比
   - **值域范围**：最大值-最小值
   - **重复模式**：游程长度统计
   - **唯一值数量**：用于哈夫曼编码

2. **压缩算法选择策略**：

   **权重压缩**：
   - Conv层：稀疏度>60%用2:4稀疏，否则INT8量化
   - FC层：稀疏度>70%用4:8稀疏，否则INT4量化
   - Attention层：保持高精度，使用INT16量化

   **激活值压缩**：
   - ReLU后：优先RLE（游程编码）
   - 稀疏度>50%：稀疏压缩
   - 其他：INT8动态量化

   **梯度压缩**：
   - 稀疏度>90%：Top-K稀疏（只传输最大的10%）
   - 小范围值：差分编码
   - 其他：FP16量化

3. **压缩比预测**：
   - INT4量化：8倍压缩
   - INT8量化：4倍压缩
   - 2:4稀疏：2倍压缩
   - RLE：1.5-6倍（取决于数据分布）
   - Top-K稀疏：10倍以上

4. **自适应机制**：
   - 采样窗口：64个样本
   - 统计分析：实时计算特征
   - 动态决策：基于历史数据
   - 参数调整：根据层类型优化

```verilog
// 简化的自适应压缩选择器框架
module AdaptiveCompressionSelector #(
    parameter DATA_WIDTH = 256,
    parameter SAMPLE_SIZE = 64
)(
    input wire clk,
    input wire rst_n,
    
    // 数据特征输入
    input wire [1:0] data_type, // 0: Weight, 1: Activation, 2: Gradient
    input wire [2:0] layer_type, // 0: Conv, 1: FC, 2: BN, 3: Attention
    
    // 数据采样输入
    input wire sample_valid,
    input wire [DATA_WIDTH-1:0] sample_data,
    
    // 压缩算法选择输出
    output reg [2:0] selected_algorithm,
    output reg [7:0] algorithm_params,
    output reg selection_done
);

    // 算法定义
    localparam NONE = 0, QUANTIZE = 1, RLE = 2, 
               SPARSE = 3, HUFFMAN = 4, DELTA = 5;
    
    // 统计信息
    reg [31:0] zero_count;
    reg [31:0] unique_values;
    reg [31:0] max_run_length;
    reg [31:0] value_range;
    reg signed [31:0] min_value, max_value;
    reg [31:0] sample_count;
    
    // 统计收集
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            zero_count <= 0;
            sample_count <= 0;
            min_value <= 32'h7FFFFFFF;
            max_value <= 32'h80000000;
        end else if (sample_valid && sample_count < SAMPLE_SIZE) begin
            sample_count <= sample_count + 1;
            
            // 统计零值
            for (int i = 0; i < DATA_WIDTH/32; i++) begin
                if (sample_data[i*32 +: 32] == 0) begin
                    zero_count <= zero_count + 1;
                end
                
                // 更新最大最小值
                signed [31:0] val = sample_data[i*32 +: 32];
                if (val < min_value) min_value <= val;
                if (val > max_value) max_value <= val;
            end
        end
    end
    
    // 算法选择逻辑
    always @(posedge clk) begin
        if (sample_count >= SAMPLE_SIZE && !selection_done) begin
            // 计算统计指标
            reg [15:0] sparsity = (zero_count * 100) / (sample_count * DATA_WIDTH/32);
            value_range = max_value - min_value;
            
            case (data_type)
                2'b00: begin // 权重
                    case (layer_type)
                        3'b000: begin // Conv层权重
                            if (sparsity > 60) begin
                                selected_algorithm <= SPARSE;
                                algorithm_params <= 8'h24; // 2:4稀疏
                            end else begin
                                selected_algorithm <= QUANTIZE;
                                algorithm_params <= 8'h08; // INT8量化
                            end
                        end
                        
                        3'b001: begin // FC层权重
                            // FC层通常稀疏性更高
                            if (sparsity > 70) begin
                                selected_algorithm <= SPARSE;
                                algorithm_params <= 8'h48; // 4:8稀疏
                            end else if (unique_values < 256) begin
                                selected_algorithm <= HUFFMAN;
                                algorithm_params <= 8'h00;
                            end else begin
                                selected_algorithm <= QUANTIZE;
                                algorithm_params <= 8'h04; // INT4量化
                            end
                        end
                        
                        3'b011: begin // Attention层权重
                            // Attention通常需要更高精度
                            selected_algorithm <= QUANTIZE;
                            algorithm_params <= 8'h10; // INT16量化
                        end
                    endcase
                end
                
                2'b01: begin // 激活值
                    if (layer_type == 3'b000 || layer_type == 3'b001) begin
                        // ReLU后激活值有大量零
                        if (sparsity > 50) begin
                            selected_algorithm <= RLE;
                            algorithm_params <= 8'hFF; // 最大游程255
                        end else begin
                            // 动态量化
                            selected_algorithm <= QUANTIZE;
                            algorithm_params <= 8'h88; // 动态INT8
                        end
                    end else if (layer_type == 3'b010) begin // BN层
                        // BN后数据分布较均匀
                        selected_algorithm <= DELTA;
                        algorithm_params <= 8'h01; // 一阶差分
                    end
                end
                
                2'b10: begin // 梯度
                    // 梯度通常很小且稀疏
                    if (sparsity > 80) begin
                        selected_algorithm <= SPARSE;
                        algorithm_params <= 8'h11; // 1:1稀疏（只传非零）
                    end else if (value_range < 65536) begin
                        // 小范围梯度用差分编码
                        selected_algorithm <= DELTA;
                        algorithm_params <= 8'h02; // 二阶差分
                    end else begin
                        selected_algorithm <= QUANTIZE;
                        algorithm_params <= 8'h10; // FP16量化
                    end
                end
            endcase
            
            selection_done <= 1;
        end
    end
    
    // 压缩比预测
    reg [15:0] predicted_ratio;
    always @(*) begin
        case (selected_algorithm)
            QUANTIZE: begin
                case (algorithm_params[3:0])
                    4'h4: predicted_ratio = 16'h0800;  // 8x (INT4)
                    4'h8: predicted_ratio = 16'h0400;  // 4x (INT8)
                    default: predicted_ratio = 16'h0200; // 2x
                endcase
            end
            
            RLE: begin
                // 基于稀疏性预测
                if (sparsity > 75) predicted_ratio = 16'h0600; // 6x
                else if (sparsity > 50) predicted_ratio = 16'h0300; // 3x
                else predicted_ratio = 16'h0150; // 1.5x
            end
            
            SPARSE: begin
                // 基于稀疏模式
                case (algorithm_params)
                    8'h24: predicted_ratio = 16'h0200; // 2x (2:4)
                    8'h48: predicted_ratio = 16'h0200; // 2x (4:8)
                    8'h11: predicted_ratio = sparsity * 16'h0010; // 可变
                endcase
            end
            
            default: predicted_ratio = 16'h0100; // 1x
        endcase
    end
endmodule
```

**压缩策略总结：**

| 数据类型 | 特征 | 推荐算法 | 预期压缩比 |
|---------|------|---------|-----------|
| Conv权重 | 中等稀疏性，分布集中 | INT8量化/2:4稀疏 | 2-4x |
| FC权重 | 高稀疏性，可剪枝 | 4:8稀疏/INT4量化 | 4-8x |
| 激活值(ReLU后) | 大量零值，正值分布 | RLE/动态量化 | 3-6x |
| 梯度 | 极稀疏，小值 | Top-K稀疏/差分编码 | 10-100x |

### 练习题

#### 习题6：存储带宽优化
**问题：**设计一个存储带宽监控和优化系统，动态调整各个模块的带宽分配。

<details>
<summary>💡 提示</summary>

思考方向：监控各个模块的带宽使用率和队列长度。实现QoS（服务质量）策略，为关键路径分配更多带宽。考虑使用令牌桶或加权轮询算法。

</details>

<details>
<summary>答案</summary>

```verilog
module BandwidthOptimizer #(
    parameter NUM_CLIENTS = 8,
    parameter TOTAL_BANDWIDTH = 1000, // GB/s
    parameter MONITOR_WINDOW = 1024
)(
    input wire clk,
    input wire rst_n,
    
    // 客户端请求
    input wire [NUM_CLIENTS-1:0] client_req,
    input wire [31:0] client_addr [NUM_CLIENTS-1:0],
    input wire [15:0] client_len [NUM_CLIENTS-1:0],
    input wire [2:0] client_priority [NUM_CLIENTS-1:0],
    
    // 带宽分配输出
    output reg [NUM_CLIENTS-1:0] client_grant,
    output reg [9:0] client_bandwidth [NUM_CLIENTS-1:0], // MB/s
    
    // 性能监控
    output reg [31:0] total_throughput,
    output reg [15:0] bandwidth_efficiency // 0-100%
);

    // 带宽使用统计
    reg [31:0] bytes_transferred [NUM_CLIENTS-1:0];
    reg [31:0] request_count [NUM_CLIENTS-1:0];
    reg [31:0] stall_cycles [NUM_CLIENTS-1:0];
    reg [15:0] monitor_cycles;
    
    // QoS参数
    reg [9:0] min_bandwidth [NUM_CLIENTS-1:0];
    reg [9:0] max_bandwidth [NUM_CLIENTS-1:0];
    reg [15:0] burst_allowance [NUM_CLIENTS-1:0];
    
    // 初始化QoS参数
    initial begin
        for (int i = 0; i < NUM_CLIENTS; i++) begin
            min_bandwidth[i] = 50;  // 50 MB/s minimum
            max_bandwidth[i] = 300; // 300 MB/s maximum
            burst_allowance[i] = 100; // 100 MB burst
        end
    end
    
    // 带宽令牌桶
    reg [15:0] tokens [NUM_CLIENTS-1:0];
    reg [3:0] refill_counter;
    
    // 令牌补充
    always @(posedge clk) begin
        refill_counter <= refill_counter + 1;
        if (refill_counter == 0) begin // 每16周期补充一次
            for (int i = 0; i < NUM_CLIENTS; i++) begin
                if (tokens[i] < client_bandwidth[i]) begin
                    tokens[i] <= tokens[i] + (client_bandwidth[i] >> 4);
                end
            end
        end
    end
    
    // 动态带宽分配算法
    reg [2:0] allocation_state;
    reg [31:0] total_demand;
    reg [31:0] allocated_bandwidth;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            allocation_state <= 0;
            monitor_cycles <= 0;
            // 初始均分带宽
            for (int i = 0; i < NUM_CLIENTS; i++) begin
                client_bandwidth[i] <= TOTAL_BANDWIDTH / NUM_CLIENTS;
            end
        end else begin
            monitor_cycles <= monitor_cycles + 1;
            
            // 收集统计信息
            for (int i = 0; i < NUM_CLIENTS; i++) begin
                if (client_req[i]) begin
                    request_count[i] <= request_count[i] + 1;
                    if (!client_grant[i]) begin
                        stall_cycles[i] <= stall_cycles[i] + 1;
                    end
                end
                
                if (client_grant[i]) begin
                    bytes_transferred[i] <= bytes_transferred[i] + client_len[i];
                end
            end
            
            // 周期性重新分配
            if (monitor_cycles >= MONITOR_WINDOW) begin
                monitor_cycles <= 0;
                allocation_state <= 1;
            end
            
            case (allocation_state)
                1: begin // 计算需求
                    total_demand = 0;
                    for (int i = 0; i < NUM_CLIENTS; i++) begin
                        // 基于历史使用计算需求
                        reg [31:0] demand;
                        demand = (bytes_transferred[i] * 1000) / MONITOR_WINDOW;
                        
                        // 考虑停顿率
                        if (stall_cycles[i] > MONITOR_WINDOW/10) begin
                            demand = demand * 15 / 10; // 增加50%
                        end
                        
                        total_demand = total_demand + demand;
                    end
                    allocation_state <= 2;
                end
                
                2: begin // 分配带宽
                    allocated_bandwidth = 0;
                    
                    // 第一轮：保证最小带宽
                    for (int i = 0; i < NUM_CLIENTS; i++) begin
                        client_bandwidth[i] <= min_bandwidth[i];
                        allocated_bandwidth = allocated_bandwidth + min_bandwidth[i];
                    end
                    
                    allocation_state <= 3;
                end
                
                3: begin // 分配剩余带宽
                    reg [31:0] remaining = TOTAL_BANDWIDTH - allocated_bandwidth;
                    
                    // 按优先级和需求比例分配
                    for (int i = 0; i < NUM_CLIENTS; i++) begin
                        reg [31:0] extra;
                        if (total_demand > 0) begin
                            extra = (remaining * bytes_transferred[i]) / total_demand;
                            
                            // 优先级加权
                            extra = extra * (client_priority[i] + 1) / 4;
                            
                            // 限制在最大带宽内
                            if (client_bandwidth[i] + extra > max_bandwidth[i]) begin
                                extra = max_bandwidth[i] - client_bandwidth[i];
                            end
                            
                            client_bandwidth[i] <= client_bandwidth[i] + extra;
                        end
                    end
                    
                    // 清除统计
                    for (int i = 0; i < NUM_CLIENTS; i++) begin
                        bytes_transferred[i] <= 0;
                        request_count[i] <= 0;
                        stall_cycles[i] <= 0;
                    end
                    
                    allocation_state <= 0;
                end
            endcase
        end
    end
    
    // 授权仲裁器
    reg [2:0] rr_pointer;
    always @(posedge clk) begin
        client_grant <= 0;
        
        // 轮询检查有令牌的请求者
        for (int i = 0; i < NUM_CLIENTS; i++) begin
            int idx = (rr_pointer + i) % NUM_CLIENTS;
            if (client_req[idx] && tokens[idx] >= client_len[idx]) begin
                client_grant[idx] <= 1;
                tokens[idx] <= tokens[idx] - client_len[idx];
                rr_pointer <= (idx + 1) % NUM_CLIENTS;
                break;
            end
        end
    end
    
    // 性能计算
    always @(posedge clk) begin
        reg [31:0] total_bytes = 0;
        reg [31:0] total_allocated = 0;
        
        for (int i = 0; i < NUM_CLIENTS; i++) begin
            total_bytes = total_bytes + bytes_transferred[i];
            total_allocated = total_allocated + client_bandwidth[i];
        end
        
        total_throughput <= (total_bytes * 1000) / monitor_cycles;
        bandwidth_efficiency <= (total_throughput * 100) / TOTAL_BANDWIDTH;
    end
endmodule
```

**优化策略：**
1. **动态带宽分配：**基于历史使用和停顿率
2. **QoS保证：**最小/最大带宽限制
3. **令牌桶限流：**平滑突发流量
4. **优先级支持：**关键路径获得更多带宽
5. **效率监控：**实时跟踪带宽利用率

</details>

#### 习题7：存储层次优化
**问题：**为CNN推理设计一个三级存储层次（L0/L1/L2），优化数据复用。

<details>
<summary>答案</summary>

```verilog
// CNN优化的三级存储层次
module CNNMemoryHierarchy #(
    parameter PE_ARRAY_DIM = 16,
    parameter L0_SIZE = 256,      // 每个PE 256B
    parameter L1_SIZE = 16384,    // 每个PE组 16KB
    parameter L2_SIZE = 2097152,  // 全局 2MB
    parameter DATA_WIDTH = 8
)(
    input wire clk,
    input wire rst_n,
    
    // CNN层配置
    input wire [15:0] layer_h, layer_w, layer_c_in, layer_c_out,
    input wire [3:0] kernel_size,
    input wire [3:0] stride,
    
    // 数据流控制
    input wire start_layer,
    output reg layer_done,
    
    // 性能统计
    output reg [31:0] l0_hits, l0_misses,
    output reg [31:0] l1_hits, l1_misses,
    output reg [31:0] l2_hits, l2_misses,
    output reg [31:0] dram_accesses
);

    // 数据复用分析
    reg [2:0] dataflow_mode;
    localparam WEIGHT_STATIONARY = 0, OUTPUT_STATIONARY = 1, 
               ROW_STATIONARY = 2, NO_LOCAL_REUSE = 3;
    
    // 复用距离计算
    function [31:0] calc_reuse_distance(
        input [2:0] data_type, // 0: weight, 1: input, 2: output
        input [2:0] df_mode
    );
        case (df_mode)
            WEIGHT_STATIONARY: begin
                case (data_type)
                    0: calc_reuse_distance = layer_h * layer_w; // 权重复用整个特征图
                    1: calc_reuse_distance = kernel_size * kernel_size; // 输入复用卷积窗口
                    2: calc_reuse_distance = 1; // 输出无复用
                endcase
            end
            
            OUTPUT_STATIONARY: begin
                case (data_type)
                    0: calc_reuse_distance = 1; // 权重无复用
                    1: calc_reuse_distance = layer_c_out; // 输入复用所有输出通道
                    2: calc_reuse_distance = layer_c_in * kernel_size * kernel_size; // 输出累加
                endcase
            end
            
            ROW_STATIONARY: begin
                // 行固定：平衡三种数据的复用
                case (data_type)
                    0: calc_reuse_distance = layer_w / stride; // 权重复用一行
                    1: calc_reuse_distance = kernel_size; // 输入复用卷积行
                    2: calc_reuse_distance = kernel_size * layer_c_in / PE_ARRAY_DIM; // 部分累加
                endcase
            end
        endcase
    endfunction
    
    // 选择最优数据流
    always @(*) begin
        reg [31:0] weight_size = kernel_size * kernel_size * layer_c_in * layer_c_out;
        reg [31:0] input_size = layer_h * layer_w * layer_c_in;
        reg [31:0] output_size = (layer_h/stride) * (layer_w/stride) * layer_c_out;
        
        // 基于层参数选择数据流
        if (kernel_size == 1) begin
            // 1x1卷积，输出固定最优
            dataflow_mode = OUTPUT_STATIONARY;
        end else if (weight_size < L1_SIZE) begin
            // 权重能放入L1，权重固定
            dataflow_mode = WEIGHT_STATIONARY;
        end else if (layer_c_in < 16 && layer_c_out > 256) begin
            // 深度可分离卷积，行固定
            dataflow_mode = ROW_STATIONARY;
        end else begin
            // 默认行固定
            dataflow_mode = ROW_STATIONARY;
        end
    end
    
    // L0缓存管理（每个PE私有）
    reg [DATA_WIDTH-1:0] l0_weight_reg [PE_ARRAY_DIM-1:0];
    reg [DATA_WIDTH-1:0] l0_input_reg [PE_ARRAY_DIM-1:0];
    reg [31:0] l0_partial_sum [PE_ARRAY_DIM-1:0];
    
    // L1缓存管理（PE组共享）
    reg [2:0] l1_allocation_mode;
    reg [15:0] l1_weight_lines;
    reg [15:0] l1_input_lines; 
    reg [15:0] l1_output_lines;
    
    always @(posedge clk) begin
        if (start_layer) begin
            // 根据数据流模式分配L1空间
            case (dataflow_mode)
                WEIGHT_STATIONARY: begin
                    l1_weight_lines <= L1_SIZE * 7 / 10; // 70%给权重
                    l1_input_lines <= L1_SIZE * 2 / 10;  // 20%给输入
                    l1_output_lines <= L1_SIZE * 1 / 10; // 10%给输出
                end
                
                OUTPUT_STATIONARY: begin
                    l1_weight_lines <= L1_SIZE * 2 / 10; // 20%给权重
                    l1_input_lines <= L1_SIZE * 3 / 10;  // 30%给输入
                    l1_output_lines <= L1_SIZE * 5 / 10; // 50%给输出
                end
                
                ROW_STATIONARY: begin
                    l1_weight_lines <= L1_SIZE * 4 / 10; // 40%给权重
                    l1_input_lines <= L1_SIZE * 4 / 10;  // 40%给输入
                    l1_output_lines <= L1_SIZE * 2 / 10; // 20%给输出
                end
            endcase
        end
    end
    
    // L2缓存管理（全局共享）
    reg [2:0] l2_partition_mode;
    reg [20:0] l2_weight_base, l2_input_base, l2_output_base;
    
    // Tile大小计算
    reg [15:0] tile_h, tile_w, tile_c;
    
    always @(posedge clk) begin
        if (start_layer) begin
            // 计算能装入L2的最大tile
            reg [31:0] weight_per_tile, input_per_tile, output_per_tile;
            
            // 尝试不同的tile大小
            for (tile_h = layer_h; tile_h > 0; tile_h = tile_h >> 1) begin
                for (tile_w = layer_w; tile_w > 0; tile_w = tile_w >> 1) begin
                    for (tile_c = layer_c_in; tile_c > 0; tile_c = tile_c >> 1) begin
                        weight_per_tile = kernel_size * kernel_size * tile_c * layer_c_out;
                        input_per_tile = (tile_h + kernel_size - 1) * 
                                       (tile_w + kernel_size - 1) * tile_c;
                        output_per_tile = (tile_h/stride) * (tile_w/stride) * layer_c_out;
                        
                        if (weight_per_tile + input_per_tile + output_per_tile <= L2_SIZE) begin
                            // 找到合适的tile大小
                            break;
                        end
                    end
                    if (weight_per_tile + input_per_tile + output_per_tile <= L2_SIZE) break;
                end
                if (weight_per_tile + input_per_tile + output_per_tile <= L2_SIZE) break;
            end
            
            // 设置L2分区
            l2_weight_base = 0;
            l2_input_base = weight_per_tile;
            l2_output_base = weight_per_tile + input_per_tile;
        end
    end
    
    // 预取调度器
    reg [3:0] prefetch_state;
    reg [15:0] current_tile_h, current_tile_w, current_tile_c;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prefetch_state <= 0;
            layer_done <= 0;
        end else begin
            case (prefetch_state)
                0: begin // 空闲
                    if (start_layer) begin
                        current_tile_h <= 0;
                        current_tile_w <= 0;
                        current_tile_c <= 0;
                        prefetch_state <= 1;
                    end
                end
                
                1: begin // 预取权重到L2
                    // 预取当前tile的权重
                    dram_accesses <= dram_accesses + 
                        (kernel_size * kernel_size * tile_c * layer_c_out) / 64;
                    prefetch_state <= 2;
                end
                
                2: begin // 预取输入到L2
                    // 预取当前tile的输入（考虑halo）
                    dram_accesses <= dram_accesses + 
                        ((tile_h + kernel_size - 1) * (tile_w + kernel_size - 1) * tile_c) / 64;
                    prefetch_state <= 3;
                end
                
                3: begin // L2到L1传输
                    // 根据数据流模式，将数据从L2搬到L1
                    case (dataflow_mode)
                        WEIGHT_STATIONARY: begin
                            // 权重常驻L1
                            l2_hits <= l2_hits + kernel_size * kernel_size;
                            l1_misses <= l1_misses + kernel_size * kernel_size;
                        end
                        
                        OUTPUT_STATIONARY: begin
                            // 输出块常驻L1
                            l2_hits <= l2_hits + tile_h * tile_w / (stride * stride);
                            l1_misses <= l1_misses + tile_h * tile_w / (stride * stride);
                        end
                    endcase
                    prefetch_state <= 4;
                end
                
                4: begin // L1到L0传输并计算
                    // 模拟PE阵列计算
                    // 统计L0/L1命中率
                    case (dataflow_mode)
                        WEIGHT_STATIONARY: begin
                            l0_hits <= l0_hits + tile_h * tile_w; // 权重复用
                            l1_hits <= l1_hits + tile_h * tile_w * kernel_size * kernel_size;
                        end
                        
                        OUTPUT_STATIONARY: begin
                            l0_hits <= l0_hits + layer_c_in * kernel_size * kernel_size; // 输出复用
                            l1_hits <= l1_hits + layer_c_in;
                        end
                    endcase
                    
                    // 检查是否完成当前tile
                    prefetch_state <= 5;
                end
                
                5: begin // 下一个tile
                    current_tile_w <= current_tile_w + tile_w;
                    if (current_tile_w >= layer_w) begin
                        current_tile_w <= 0;
                        current_tile_h <= current_tile_h + tile_h;
                        
                        if (current_tile_h >= layer_h) begin
                            current_tile_h <= 0;
                            current_tile_c <= current_tile_c + tile_c;
                            
                            if (current_tile_c >= layer_c_in) begin
                                // 层计算完成
                                layer_done <= 1;
                                prefetch_state <= 0;
                            end else begin
                                prefetch_state <= 1;
                            end
                        end else begin
                            prefetch_state <= 2; // 只需预取新的输入
                        end
                    end else begin
                        prefetch_state <= 2; // 只需预取新的输入
                    end
                end
            endcase
        end
    end
endmodule
```

**优化要点：**
1. **自适应数据流：**根据层类型选择最优数据流模式
2. **动态空间分配：**L1/L2空间根据复用模式动态分配
3. **Tile优化：**计算最大可容纳的tile尺寸
4. **预取流水：**L2预取与L1计算重叠
5. **层次化复用：**L0复用最频繁数据，L1复用中等，L2缓存tile

</details>

#### 习题8：综合设计题
**问题：**设计一个完整的NPU存储子系统，支持8×8 MAC阵列，目标是在7nm工艺下达到1TOPS@1GHz。要求：
1) 设计存储层次结构
2) 实现高效的数据搬运
3) 支持INT8/INT16混合精度
4) 功耗预算2W

<details>
<summary>答案</summary>

```verilog
// NPU存储子系统顶层设计
module NPUMemorySubsystem (
    input wire clk,              // 1GHz
    input wire rst_n,
    
    // 配置接口
    input wire [1:0] precision_mode, // 0: INT8, 1: INT16, 2: Mixed
    input wire [2:0] dataflow_mode,
    
    // 性能监控
    output wire [31:0] actual_tops,
    output wire [15:0] power_estimate_mw
);

    // ===== 1. 存储层次设计 =====
    // L0: 64 × 256b = 2KB (寄存器文件，每个PE)
    // L1: 8 × 8KB = 64KB (PE组本地缓存)
    // L2: 512KB (全局缓存)
    // 带宽需求：1TOPS × 3操作数 × 1B = 3TB/s
    
    // MAC阵列：8×8 = 64 MACs
    // 峰值性能：64 MACs × 2 ops/MAC × 1GHz = 128 GOPS (INT8)
    //           64 MACs × 2 ops/MAC × 0.5GHz = 64 GOPS (INT16)
    
    // ===== 2. 多级SRAM设计 =====
    // L0 Register File (每个PE)
    genvar i, j;
    generate
        for (i = 0; i < 8; i = i + 1) begin
            for (j = 0; j < 8; j = j + 1) begin
                L0_RegFile #(
                    .NUM_REGS(8),
                    .REG_WIDTH(256)
                ) pe_rf (
                    .clk(clk),
                    .rd_en(pe_rf_rd_en[i][j]),
                    .rd_addr(pe_rf_rd_addr[i][j]),
                    .rd_data(pe_rf_rd_data[i][j]),
                    .wr_en(pe_rf_wr_en[i][j]),
                    .wr_addr(pe_rf_wr_addr[i][j]),
                    .wr_data(pe_rf_wr_data[i][j])
                );
            end
        end
    endgenerate
    
    // L1 SRAM (PE行共享，8个8KB banks)
    generate
        for (i = 0; i < 8; i = i + 1) begin
            MultiPortSRAM #(
                .DEPTH(1024),      // 1K × 64B = 64KB
                .WIDTH(512),       // 64B宽
                .NUM_PORTS(8),     // 8个PE访问
                .BANK_COUNT(4)     // 4-way banked
            ) l1_sram (
                .clk(clk),
                .en(l1_en[i]),
                .wr(l1_wr[i]),
                .addr(l1_addr[i]),
                .wdata(l1_wdata[i]),
                .rdata(l1_rdata[i])
            );
        end
    endgenerate
    
    // L2 Global Buffer (512KB, 16-way banked)
    GlobalBuffer #(
        .SIZE(524288),      // 512KB
        .WIDTH(512),        // 64B接口
        .NUM_BANKS(16),
        .NUM_PORTS(8)       // 8个L1可同时访问
    ) l2_buffer (
        .clk(clk),
        .req_valid(l2_req_valid),
        .req_addr(l2_req_addr),
        .req_wr(l2_req_wr),
        .req_data(l2_req_data),
        .resp_valid(l2_resp_valid),
        .resp_data(l2_resp_data)
    );
    
    // ===== 3. 高带宽互连网络 =====
    // L0-L1互连：64个256b端口，聚合带宽 = 64×256b×1GHz = 2TB/s
    // L1-L2互连：8个512b端口，聚合带宽 = 8×512b×1GHz = 512GB/s
    
    CrossbarNetwork #(
        .NUM_MASTERS(64),   // 64个PE
        .NUM_SLAVES(8),     // 8个L1
        .DATA_WIDTH(256)
    ) l0_l1_xbar (
        .clk(clk),
        .master_req(pe_to_l1_req),
        .master_addr(pe_to_l1_addr),
        .master_data(pe_to_l1_data),
        .slave_ack(l1_to_pe_ack),
        .slave_data(l1_to_pe_data)
    );
    
    // ===== 4. 智能DMA引擎 =====
    IntelligentDMA #(
        .NUM_CHANNELS(4),
        .ADDR_WIDTH(32),
        .MAX_2D_SIZE(256)
    ) dma_engine (
        .clk(clk),
        .rst_n(rst_n),
        
        // 描述符接口
        .desc_valid(dma_desc_valid),
        .desc_2d_mode(dma_2d_mode),
        .desc_src_addr(dma_src_addr),
        .desc_dst_addr(dma_dst_addr),
        .desc_x_size(dma_x_size),
        .desc_y_size(dma_y_size),
        .desc_src_stride(dma_src_stride),
        .desc_dst_stride(dma_dst_stride),
        
        // L2接口
        .l2_req(dma_l2_req),
        .l2_addr(dma_l2_addr),
        .l2_wdata(dma_l2_wdata),
        .l2_rdata(l2_dma_rdata),
        
        // DRAM接口
        .dram_req(dma_dram_req),
        .dram_addr(dma_dram_addr),
        .dram_wdata(dma_dram_wdata),
        .dram_rdata(dram_dma_rdata)
    );
    
    // ===== 5. 混合精度支持 =====
    MixedPrecisionController #(
        .MAC_ARRAY_DIM(8)
    ) prec_ctrl (
        .clk(clk),
        .precision_mode(precision_mode),
        .weight_precision(weight_prec),
        .activation_precision(act_prec),
        
        // PE配置输出
        .pe_mode(pe_precision_mode),
        .pe_grouping(pe_group_config)
    );
    
    // ===== 6. 功耗优化控制 =====
    PowerController #(
        .NUM_DOMAINS(4)     // L0, L1, L2, DMA
    ) pwr_ctrl (
        .clk(clk),
        .rst_n(rst_n),
        
        // 活动监控
        .l0_active(|pe_rf_rd_en),
        .l1_active(|l1_en),
        .l2_active(|l2_req_valid),
        .dma_active(dma_busy),
        
        // 功耗控制
        .clock_gate_en(clk_gate_en),
        .voltage_scale(vdd_scale),
        .power_gate_en(pwr_gate_en),
        
        // 功耗估计
        .power_estimate(power_estimate_mw)
    );
    
    // ===== 7. 数据流协调器 =====
    DataflowOrchestrator orch (
        .clk(clk),
        .rst_n(rst_n),
        .dataflow_mode(dataflow_mode),
        
        // 层参数
        .layer_params(layer_params),
        
        // PE控制
        .pe_config(pe_config),
        .pe_enable(pe_enable),
        
        // 存储控制
        .l1_alloc_map(l1_allocation),
        .l2_alloc_map(l2_allocation),
        
        // DMA控制
        .dma_schedule(dma_schedule)
    );
    
    // ===== 8. 性能计数器 =====
    PerformanceCounters perf_cnt (
        .clk(clk),
        .rst_n(rst_n),
        
        // 输入事件
        .mac_active(mac_active),
        .l0_hit(l0_cache_hit),
        .l1_hit(l1_cache_hit),
        .l2_hit(l2_cache_hit),
        .dram_access(dram_access),
        
        // 输出统计
        .total_ops(total_operations),
        .actual_tops(actual_tops),
        .bandwidth_utilization(bw_util),
        .cache_hit_rate(cache_hit_rate)
    );
    
    // ===== 功耗分解（2W预算）=====
    // MAC阵列：~800mW (40%)
    // L0 (RegFile)：~200mW (10%)
    // L1 SRAM：~400mW (20%)
    // L2 SRAM：~300mW (15%)
    // 互连网络：~200mW (10%)
    // 控制逻辑：~100mW (5%)
    
endmodule

// 关键子模块：智能预取器
module SmartPrefetcher #(
    parameter ADDR_WIDTH = 32,
    parameter PATTERN_DEPTH = 8
)(
    input wire clk,
    input wire rst_n,
    
    // 访问监控
    input wire access_valid,
    input wire [ADDR_WIDTH-1:0] access_addr,
    
    // 预取输出
    output reg prefetch_req,
    output reg [ADDR_WIDTH-1:0] prefetch_addr,
    output reg [7:0] prefetch_len,
    
    // 模式识别
    output reg [2:0] detected_pattern // 0:Sequential, 1:Strided, 2:2D
);

    // 访问历史
    reg [ADDR_WIDTH-1:0] addr_history [PATTERN_DEPTH-1:0];
    reg [31:0] stride_history [PATTERN_DEPTH-2:0];
    reg [3:0] history_ptr;
    
    // 模式检测
    always @(posedge clk) begin
        if (access_valid) begin
            // 更新历史
            addr_history[history_ptr] <= access_addr;
            history_ptr <= (history_ptr + 1) % PATTERN_DEPTH;
            
            // 计算步长
            if (history_ptr > 0) begin
                stride_history[history_ptr-1] <= 
                    access_addr - addr_history[history_ptr-1];
            end
            
            // 检测模式
            if (history_ptr >= 3) begin
                if (stride_history[history_ptr-1] == stride_history[history_ptr-2] &&
                    stride_history[history_ptr-2] == stride_history[history_ptr-3]) begin
                    // 固定步长模式
                    detected_pattern <= 1;
                    prefetch_req <= 1;
                    prefetch_addr <= access_addr + stride_history[history_ptr-1];
                    prefetch_len <= 8; // 预取8个元素
                end
            end
        end
    end
endmodule
```

**设计总结：**

| 组件 | 规格 | 带宽 | 功耗 |
|-----|------|------|------|
| MAC阵列 | 8×8 INT8/INT16 | - | 800mW |
| L0 RegFile | 64×2KB | 2TB/s | 200mW |
| L1 SRAM | 8×8KB | 512GB/s | 400mW |
| L2 Buffer | 512KB | 128GB/s | 300mW |
| NoC+DMA | - | - | 300mW |
| **总计** | - | - | **2000mW** |

**关键设计决策：**
1. **存储层次：**三级结构平衡容量、带宽和功耗
2. **Banking策略：**L1/L2多体设计减少冲突
3. **预取机制：**模式识别的智能预取
4. **功耗优化：**细粒度时钟门控和电源门控
5. **混合精度：**动态配置支持INT8/INT16

</details>

### 5.7 Cache与Scratchpad对比

NPU设计中的一个关键决策是选择Cache还是Scratchpad存储器。两者各有优势，理解其特点对优化NPU存储系统至关重要。

#### 5.7.1 架构对比
```
Cache与Scratchpad的根本区别：

1. Cache（硬件管理）
   - 自动的数据加载/替换
   - 透明的地址映射
   - 需要标签存储和比较逻辑
   - 访问延迟不确定（命中/未命中）

2. Scratchpad（软件管理）
   - 显式的数据搬移（DMA）
   - 直接的地址映射
   - 无需标签，面积效率高
   - 访问延迟固定且低

3. NPU的典型选择
   - 主流NPU多采用Scratchpad
   - 原因：可预测的访问模式
   - 软件可精确控制数据布局
```

#### 5.7.2 性能与成本分析
```verilog
// Cache实现示例
module SimpleCache #(
    parameter CACHE_SIZE = 32768,    // 32KB
    parameter LINE_SIZE = 64,        // 64字节缓存行
    parameter WAYS = 4               // 4路组相联
)(
    input wire clk,
    input wire rst_n,
    input wire [31:0] addr,
    input wire req_valid,
    output reg [511:0] data_out,
    output reg hit,
    output reg miss
);
    localparam SETS = CACHE_SIZE / (LINE_SIZE * WAYS);
    localparam SET_BITS = $clog2(SETS);
    localparam TAG_BITS = 32 - SET_BITS - $clog2(LINE_SIZE);
    
    // 标签存储（开销：~10-15%容量）
    reg [TAG_BITS-1:0] tags [WAYS-1:0][SETS-1:0];
    reg valid [WAYS-1:0][SETS-1:0];
    reg [1:0] lru [SETS-1:0]; // LRU替换
    
    // 数据存储
    reg [LINE_SIZE*8-1:0] data [WAYS-1:0][SETS-1:0];
    
    // 地址解码
    wire [TAG_BITS-1:0] tag = addr[31:32-TAG_BITS];
    wire [SET_BITS-1:0] set = addr[32-TAG_BITS-1:6];
    
    // 标签比较（关键路径）
    always @(posedge clk) begin
        hit <= 0;
        miss <= 0;
        
        if (req_valid) begin
            for (int i = 0; i < WAYS; i++) begin
                if (valid[i][set] && tags[i][set] == tag) begin
                    hit <= 1;
                    data_out <= data[i][set];
                    // 更新LRU
                end
            end
            
            if (!hit) begin
                miss <= 1;
                // 触发缺失处理
            end
        end
    end
endmodule

// Scratchpad实现示例
module Scratchpad #(
    parameter SIZE = 32768,          // 32KB
    parameter WIDTH = 512            // 512位宽
)(
    input wire clk,
    input wire en,
    input wire wr,
    input wire [13:0] addr,          // 直接地址
    input wire [WIDTH-1:0] wdata,
    output reg [WIDTH-1:0] rdata
);
    // 简单的SRAM阵列
    reg [WIDTH-1:0] mem [0:SIZE/(WIDTH/8)-1];
    
    always @(posedge clk) begin
        if (en) begin
            if (wr)
                mem[addr] <= wdata;
            else
                rdata <= mem[addr];  // 固定1周期延迟
        end
    end
endmodule
```

#### 5.7.3 CPU中的Scratchpad历史演变

在NPU广泛采用Scratchpad之前，这种可编程的片上存储已经在各种CPU架构中有着丰富的应用历史。理解这些历史案例，有助于我们更好地把握NPU存储设计的演进脉络。

> **CPU架构中的Scratchpad应用案例**
> 
> **1. Cell Broadband Engine (2006) - Sony/IBM/Toshiba**
> Cell处理器是Scratchpad在通用处理器中最著名的应用案例。每个SPE（Synergistic Processing Element）配备256KB的本地存储（Local Store），完全由软件管理：
> - **架构特点：**每个SPE只能访问自己的Local Store，不能直接访问主存或其他SPE的存储
> - **数据传输：**通过DMA引擎在Local Store和主存之间传输数据，支持双缓冲
> - **编程模型：**程序员需要显式管理数据布局和传输，类似今天的GPU编程
> - **性能优势：**确定性的访问延迟，无Cache miss，峰值性能达204.8 GFLOPS
> - **应用困境：**编程复杂度高，难以移植传统代码，最终限制了其广泛应用
> 
> **2. TI C6000 DSP系列**
> 德州仪器的DSP广泛使用L1/L2 Scratchpad（称为TCM - Tightly Coupled Memory）：
> - **分级设计：**L1P/L1D各32KB，L2统一256KB，可配置为Cache或SRAM
> - **灵活配置：**L2可以部分配置为Cache，部分配置为Scratchpad
> - **DMA协处理器：**EDMA3支持复杂的2D/3D传输模式，自动处理数据重排
> - **实时保证：**Scratchpad模式下访问延迟固定，满足硬实时系统需求
> 
> **3. ARM TCM（Tightly Coupled Memory）**
> ARM在Cortex-R和部分Cortex-M系列中提供TCM选项：
> - **ITCM/DTCM分离：**指令TCM和数据TCM独立，避免冲突
> - **零等待访问：**单周期访问，比通过AXI总线访问快5-10倍
> - **典型应用：**中断处理程序、关键数据结构、实时控制算法
> - **容量限制：**通常较小（16KB-256KB），仅用于最关键的代码和数据

#### 5.7.4 NPU优化的混合方案

基于历史经验，现代NPU普遍采用混合存储架构，结合Scratchpad的确定性和Cache的灵活性：

> **混合存储架构设计原则**
> 
> **1. 主体采用Scratchpad（90%以上容量）**
> - **用途：**存储可预测的张量数据（权重、激活值、中间结果）
> - **管理：**编译器静态分配，DMA预取，双缓冲或多缓冲
> - **优势：**零冲突、确定延迟、功耗最优、利用率可达100%
> - **典型容量：**256KB-4MB，取决于目标应用和工艺节点
> 
> **2. 辅助小容量Cache（5-10%容量）**
> - **用途：**处理不规则访问模式：
>   - 稀疏网络的索引查找
>   - 动态形状网络的元数据
>   - 激活函数的查找表
>   - 归一化层的统计量
> - **设计：**通常采用简单的直接映射或2路组相联
> - **容量：**8KB-64KB，够用即可，避免复杂度
> 
> **3. 统一的地址空间和访问仲裁**
> - **地址划分：**Scratchpad和Cache映射到不同地址范围
> - **访问优先级：**
>   - 计算单元访问Scratchpad：最高优先级
>   - DMA传输：中等优先级，可被计算抢占
>   - Cache访问：最低优先级，用于辅助功能
> - **带宽分配：**保证Scratchpad带宽，Cache使用剩余带宽
> 
> **4. 编译器和运行时协同**
> - **静态分析：**编译时识别规则和不规则访问模式
> - **数据布局：**将规则数据分配到Scratchpad，不规则数据通过Cache访问
> - **预取调度：**生成DMA指令序列，隐藏数据传输延迟
> - **动态调整：**运行时根据Cache命中率调整数据分配策略

**实践案例：NVIDIA Tensor Core的存储层次**
虽然NVIDIA GPU主要使用Cache层次，但在Tensor Core的设计中也体现了混合思想：
- **Register File：**类似Scratchpad，由编译器完全控制
- **Shared Memory：**可配置为L1 Cache或软件管理的Scratchpad
- **L2 Cache：**统一的Cache，处理不规则访问
- **权衡：**通过配置比例，在确定性和灵活性之间平衡

```verilog
// 性能对比表
/*
┌─────────────────┬────────────────┬────────────────┬────────────────┐
│     指标        │     Cache      │   Scratchpad   │   混合方案     │
├─────────────────┼────────────────┼────────────────┼────────────────┤
│ 面积效率        │      低        │      高        │      中        │
│ (数据/总面积)   │   (~85%)       │   (~95%)       │   (~92%)       │
├─────────────────┼────────────────┼────────────────┼────────────────┤
│ 访问延迟        │   1-3周期      │    1周期       │   1周期(SP)    │
│                 │   (变化)       │    (固定)      │   1-3周期(Cache)│
├─────────────────┼────────────────┼────────────────┼────────────────┤
│ 功耗            │      高        │      低        │      中        │
│                 │  (标签比较)    │   (直接访问)   │                │
├─────────────────┼────────────────┼────────────────┼────────────────┤
│ 编程复杂度      │      低        │      高        │      中        │
│                 │   (自动)       │   (手动DMA)    │   (混合)       │
├─────────────────┼────────────────┼────────────────┼────────────────┤
│ 适用场景        │  不规则访问    │   规则访问     │   通用NPU      │
│                 │  通用处理器    │   专用加速器   │   灵活性+效率  │
└─────────────────┴────────────────┴────────────────┴────────────────┘
*/
```

Chisel版本的混合存储系统：
```scala
import chisel3._
import chisel3.util._

class HybridMemorySystem(
    scratchpadSize: Int = 256 * 1024,
    cacheSize: Int = 8 * 1024,
    dataWidth: Int = 256
) extends Module {
    val io = IO(new Bundle {
        // Scratchpad接口
        val sp = new Bundle {
            val en = Input(Bool())
            val wr = Input(Bool())
            val addr = Input(UInt(log2Ceil(scratchpadSize/(dataWidth/8)).W))
            val wdata = Input(UInt(dataWidth.W))
            val rdata = Output(UInt(dataWidth.W))
        }
        
        // Cache接口
        val cache = new Bundle {
            val req = Input(Bool())
            val addr = Input(UInt(32.W))
            val hit = Output(Bool())
            val data = Output(UInt(dataWidth.W))
        }
    })
    
    // Scratchpad模块
    val scratchpad = Module(new Scratchpad(scratchpadSize, dataWidth))
    scratchpad.io <> io.sp
    
    // Cache模块
    val cache = Module(new SimpleCache(cacheSize, dataWidth))
    cache.io.req := io.cache.req
    cache.io.addr := io.cache.addr
    io.cache.hit := cache.io.hit
    io.cache.data := cache.io.data
}

// 优化建议实现
class OptimizedNPUMemory extends Module {
    val io = IO(new Bundle {
        val cmd = Input(new MemoryCommand)
        val status = Output(new MemoryStatus)
    })
    
    // 根据访问模式自动选择存储类型
    val accessPatternDetector = Module(new AccessPatternDetector)
    val memoryAllocator = Module(new DynamicMemoryAllocator)
    
    // 自适应分配策略
    when(accessPatternDetector.io.isRegular) {
        // 规则访问 -> Scratchpad
        memoryAllocator.io.allocType := MemType.Scratchpad
    }.elsewhen(accessPatternDetector.io.isRandom) {
        // 随机访问 -> Cache
        memoryAllocator.io.allocType := MemType.Cache
    }.otherwise {
        // 混合访问 -> 智能分配
        memoryAllocator.io.allocType := MemType.Hybrid
    }
}
```

#### 练习题9：Cache vs Scratchpad权衡
**问题：**为一个处理稀疏矩阵乘法的NPU设计存储系统。稀疏数据访问不规则，但计算密集部分访问规则。如何设计？

<details>
<summary>💡 提示</summary>

思考方向：稀疏矩阵的索引访问不规则（适合Cache），但非零元素的值访问可能是连续的（适合Scratchpad）。考虑混合方案：索引用Cache，数据值用Scratchpad。

</details>

<details>
<summary>答案</summary>

```verilog
module SparseMatrixMemorySystem #(
    parameter INDEX_CACHE_SIZE = 16384,     // 16KB索引Cache
    parameter VALUE_SCRATCHPAD_SIZE = 262144, // 256KB值Scratchpad
    parameter DATA_WIDTH = 16
)(
    input wire clk,
    input wire rst_n,
    
    // 稀疏矩阵参数
    input wire [15:0] nnz,  // 非零元素数量
    
    // 索引访问（通过Cache）
    input wire idx_req,
    input wire [31:0] idx_addr,
    output wire idx_hit,
    output wire [31:0] idx_data,
    
    // 值访问（通过Scratchpad）
    input wire val_en,
    input wire [16:0] val_addr,
    output wire [DATA_WIDTH-1:0] val_data
);
    
    // 索引Cache（CSR格式的行指针和列索引）
    IndexCache #(
        .SIZE(INDEX_CACHE_SIZE),
        .LINE_SIZE(64)  // 一次加载多个索引
    ) idx_cache (
        .clk(clk),
        .rst_n(rst_n),
        .req(idx_req),
        .addr(idx_addr),
        .hit(idx_hit),
        .data(idx_data)
    );
    
    // 值Scratchpad（连续存储非零值）
    ValueScratchpad #(
        .SIZE(VALUE_SCRATCHPAD_SIZE),
        .WIDTH(DATA_WIDTH)
    ) val_sp (
        .clk(clk),
        .en(val_en),
        .addr(val_addr),
        .rdata(val_data)
    );
    
    // 预取控制器（基于访问模式）
    SparsePrefetcher prefetcher (
        .clk(clk),
        .rst_n(rst_n),
        .idx_access(idx_req),
        .idx_addr(idx_addr),
        .prefetch_trigger(prefetch_en),
        .prefetch_addr(prefetch_addr)
    );
endmodule
```

**设计要点：**
1. **混合存储：**索引用Cache处理不规则访问，值用Scratchpad保证带宽
2. **预取优化：**基于CSR访问模式的智能预取
3. **分离路径：**索引和数据值独立访问，避免冲突
4. **带宽匹配：**Scratchpad宽接口匹配计算吞吐率

</details>

### 本章小结

- **存储系统是NPU性能的决定性因素，**数据搬运的能耗是计算的10-100倍，优化存储访问是提升能效的关键
- **片上SRAM设计需要精细权衡，**通过Banking、多端口、定制单元等技术在容量、带宽、功耗间找到最优解
- **Memory Banking是实现高带宽的核心技术，**通过交织访问、冲突避免、动态仲裁等机制支持多个计算单元并行访问
- **数据预取隐藏访存延迟，**硬件预取器通过模式识别自动预测访问地址，软件预取通过编译器插入预取指令
- **缓存一致性在NPU中可以简化，**通过软件管理的Scratchpad、单向数据流等设计避免复杂的硬件一致性协议
- **DMA设计是高效数据搬运的关键，**多通道、描述符链、2D/3D传输模式支持复杂的数据布局转换
- **内存压缩技术大幅提升有效带宽，**权重压缩、激活值压缩、稀疏压缩等技术可将带宽提升2-10倍
- **未来的存储系统将更加智能，**近数据计算、存算一体等新技术有望从根本上解决存储墙问题