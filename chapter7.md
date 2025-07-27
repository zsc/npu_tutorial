# 第7章：验证与测试

## <a name="71"></a>7.1 验证方法学概述

### 7.1.1 验证的重要性

在NPU设计中，验证是确保设计正确性的关键环节。现代NPU设计复杂度极高，包含数百万甚至数千万个逻辑门，传统的仿真验证方法已无法满足验证需求。

**验证面临的主要挑战：**

1. **设计复杂度急剧增长**
   - NPU包含复杂的计算阵列、存储层次结构和控制逻辑
   - 多层次的并行性增加了验证的困难

2. **验证覆盖率要求提高**
   - 功能覆盖率、代码覆盖率、断言覆盖率等多维度要求
   - 需要达到99%以上的覆盖率才能确保设计质量

3. **上市时间压力**
   - 验证时间占整个设计周期的60-70%
   - 需要并行验证、重用验证IP来缩短周期

### 7.1.2 现代验证方法学

#### 系统级验证策略

现代NPU验证采用多层次、多方法结合的策略：

| 验证层次 | 验证方法 | 主要目标 | 覆盖率要求 |
|---------|---------|---------|-----------|
| 单元级 | 定向测试 | 基本功能验证 | 功能覆盖率>95% |
| 模块级 | 随机验证 | 接口协议验证 | 代码覆盖率>98% |
| 子系统级 | UVM验证 | 端到端功能 | 场景覆盖率>99% |
| 系统级 | 形式化验证 | 关键属性证明 | 数学证明完备 |

#### 验证环境架构

```systemverilog
// NPU验证环境顶层架构
class npu_verification_env extends uvm_env;
    
    // 验证组件
    npu_sequencer   m_sequencer;
    npu_driver      m_driver;
    npu_monitor     m_monitor;
    npu_scoreboard  m_scoreboard;
    npu_coverage    m_coverage;
    
    // 接口VIP
    axi_vip         m_axi_vip;
    ddr_vip         m_ddr_vip;
    
    // 配置对象
    npu_config      m_config;
    
    function new(string name, uvm_component parent);
        super.new(name, parent);
    endfunction
    
    function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        
        // 创建验证组件
        m_sequencer = npu_sequencer::type_id::create("m_sequencer", this);
        m_driver = npu_driver::type_id::create("m_driver", this);
        m_monitor = npu_monitor::type_id::create("m_monitor", this);
        m_scoreboard = npu_scoreboard::type_id::create("m_scoreboard", this);
        m_coverage = npu_coverage::type_id::create("m_coverage", this);
        
        // 创建VIP
        m_axi_vip = axi_vip::type_id::create("m_axi_vip", this);
        m_ddr_vip = ddr_vip::type_id::create("m_ddr_vip", this);
        
        // 获取配置
        if (!uvm_config_db#(npu_config)::get(this, "", "config", m_config))
            `uvm_fatal("CONFIG_ERROR", "Failed to get NPU config")
    endfunction
    
    function void connect_phase(uvm_phase phase);
        super.connect_phase(phase);
        
        // 连接driver和sequencer
        m_driver.seq_item_port.connect(m_sequencer.seq_item_export);
        
        // 连接monitor到scoreboard
        m_monitor.analysis_port.connect(m_scoreboard.analysis_export);
        m_monitor.analysis_port.connect(m_coverage.analysis_export);
    endfunction
    
endclass
```

## <a name="72"></a>7.2 制定NPU验证计划

### 7.2.1 验证计划的重要性

验证计划是指导整个验证工作的纲领性文档，定义了验证的目标、范围、策略和资源分配。一个完善的验证计划能够确保验证工作的系统性和完整性。

在NPU验证领域，业界有一个著名的经验法则："验证工作量通常占整个项目的60-70%"。这个数字在NPU这样的复杂系统中可能更高。例如，Google TPU的验证团队规模是设计团队的1.5-2倍，而且验证周期通常比设计周期还要长3-6个月。

**验证面临的特殊挑战：**

1. **深度学习算法的快速演进** - 新的网络结构层出不穷
2. **数据精度的多样性** - 从INT4到FP32的各种数据类型
3. **巨大的配置空间** - 各种卷积核大小、步长、填充等参数组合
4. **并行计算的复杂性** - 海量数据的同步和协调

### 7.2.2 验证目标与范围定义

定义清晰的验证目标和范围是成功验证的第一步。这就像是在地图上划定探索区域——如果范围太大，资源会被稀释；如果范围太小，可能会遗漏重要的风险点。

> **📋 NPU验证计划模板**
>
> **项目概述：**
> - NPU架构描述（计算核心数量、存储层次、互连拓扑）
> - 目标应用场景（边缘推理、数据中心训练等）
> - 关键性能指标（TOPS、功耗、面积）
>
> **验证范围定义：**
> - 功能验证：指令集、数据流、控制逻辑
> - 性能验证：吞吐量、延迟、带宽利用率
> - 功耗验证：动态功耗、静态功耗、功耗管理
> - 兼容性验证：软件栈、编译器、驱动程序
>
> **验证边界：**
> - 包含的模块：MAC阵列、DMA控制器、调度器、互连
> - 排除的模块：外部DDR控制器、PCIe接口（假设已验证）
> - 配置范围：支持的数据类型、批处理大小、网络层类型

### 7.2.3 验证策略与方法选择

选择合适的验证策略就像是选择武器——不同的挑战需要不同的工具。NPU验证的特殊性在于它涵盖了从底层硬件到上层软件的整个栈。

**验证策略金字塔：**

```
┌─────────────────┐
│  系统级验证      │ ← 软硬件协同、真实应用
├─────────────────┤
│  子系统验证      │ ← 多模块集成、数据流
├─────────────────┤  
│   模块验证       │ ← UVM环境、功能覆盖
├─────────────────┤
│   单元验证       │ ← 形式化验证、定向测试
└─────────────────┘
```

**方法选择准则：**
- **形式化验证：** 适用于控制密集型模块（如仲裁器、FSM）
- **约束随机验证：** 适用于数据路径和配置空间大的模块
- **定向测试：** 适用于特定场景和边界条件
- **硬件加速：** 适用于系统级性能验证和软件开发

### 7.2.4 覆盖率驱动的验证

覆盖率驱动验证（Coverage-Driven Verification）是现代验证方法学的核心。它的基本理念是："你无法改进你不能测量的东西"。覆盖率就像是验证工作的"仪表盘"，告诉我们已经探索了设计空间的哪些部分，还有哪些"盲区"。

**覆盖率类型：**

| 覆盖率类型 | 定义 | NPU中的应用 | 目标 |
|-----------|------|------------|------|
| 代码覆盖率 | 执行的代码行/分支百分比 | 控制逻辑验证 | >98% |
| 功能覆盖率 | 功能点/场景覆盖百分比 | 指令集、数据流验证 | >99% |
| 断言覆盖率 | 触发的断言百分比 | 接口协议验证 | 100% |
| 交叉覆盖率 | 参数组合覆盖百分比 | 配置空间验证 | >95% |

```systemverilog
// NPU功能覆盖率定义示例
covergroup npu_operation_cg @(posedge clk);
    
    // 操作类型覆盖
    operation_type: coverpoint op_type {
        bins conv2d = {CONV2D};
        bins matmul = {MATMUL};
        bins pool   = {POOL};
        bins relu   = {RELU};
        bins add    = {ADD};
        bins mul    = {MUL};
    }
    
    // 数据类型覆盖
    data_type: coverpoint dtype {
        bins int8   = {INT8};
        bins int16  = {INT16};
        bins fp16   = {FP16};
        bins fp32   = {FP32};
    }
    
    // 张量形状覆盖
    tensor_shape: coverpoint {batch, height, width, channel} {
        bins small  = {[1:4], [1:32], [1:32], [1:64]};
        bins medium = {[1:16], [33:224], [33:224], [65:512]};
        bins large  = {[17:64], [225:1024], [225:1024], [513:2048]};
    }
    
    // 交叉覆盖：操作类型与数据类型的组合
    op_dtype_cross: cross operation_type, data_type {
        ignore_bins unsupported = binsof(operation_type.pool) && 
                                  binsof(data_type.fp32);
    }
    
    // 交叉覆盖：数据类型与张量形状的组合
    dtype_shape_cross: cross data_type, tensor_shape;
    
endgroup
```

## <a name="73"></a>7.3 UVM验证环境构建

### 7.3.1 UVM在NPU验证中的应用

UVM（Universal Verification Methodology）提供了标准化的验证组件和可重用的验证环境架构。在NPU验证中，UVM就像是一个精密的工厂流水线——它能够持续不断地生产测试用例，执行测试，收集结果，并分析覆盖率。

NPU的UVM环境设计面临着独特的挑战。与传统处理器不同，NPU的输入不是指令流，而是大量的张量数据。这意味着我们需要创建能够生成各种大小、形状和数据分布的测试激励。

### 7.3.2 NPU验证环境架构

```systemverilog
// NPU卷积模块的高级UVM测试环境
class conv_sequence_item extends uvm_sequence_item;
    `uvm_object_utils(conv_sequence_item)
    
    // 输入数据
    rand bit [7:0] input_data[];
    rand bit [7:0] weight_data[];
    rand int kernel_size;
    rand int stride;
    rand int padding;
    
    // 错误注入控制
    rand bit enable_error_injection;
    rand error_type_e error_type;
    rand int error_location;
    
    // 错误类型定义
    typedef enum {
        NO_ERROR,
        DATA_CORRUPTION,      // 数据损坏
        WEIGHT_CORRUPTION,    // 权重损坏
        OVERFLOW_ERROR,       // 溢出错误
        BUS_ERROR,           // 总线错误
        MEMORY_ECC_ERROR     // 内存ECC错误
    } error_type_e;
    
    // 约束
    constraint valid_params_c {
        kernel_size inside {1, 3, 5, 7};
        stride inside {1, 2, 4};
        padding inside {0, 1, 2, 3};
        input_data.size() == 224*224*3;  // 假设输入是224x224x3
        weight_data.size() == kernel_size*kernel_size*3*64;  // 输出64通道
    }
    
    // 错误注入约束
    constraint error_injection_c {
        enable_error_injection dist {0 := 90, 1 := 10};  // 10%概率注入错误
        if (enable_error_injection) {
            error_type dist {
                NO_ERROR := 0,
                DATA_CORRUPTION := 30,
                WEIGHT_CORRUPTION := 20,
                OVERFLOW_ERROR := 20,
                BUS_ERROR := 20,
                MEMORY_ECC_ERROR := 10
            };
            error_location inside {[0:input_data.size()-1]};
        } else {
            error_type == NO_ERROR;
        }
    }
    
    function new(string name = "conv_sequence_item");
        super.new(name);
    endfunction
    
    // 后随机化处理
    function void post_randomize();
        // 根据错误类型注入错误
        if (enable_error_injection) begin
            case (error_type)
                DATA_CORRUPTION: begin
                    // 随机翻转数据中的几个比特
                    for (int i = 0; i < 5; i++) begin
                        int idx = $urandom_range(0, input_data.size()-1);
                        input_data[idx] = input_data[idx] ^ (1 << $urandom_range(0, 7));
                    end
                end
                WEIGHT_CORRUPTION: begin
                    // 将某些权重设置为极值
                    for (int i = 0; i < 10; i++) begin
                        int idx = $urandom_range(0, weight_data.size()-1);
                        weight_data[idx] = $urandom_range(0, 1) ? 8'hFF : 8'h00;
                    end
                end
            endcase
        end
    endfunction
endclass
```

### 7.3.3 增强型Driver设计

```systemverilog
// 增强型卷积模块Driver（支持错误注入）
class conv_driver extends uvm_driver #(conv_sequence_item);
    `uvm_component_utils(conv_driver)
    
    virtual conv_if vif;
    int error_count = 0;
    
    function new(string name, uvm_component parent);
        super.new(name, parent);
    endfunction
    
    task run_phase(uvm_phase phase);
        forever begin
            seq_item_port.get_next_item(req);
            drive_transaction(req);
            seq_item_port.item_done();
        end
    endtask
    
    task drive_transaction(conv_sequence_item trans);
        // 配置卷积参数
        vif.kernel_size <= trans.kernel_size;
        vif.stride <= trans.stride;
        vif.padding <= trans.padding;
        @(posedge vif.clk);
        
        // 根据错误类型注入总线错误
        if (trans.enable_error_injection && trans.error_type == conv_sequence_item::BUS_ERROR) begin
            inject_bus_error();
        end
        
        // 加载权重（可能注入ECC错误）
        vif.weight_valid <= 1'b1;
        foreach(trans.weight_data[i]) begin
            vif.weight_data <= trans.weight_data[i];
            
            // 注入内存ECC错误
            if (trans.enable_error_injection && 
                trans.error_type == conv_sequence_item::MEMORY_ECC_ERROR &&
                i == trans.error_location) begin
                vif.mem_ecc_error <= 1'b1;
                `uvm_info("DRIVER", $sformatf("Injecting ECC error at weight[%0d]", i), UVM_LOW)
            end else begin
                vif.mem_ecc_error <= 1'b0;
            end
            
            @(posedge vif.clk);
        end
        vif.weight_valid <= 1'b0;
        vif.mem_ecc_error <= 1'b0;
        
        // 输入数据（可能注入溢出）
        vif.data_valid <= 1'b1;
        foreach(trans.input_data[i]) begin
            vif.input_data <= trans.input_data[i];
            
            // 注入溢出错误
            if (trans.enable_error_injection && 
                trans.error_type == conv_sequence_item::OVERFLOW_ERROR &&
                i % 100 == 0) begin
                vif.force_overflow <= 1'b1;
                `uvm_info("DRIVER", "Forcing accumulator overflow", UVM_LOW)
            end else begin
                vif.force_overflow <= 1'b0;
            end
            
            @(posedge vif.clk);
        end
        vif.data_valid <= 1'b0;
        vif.force_overflow <= 1'b0;
        
        // 记录错误注入统计
        if (trans.enable_error_injection) begin
            error_count++;
            `uvm_info("DRIVER", $sformatf("Total errors injected: %0d", error_count), UVM_MEDIUM)
        end
    endtask
    
    // 注入总线错误
    task inject_bus_error();
        `uvm_info("DRIVER", "Injecting AXI bus error", UVM_LOW)
        vif.axi_error_inject <= 1'b1;
        @(posedge vif.clk);
        vif.axi_error_inject <= 1'b0;
    endtask
    
endclass
```

### 7.3.4 智能Monitor设计

Monitor是验证环境的"眼睛"，负责观察设计的行为并收集数据。在NPU验证中，Monitor需要处理大量的并行数据流。

```systemverilog
// 智能NPU Monitor设计
class npu_monitor extends uvm_monitor;
    `uvm_component_utils(npu_monitor)
    
    virtual npu_if vif;
    uvm_analysis_port #(conv_sequence_item) analysis_port;
    
    // 性能计数器
    int total_operations = 0;
    int error_detected = 0;
    real average_latency = 0;
    real peak_throughput = 0;
    
    // 数据完整性检查
    bit [31:0] checksum_input = 0;
    bit [31:0] checksum_output = 0;
    
    function new(string name, uvm_component parent);
        super.new(name, parent);
        analysis_port = new("analysis_port", this);
    endfunction
    
    task run_phase(uvm_phase phase);
        fork
            monitor_data_flow();
            monitor_performance();
            monitor_error_signals();
            monitor_power_events();
        join
    endtask
    
    // 监控数据流
    task monitor_data_flow();
        conv_sequence_item trans;
        forever begin
            @(posedge vif.clk);
            
            if (vif.start_conv) begin
                trans = conv_sequence_item::type_id::create("trans");
                
                // 收集输入数据
                collect_input_data(trans);
                
                // 等待计算完成
                wait(vif.conv_done);
                
                // 收集输出数据
                collect_output_data(trans);
                
                // 数据完整性检查
                verify_data_integrity(trans);
                
                // 发送到scoreboard
                analysis_port.write(trans);
                
                total_operations++;
            end
        end
    endtask
    
    // 监控性能指标
    task monitor_performance();
        time start_time, end_time;
        int cycle_count = 0;
        
        forever begin
            @(posedge vif.clk);
            
            if (vif.start_conv) begin
                start_time = $time;
                cycle_count = 0;
            end
            
            if (vif.conv_done) begin
                end_time = $time;
                cycle_count++;
                
                // 计算延迟
                real latency = (end_time - start_time) / 1ns;
                average_latency = (average_latency * (total_operations - 1) + latency) / total_operations;
                
                // 计算吞吐量
                real current_throughput = 1.0 / latency * 1000; // 每秒操作数
                if (current_throughput > peak_throughput)
                    peak_throughput = current_throughput;
                
                `uvm_info("MONITOR", $sformatf("Op %0d: Latency=%.2f ns, Throughput=%.2f ops/s", 
                         total_operations, latency, current_throughput), UVM_LOW)
            end
        end
    endtask
    
    // 监控错误信号
    task monitor_error_signals();
        forever begin
            @(posedge vif.clk);
            
            if (vif.overflow_error) begin
                `uvm_error("MONITOR", "Arithmetic overflow detected")
                error_detected++;
            end
            
            if (vif.underflow_error) begin
                `uvm_error("MONITOR", "Arithmetic underflow detected") 
                error_detected++;
            end
            
            if (vif.memory_error) begin
                `uvm_error("MONITOR", "Memory access error detected")
                error_detected++;
            end
            
            if (vif.protocol_error) begin
                `uvm_error("MONITOR", "Bus protocol violation detected")
                error_detected++;
            end
        end
    endtask
    
    // 监控功耗事件
    task monitor_power_events();
        forever begin
            @(posedge vif.clk);
            
            if (vif.power_gating_event) begin
                `uvm_info("MONITOR", "Power gating event detected", UVM_MEDIUM)
            end
            
            if (vif.clock_gating_event) begin
                `uvm_info("MONITOR", "Clock gating event detected", UVM_MEDIUM)
            end
            
            if (vif.dvfs_event) begin
                `uvm_info("MONITOR", "DVFS transition detected", UVM_MEDIUM)
            end
        end
    endtask
    
    // 数据完整性验证
    function void verify_data_integrity(conv_sequence_item trans);
        // 计算输入数据校验和
        checksum_input = 0;
        foreach(trans.input_data[i]) begin
            checksum_input += trans.input_data[i];
        end
        
        // 验证输出数据的合理性
        foreach(trans.output_data[i]) begin
            if (trans.output_data[i] > 16'h7FFF) begin
                `uvm_warning("MONITOR", $sformatf("Suspicious large output value: %h", trans.output_data[i]))
            end
        end
    endfunction
    
    // 生成最终报告
    function void report_phase(uvm_phase phase);
        `uvm_info("MONITOR", "=== NPU Verification Statistics ===", UVM_NONE)
        `uvm_info("MONITOR", $sformatf("Total Operations: %0d", total_operations), UVM_NONE)
        `uvm_info("MONITOR", $sformatf("Errors Detected: %0d", error_detected), UVM_NONE) 
        `uvm_info("MONITOR", $sformatf("Average Latency: %.2f ns", average_latency), UVM_NONE)
        `uvm_info("MONITOR", $sformatf("Peak Throughput: %.2f ops/s", peak_throughput), UVM_NONE)
        
        if (error_detected > 0) begin
            `uvm_error("MONITOR", "Verification completed with errors!")
        end else begin
            `uvm_info("MONITOR", "Verification completed successfully!", UVM_NONE)
        end
    endfunction
    
endclass
```

### 7.3.5 高级Scoreboard设计

```systemverilog
// 预测型Scoreboard，包含硬件加速的参考模型
class npu_scoreboard extends uvm_scoreboard;
    `uvm_component_utils(npu_scoreboard)
    
    uvm_analysis_export #(conv_sequence_item) analysis_export;
    uvm_tlm_analysis_fifo #(conv_sequence_item) analysis_fifo;
    
    // 参考模型（可以是软件模型或硬件加速模型）
    npu_reference_model ref_model;
    
    // 比较统计
    int total_comparisons = 0;
    int mismatches = 0;
    real max_error = 0.0;
    real average_error = 0.0;
    
    // 误差容忍度（对于量化计算）
    real error_tolerance = 0.01;  // 1%误差容忍度
    
    function new(string name, uvm_component parent);
        super.new(name, parent);
        analysis_export = new("analysis_export", this);
        analysis_fifo = new("analysis_fifo", this);
    endfunction
    
    function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        ref_model = npu_reference_model::type_id::create("ref_model", this);
    endfunction
    
    function void connect_phase(uvm_phase phase);
        analysis_export.connect(analysis_fifo.analysis_export);
    endfunction
    
    task run_phase(uvm_phase phase);
        conv_sequence_item trans;
        forever begin
            analysis_fifo.get(trans);
            check_transaction(trans);
        end
    endtask
    
    // 检查单个transaction
    function void check_transaction(conv_sequence_item trans);
        conv_sequence_item expected_trans;
        real error_percentage;
        
        // 使用参考模型计算期望结果
        expected_trans = ref_model.predict(trans);
        
        // 比较结果
        foreach(trans.output_data[i]) begin
            real actual = $itor(trans.output_data[i]);
            real expected = $itor(expected_trans.output_data[i]);
            real error = abs(actual - expected) / (expected + 0.001); // 避免除零
            
            if (error > error_tolerance) begin
                `uvm_error("SCOREBOARD", 
                    $sformatf("Mismatch at output[%0d]: Expected=%0d, Actual=%0d, Error=%.2f%%", 
                             i, expected_trans.output_data[i], trans.output_data[i], error*100))
                mismatches++;
            end
            
            // 更新统计信息
            if (error > max_error) max_error = error;
            average_error = (average_error * total_comparisons + error) / (total_comparisons + 1);
        end
        
        total_comparisons++;
        
        // 特殊检查：量化误差分析
        if (trans.data_type == INT8) begin
            analyze_quantization_error(trans, expected_trans);
        end
        
        `uvm_info("SCOREBOARD", $sformatf("Transaction %0d checked. Max error: %.4f%%", 
                 total_comparisons, max_error*100), UVM_HIGH)
    endfunction
    
    // 量化误差分析
    function void analyze_quantization_error(conv_sequence_item actual, conv_sequence_item expected);
        int histogram[10]; // 误差分布直方图
        
        foreach(actual.output_data[i]) begin
            real error = abs($itor(actual.output_data[i]) - $itor(expected.output_data[i])) / 
                         ($itor(expected.output_data[i]) + 0.001);
            
            int bucket = error * 1000; // 转换为千分比
            if (bucket < 10) histogram[bucket]++;
        end
        
        // 输出量化误差分析
        `uvm_info("SCOREBOARD", "Quantization Error Distribution:", UVM_MEDIUM)
        for (int i = 0; i < 10; i++) begin
            if (histogram[i] > 0) begin
                `uvm_info("SCOREBOARD", $sformatf("  %0d.%0d-%0d.%0d%%: %0d samples", 
                         i/10, i%10, (i+1)/10, (i+1)%10, histogram[i]), UVM_MEDIUM)
            end
        end
    endfunction
    
    function void report_phase(uvm_phase phase);
        `uvm_info("SCOREBOARD", "=== Scoreboard Final Report ===", UVM_NONE)
        `uvm_info("SCOREBOARD", $sformatf("Total Comparisons: %0d", total_comparisons), UVM_NONE)
        `uvm_info("SCOREBOARD", $sformatf("Mismatches: %0d (%.2f%%)", mismatches, 
                 mismatches*100.0/total_comparisons), UVM_NONE)
        `uvm_info("SCOREBOARD", $sformatf("Max Error: %.4f%%", max_error*100), UVM_NONE)
        `uvm_info("SCOREBOARD", $sformatf("Average Error: %.4f%%", average_error*100), UVM_NONE)
        
        if (mismatches == 0) begin
            `uvm_info("SCOREBOARD", "✓ All comparisons passed!", UVM_NONE)
        end else begin
            `uvm_error("SCOREBOARD", "✗ Verification failed with mismatches!")
        end
    endfunction
    
endclass
```

## <a name="74"></a>7.4 形式化验证

### 7.4.1 形式化验证在NPU中的应用

形式化验证使用数学方法证明设计的正确性，特别适用于关键控制逻辑的验证。

```systemverilog
// NPU仲裁器的形式化验证属性
module npu_arbiter_properties(
    input clk,
    input rstn,
    input [3:0] request,
    input [3:0] grant,
    input [1:0] grant_id
);

// 属性1：互斥性 - 同时只能有一个grant有效
property mutual_exclusion;
    @(posedge clk) disable iff (!rstn)
    $onehot0(grant);  // 最多一个bit为1
endproperty
assert_mutual_exclusion: assert property(mutual_exclusion)
    else `uvm_error("FORMAL", "Multiple grants active simultaneously")

// 属性2：活跃性 - 有请求必须有响应
property liveness;
    @(posedge clk) disable iff (!rstn)
    |request |-> ##[1:10] |grant;  // 请求后1-10个周期内必须有grant
endproperty
assert_liveness: assert property(liveness)
    else `uvm_error("FORMAL", "Request not granted within timeout")

// 属性3：公平性 - 长期来看各请求者获得公平服务
property fairness;
    @(posedge clk) disable iff (!rstn)
    request[0] && !grant[0] |-> ##[1:20] grant[0];
endproperty
assert_fairness_0: assert property(fairness)
    else `uvm_error("FORMAL", "Request 0 starved")

// 属性4：grant_id的正确性
property grant_id_correct;
    @(posedge clk) disable iff (!rstn)
    grant[0] |-> grant_id == 0;
endproperty
assert_grant_id_0: assert property(grant_id_correct)
    else `uvm_error("FORMAL", "Grant ID mismatch for request 0")

// 重复为其他请求者定义类似属性...

// 覆盖率属性 - 确保所有请求者都被测试到
cover_all_requesters: cover property(
    @(posedge clk) disable iff (!rstn)
    (grant[0] ##1 grant[1] ##1 grant[2] ##1 grant[3])
);

endmodule
```

### 7.4.2 MAC阵列数据流形式化验证

```systemverilog
// MAC阵列数据流的形式化验证
module mac_array_properties(
    input clk,
    input rstn,
    input start_compute,
    input [7:0] input_data,
    input [7:0] weight_data,
    input data_valid,
    input weight_valid,
    output [31:0] result,
    output result_valid
);

// 属性1：数据流延迟确定性
property deterministic_latency;
    @(posedge clk) disable iff (!rstn)
    start_compute && data_valid && weight_valid |-> ##4 result_valid;
endproperty
assert_deterministic_latency: assert property(deterministic_latency)
    else `uvm_error("FORMAL", "MAC array latency not deterministic")

// 属性2：数据有效性传播
property valid_propagation;
    @(posedge clk) disable iff (!rstn)
    !data_valid || !weight_valid |-> ##[1:5] !result_valid;
endproperty
assert_valid_propagation: assert property(valid_propagation)
    else `uvm_error("FORMAL", "Invalid data propagated through MAC array")

// 属性3：溢出检测
property overflow_detection;
    @(posedge clk) disable iff (!rstn)
    result_valid && (result > 32'h7FFFFFFF) |-> overflow_flag;
endproperty
assert_overflow_detection: assert property(overflow_detection)
    else `uvm_error("FORMAL", "Overflow not detected")

// 覆盖率：边界条件
cover_max_input: cover property(
    @(posedge clk) disable iff (!rstn)
    input_data == 8'hFF && weight_data == 8'hFF
);

cover_zero_input: cover property(
    @(posedge clk) disable iff (!rstn)
    input_data == 8'h00 || weight_data == 8'h00
);

endmodule
```

### 7.4.3 功耗管理形式化验证

```systemverilog
// 功耗管理的形式化验证
module power_management_properties(
    input clk,
    input rstn,
    input power_down_req,
    input power_up_req,
    input idle_state,
    input power_gated,
    input clock_gated,
    input retention_enable
);

// 属性1：功耗状态转换安全性
property safe_power_down;
    @(posedge clk) disable iff (!rstn)
    power_down_req && !idle_state |-> !power_gated until idle_state;
endproperty
assert_safe_power_down: assert property(safe_power_down)
    else `uvm_error("FORMAL", "Unsafe power down during active operation")

// 属性2：数据保持
property data_retention;
    @(posedge clk) disable iff (!rstn)
    power_down_req |-> ##[0:3] retention_enable;
endproperty
assert_data_retention: assert property(data_retention)
    else `uvm_error("FORMAL", "Data retention not enabled before power down")

// 属性3：时钟门控在功耗门控之前
property clock_before_power;
    @(posedge clk) disable iff (!rstn)
    power_gated |-> clock_gated;
endproperty
assert_clock_before_power: assert property(clock_before_power)
    else `uvm_error("FORMAL", "Power gated without clock gating")

// 属性4：上电序列
property power_up_sequence;
    @(posedge clk) disable iff (!rstn)
    power_up_req && power_gated |-> 
        ##1 !power_gated ##1 !clock_gated ##1 !retention_enable;
endproperty
assert_power_up_sequence: assert property(power_up_sequence)
    else `uvm_error("FORMAL", "Incorrect power up sequence")

endmodule
```
