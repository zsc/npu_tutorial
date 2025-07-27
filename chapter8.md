# 第8章：物理设计基础

## 8.1 ASIC设计流程概述

### 8.1.1 从RTL到GDSII的完整流程

物理设计是将RTL代码转换为可制造的芯片版图（GDSII）的关键步骤。这个过程就像是将建筑师的蓝图转化为实际的建筑物——需要考虑材料限制、工艺约束、成本控制等现实因素。

在NPU设计中，物理设计的重要性尤为突出。NPU通常包含大量的计算单元和存储器，这些模块的物理布局直接影响芯片的性能、功耗和面积。一个优秀的物理设计可以让相同的RTL代码获得20-30%的性能提升。

**ASIC设计流程的主要阶段：**

1. **综合（Synthesis）** - 将RTL转换为门级网表
2. **布图规划（Floorplanning）** - 确定主要功能块的位置
3. **布局（Placement）** - 确定每个逻辑门的具体位置
4. **时钟树综合（CTS）** - 构建时钟分发网络
5. **布线（Routing）** - 连接各个逻辑门
6. **时序优化** - 满足时序约束
7. **物理验证** - DRC、LVS、ERC检查
8. **签核（Sign-off）** - 最终版图验证

### 8.1.2 NPU物理设计的特殊挑战

NPU的物理设计面临着独特的挑战，这些挑战源于其计算密集、数据密集的特性：

**计算阵列的规整性要求：**
- MAC阵列需要规整的布局以最小化布线长度
- 脉动阵列要求相邻PE之间的延迟平衡
- 大规模并行计算单元的时钟树设计复杂

**巨大的存储容量：**
- 片上SRAM占芯片面积的60-80%
- 存储器编译器的选择和配置影响PPA
- 存储器的bank化设计需要精心规划

**功耗密度的挑战：**
- NPU的功耗密度可达CPU的3-5倍
- 需要精心设计电源网络和热管理
- 动态功耗优化技术的应用

```tcl
# NPU物理设计基本约束示例
# 设置工艺库和技术文件
set_db design_process_node 7
set_db design_power_domains {{PD_CORE 0.8} {PD_IO 1.8}}

# 时序约束
create_clock -name "clk_main" -period 2.0 [get_ports clk]
create_clock -name "clk_ddr" -period 1.6 [get_ports ddr_clk]

# 时钟域交叉约束
set_clock_groups -asynchronous \
    -group [get_clocks clk_main] \
    -group [get_clocks clk_ddr]

# 面积约束
set_max_area 50000000  # 50mm²

# 功耗约束
set_max_dynamic_power 15  # 15W动态功耗
set_max_leakage_power 0.5 # 0.5W静态功耗

# NPU特殊约束
# MAC阵列的规整布局约束
create_bound_box mac_array_16x16 {100 100 1600 1600}
set_dont_touch [get_cells mac_array_16x16/*]

# 存储器的层次化约束
create_voltage_island weight_memory 0.8
create_voltage_island activation_memory 0.8

# 热点控制
set_max_transition 0.1 [get_nets -hier *clk*]
set_max_fanout 32 [all_inputs]
```

## 8.2 综合与逻辑优化

### 8.2.1 RTL综合基础

RTL综合是将高级硬件描述语言转换为门级网表的过程。这个过程需要在面积、时序、功耗三个维度之间找到最优平衡点。

**综合的主要步骤：**

1. **RTL分析和精化**
   - 语法检查和语义分析
   - 层次结构的建立
   - 约束条件的解析

2. **高级综合优化**
   - 资源共享和调度
   - 数据路径优化
   - 控制逻辑简化

3. **技术映射**
   - 将通用逻辑映射到工艺库
   - 门级优化
   - 物理感知的映射

4. **时序驱动优化**
   - 关键路径优化
   - 时钟树预估
   - 负载平衡

### 8.2.2 NPU模块的综合策略

```tcl
# NPU卷积核综合脚本示例
# 设置综合环境
source setup.tcl

# 读取RTL代码
analyze -format verilog {
    ../rtl/conv_pe.v
    ../rtl/conv_array.v
    ../rtl/conv_controller.v
    ../rtl/conv_top.v
}

elaborate conv_top

# 设置约束
source constraints.tcl

# 编译策略设置
set_app_var compile_ultra_ungroup_dw false
set_app_var hdlin_check_no_latch true

# 针对计算密集模块的优化
set_optimize_registers true -design conv_array
set_structure -boolean false -timing true

# 资源共享策略
set_resource_allocation area
group_path -name "conv_datapath" -from [all_inputs] -to [all_outputs]

# 高级优化选项
compile_ultra -gate_clock -scan -no_autoungroup

# 针对MAC阵列的特殊处理
set_dont_retime [get_cells mac_array_inst/*] true
set_size_only [get_cells mac_array_inst/*/mult_*] true

# 报告生成
report_area -hierarchy
report_timing -path_type summary -delay_type max
report_power -analysis_effort high
```

### 8.2.3 时序优化技术

```tcl
# 时序优化脚本
# 识别关键路径
report_timing -path_type end -delay_type max -max_paths 20

# 流水线插入
set_app_var timing_enable_multiple_clocks_per_reg true

# 逻辑重构
optimize_netlist -area
optimize_netlist -ungroup_all

# 缓冲器插入
insert_buffer -lib_cell BUFX4 -net [get_nets critical_net]

# 门级优化
size_cell [get_cells slow_cell] NAND2X8
swap_cell [get_cells logic_cell] OAI21X2
```

## 8.3 布图规划与布局优化

### 8.3.1 NPU布图规划策略

布图规划（Floorplanning）是物理设计流程中最重要的步骤之一。好的floorplan是成功物理设计的基础，而差的floorplan可能导致后续所有努力都无法弥补。

在NPU设计中，布图规划面临着独特的挑战：

**计算阵列的规整性要求：**
- MAC阵列需要规整排列以最小化延迟
- 脉动阵列要求数据流路径优化
- 存储器Bank的合理分布

**热点分散：**
- 计算密集区域的温度控制
- 功耗密度的均匀分布
- 散热路径的优化设计

```tcl
# NPU Floorplan脚本示例
# 设置芯片尺寸和核心区域
set_die_area -coordinate {0 0 8000 8000}
set_core_area -coordinate {100 100 7900 7900}

# 创建电源环
create_power_ring -ring_width 20 \
    -ring_offset 10 \
    -nets {VDD VSS}

# MAC阵列区域规划
create_bound_box mac_array_0 {1000 1000 3000 3000}
create_bound_box mac_array_1 {4000 1000 6000 3000}
create_bound_box mac_array_2 {1000 4000 3000 6000}
create_bound_box mac_array_3 {4000 4000 6000 6000}

# 存储器放置策略
place_macro weight_memory_0 -coordinate {500 500} -orientation R0
place_macro weight_memory_1 -coordinate {6500 500} -orientation R0
place_macro activation_cache -coordinate {3500 3500} -orientation R0

# 时钟区域定义
create_clock_region clock_region_1 -coordinate {0 0 4000 8000}
create_clock_region clock_region_2 -coordinate {4000 0 8000 8000}

# 电源规划
create_power_domain PD_MAC -supply {VDD_MAC VSS}
create_power_domain PD_MEM -supply {VDD_MEM VSS}

# 热感知布局约束
set_placement_blockage -type soft -coordinate {3800 3800 4200 4200}
# 在芯片中心创建软禁区，避免热点过度集中
```

### 8.3.2 布局优化技术

```tcl
# 布局优化脚本
# 全局布局
place_design -timing_driven

# 层次化布局优化
# 首先优化关键模块
place_design -incremental -inst mac_array_inst

# 时序驱动布局优化
place_opt_design -area_recovery -power

# 拥塞分析和优化
report_congestion -rerun_global_route
set_app_var place_opt_congestion_driven_max_util 0.75

# 时钟感知布局
place_design -clock_gate_aware

# 布局质量检查
check_placement -verbose
report_placement_utilization
```

## 8.4 时钟树综合

### 8.4.1 NPU时钟树设计挑战

NPU的时钟树设计面临特殊挑战：

1. **大规模时钟负载**：数万个寄存器需要时钟
2. **多时钟域协调**：计算、控制、接口多个时钟域
3. **功耗控制**：时钟功耗占总功耗的20-40%
4. **偏斜控制**：严格的时钟偏斜要求

```tcl
# NPU时钟树综合脚本
# 时钟规格设置
set_clock_tree_options -target_skew 50ps
set_clock_tree_options -target_latency 300ps

# 时钟门控设置
set_clock_gating_style -sequential_cell CKGATEHD_X2 \
    -num_stages 1 \
    -positive_edge_logic integrated

# 有用偏斜优化
set_clock_tree_options -useful_skew true
set_clock_tree_options -useful_skew_ccopt true

# 多电压域时钟树
create_clock_tree_spec -file npu_cts.spec

# 时钟树综合
clock_opt -from build_clock -to route_clock

# 时钟质量报告
report_clock_tree -summary
report_clock_timing -type skew
```

### 8.4.2 时钟门控优化

```systemverilog
// 高效的时钟门控单元设计
module advanced_clock_gate (
    input  wire clk_in,
    input  wire enable,
    input  wire test_enable,
    output wire clk_out
);

// 集成时钟门控单元，具有更好的功耗特性
CKGATEHD_X2 u_ckgate (
    .CK   (clk_in),
    .E    (enable | test_enable),
    .ECK  (clk_out)
);

endmodule

// 层次化时钟门控策略
module npu_compute_unit (
    input  wire clk,
    input  wire rstn,
    input  wire unit_enable,
    input  wire mac_enable,
    input  wire mem_enable,
    // ... 其他信号
);

// 单元级时钟门控
wire clk_unit;
advanced_clock_gate u_unit_cg (
    .clk_in(clk),
    .enable(unit_enable),
    .clk_out(clk_unit)
);

// MAC阵列时钟门控
wire clk_mac;
advanced_clock_gate u_mac_cg (
    .clk_in(clk_unit),
    .enable(mac_enable),
    .clk_out(clk_mac)
);

// 存储器时钟门控
wire clk_mem;
advanced_clock_gate u_mem_cg (
    .clk_in(clk_unit),
    .enable(mem_enable),
    .clk_out(clk_mem)
);

endmodule
```
