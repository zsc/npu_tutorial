# 第10章：软件栈与编译优化

## 10.1 NPU软件栈架构

### 10.1.1 软件栈的重要性

NPU的硬件性能再强，也需要优秀的软件栈才能充分发挥。软件栈是连接上层AI框架和底层硬件的桥梁，它决定了硬件性能能够被发挥到什么程度。

"硬件定义了性能的上限，而软件决定了能达到这个上限的多少。"这句话完美诠释了NPU软件栈的重要性。Google的研究表明，通过优化TPU的编译器，他们在不改变硬件的情况下将某些工作负载的性能提升了2.8倍。

### 10.1.2 分层架构设计

现代NPU软件栈采用分层架构，每一层专注于特定的优化目标：

```
┌─────────────────────────────────────────┐
│      AI Frameworks (TensorFlow/PyTorch) │
├─────────────────────────────────────────┤
│         Graph Representation            │
│         (ONNX, TorchScript)            │
├─────────────────────────────────────────┤
│         High-Level IR (HIR)            │
│     (Graph Optimization Pass)          │
├─────────────────────────────────────────┤
│         Mid-Level IR (MIR)             │
│    (Operator Fusion, Tiling)          │
├─────────────────────────────────────────┤
│         Low-Level IR (LIR)             │
│   (Memory Allocation, Scheduling)      │
├─────────────────────────────────────────┤
│      Code Generation Backend           │
│    (NPU Instruction Generation)        │
├─────────────────────────────────────────┤
│         Runtime Library                │
│    (Execution, Memory Management)      │
├─────────────────────────────────────────┤
│         NPU Hardware                   │
└─────────────────────────────────────────┘
```

### 10.1.3 关键组件功能

每个软件栈组件都承担着特定的职责：

> **📋 软件栈核心组件**
>
> - **前端解析器：** 支持多种框架模型格式，转换为统一的内部表示
> - **图优化器：** 执行算子融合、常量折叠、死代码消除等优化
> - **量化工具：** 支持训练后量化和量化感知训练
> - **内存分配器：** 优化片上内存使用，最小化数据搬移
> - **指令调度器：** 生成高效的指令序列，最大化硬件利用率
> - **运行时系统：** 管理任务执行、内存管理、多核调度

### 10.1.4 中间表示（IR）设计

中间表示是现代AI编译器的灵魂，NPU编译器通常采用多层IR设计：

```cpp
// 多层IR架构示例
// 1. Graph IR - 高层计算图表示
class GraphIR {
    // 节点表示算子
    struct Node {
        string op_type;        // "Conv2D", "MatMul", "Add", etc.
        vector<Tensor> inputs;
        vector<Tensor> outputs;
        map<string, Attribute> attrs;  // kernel_size, stride, etc.
    };
    
    // 边表示数据流
    struct Edge {
        Node* src;
        Node* dst;
        int src_output_idx;
        int dst_input_idx;
    };
};

// 2. Tensor IR - 张量程序表示
class TensorIR {
    // 类似TVM的张量表达式
    Tensor conv2d_tir(Tensor input, Tensor weight) {
        // 定义计算维度
        auto N = input.shape[0];
        auto H = input.shape[1];
        auto W = input.shape[2];
        auto C = input.shape[3];
        auto K = weight.shape[0];
        
        // 定义输出张量
        Tensor output({N, H-2, W-2, K});
        
        // 定义计算
        output(n, h, w, k) = sum(
            input(n, h+rh, w+rw, c) * weight(k, rh, rw, c),
            {rh, rw, c}  // reduction axes
        );
        
        return output;
    }
};

// 3. Hardware IR - 硬件指令表示
class HardwareIR {
    enum OpCode {
        LOAD_WEIGHT,    // 加载权重到片上
        LOAD_ACT,       // 加载激活值
        COMPUTE_MAC,    // MAC阵列计算
        STORE_RESULT,   // 存储结果
        SYNC            // 同步指令
    };
    
    struct Instruction {
        OpCode opcode;
        vector<int> operands;
        map<string, int> config;  // 硬件配置参数
    };
};
```

> **💡 为什么需要多层IR？**
>
> - **Graph IR：** 适合做图级别优化，如算子融合、常量折叠、死代码消除
> - **Tensor IR：** 适合做算子内部优化，如循环变换、向量化、内存访问优化
> - **Hardware IR：** 贴近硬件，便于指令调度、寄存器分配、硬件特性利用
>
> 现代框架如**MLIR（Multi-Level IR）**提供了构建多层IR的基础设施，被Google、Intel等公司广泛采用。

## 10.2 计算图优化

### 10.2.1 算子融合技术

算子融合是提升NPU性能最有效的优化技术之一，通过将多个独立的计算操作合并为一个复合操作，减少内存访问次数。

```python
# 算子融合示例：Conv + BN + ReLU融合
# 原始计算图
class OriginalGraph:
    def forward(self, x):
        # 卷积操作
        conv_out = self.conv2d(x)  # 需要写回内存
        # 批归一化
        bn_out = self.batch_norm(conv_out)  # 需要读写内存
        # 激活函数
        relu_out = self.relu(bn_out)  # 需要读写内存
        return relu_out

# 融合后的计算图
class FusedGraph:
    def forward(self, x):
        # 融合的算子，一次内存读写完成三个操作
        return self.conv_bn_relu_fused(x)

# 融合实现（伪代码）
def conv_bn_relu_fused(input, conv_weight, bn_params):
    # 在NPU内部完成所有计算
    for (oc in output_channels):
        for (oh, ow in output_positions):
            # 卷积计算
            acc = 0
            for (ic, kh, kw in kernel):
                acc += input[ic][oh+kh][ow+kw] * conv_weight[oc][ic][kh][kw]
            
            # BN计算（在线融合）
            acc = (acc - bn_mean[oc]) / sqrt(bn_var[oc] + eps)
            acc = acc * bn_scale[oc] + bn_bias[oc]
            
            # ReLU计算
            output[oc][oh][ow] = max(0, acc)
    
    return output
```

### 10.2.2 算子融合的类型与限制

```cpp
// 不同类型的算子融合模式
// 1. 垂直融合（Vertical Fusion）- 将element-wise操作融入计算密集型操作
class VerticalFusion {
    // 融合前：Conv -> Add(bias) -> BN -> ReLU
    void unfused_forward(Tensor input) {
        Tensor conv_out = conv2d(input, weight);      // 写回DDR
        Tensor bias_out = add(conv_out, bias);        // 读写DDR
        Tensor bn_out = batch_norm(bias_out);         // 读写DDR
        Tensor relu_out = relu(bn_out);               // 读写DDR
        return relu_out;
    }
    
    // 融合后：所有操作在片上完成
    void fused_forward(Tensor input) {
        // 一次性完成所有计算，只写最终结果
        return conv_bias_bn_relu_fused(input, weight, bias, bn_params);
    }
};

// 2. 水平融合（Horizontal Fusion）- 合并相同类型的并行操作
class HorizontalFusion {
    // 融合前：多个小矩阵乘法分别执行
    void unfused_multi_matmul(vector<Tensor> A_list, vector<Tensor> B_list) {
        vector<Tensor> results;
        for (int i = 0; i < A_list.size(); i++) {
            results.push_back(matmul(A_list[i], B_list[i]));
        }
        return results;
    }
    
    // 融合后：打包成一个大矩阵乘法
    void fused_batched_matmul(vector<Tensor> A_list, vector<Tensor> B_list) {
        Tensor A_packed = pack_tensors(A_list);  // [batch, M, K]
        Tensor B_packed = pack_tensors(B_list);  // [batch, K, N]
        Tensor C_packed = batched_matmul(A_packed, B_packed);
        return unpack_tensors(C_packed);
    }
};

// 3. 融合的限制条件
bool can_fuse(Node* node1, Node* node2) {
    // 检查数据依赖
    if (has_external_dependency(node1, node2)) {
        return false;  // 中间结果被其他节点使用
    }
    
    // 检查内存限制
    size_t fused_memory = estimate_memory(node1) + estimate_memory(node2);
    if (fused_memory > on_chip_memory_size) {
        return false;  // 融合后超出片上内存
    }
    
    // 检查硬件支持
    if (!hardware_supports_fused_op(node1->op_type, node2->op_type)) {
        return false;  // 硬件没有对应的融合指令
    }
    
    // 检查数值稳定性
    if (fusion_affects_numerical_stability(node1, node2)) {
        return false;  // 融合可能导致精度损失
    }
    
    return true;
}
```

> **⚠️ 算子融合的权衡**
>
> **收益：** 减少内存访问、降低带宽压力、减少kernel启动开销
>
> **代价：** 增加代码复杂度、可能降低硬件利用率、限制并行度
>
> **原则：** 优先融合内存受限（memory-bound）的操作，计算受限（compute-bound）的操作谨慎融合

## 10.3 内存优化技术

### 10.3.1 内存分配策略

NPU的片上内存通常有限且昂贵，高效的内存管理是性能优化的关键。

```cpp
class NPUMemoryAllocator {
    // 内存池管理
    struct MemoryPool {
        size_t total_size;
        size_t free_size;
        std::vector<MemoryBlock> free_blocks;
        std::map<void*, MemoryBlock> allocated_blocks;
    };
    
    // 不同类型的内存池
    MemoryPool weight_memory;      // 权重专用内存
    MemoryPool activation_memory;  // 激活值内存
    MemoryPool scratch_memory;     // 临时计算内存
    
public:
    // 智能内存分配
    void* allocate(size_t size, MemoryType type, AlignmentRequirement align) {
        // 选择合适的内存池
        MemoryPool& pool = select_pool(type);
        
        // 尝试复用现有内存块
        auto reusable_block = find_reusable_block(pool, size, align);
        if (reusable_block != nullptr) {
            return reusable_block;
        }
        
        // 分配新内存块
        return allocate_new_block(pool, size, align);
    }
    
    // 内存生命周期分析
    void analyze_memory_lifetime(ComputeGraph& graph) {
        std::map<Tensor*, std::pair<int, int>> lifetimes;
        
        // 分析每个tensor的生命周期
        for (int i = 0; i < graph.nodes.size(); i++) {
            auto& node = graph.nodes[i];
            
            // 输入tensor生命周期开始
            for (auto& input : node.inputs) {
                if (lifetimes.find(input) == lifetimes.end()) {
                    lifetimes[input].first = i;
                }
                lifetimes[input].second = i;  // 更新结束时间
            }
            
            // 输出tensor生命周期开始
            for (auto& output : node.outputs) {
                lifetimes[output].first = i;
            }
        }
        
        // 基于生命周期进行内存复用
        schedule_memory_reuse(lifetimes);
    }
    
private:
    void schedule_memory_reuse(const std::map<Tensor*, std::pair<int, int>>& lifetimes) {
        // 使用区间调度算法优化内存复用
        std::vector<std::pair<Tensor*, std::pair<int, int>>> sorted_tensors(
            lifetimes.begin(), lifetimes.end());
        
        // 按生命周期结束时间排序
        std::sort(sorted_tensors.begin(), sorted_tensors.end(),
            [](const auto& a, const auto& b) {
                return a.second.second < b.second.second;
            });
        
        // 分配内存槽位
        std::vector<int> memory_slots;
        for (auto& [tensor, lifetime] : sorted_tensors) {
            int slot = find_available_slot(memory_slots, lifetime.first);
            assign_tensor_to_slot(tensor, slot);
            memory_slots[slot] = lifetime.second;
        }
    }
};
```

### 10.3.2 数据布局优化

```cpp
// 数据布局变换优化
class DataLayoutOptimizer {
    // 常见的数据布局格式
    enum LayoutFormat {
        NCHW,    // 适合GPU计算
        NHWC,    // 适合移动端NPU
        NC4HW4,  // 4通道对齐格式
        NC8HW8,  // 8通道对齐格式
        NCHW_TO_NC4HW4  // 布局转换
    };
    
    // 为每个操作选择最优布局
    LayoutFormat select_optimal_layout(OpType op, HardwareSpec hw) {
        switch (op) {
            case CONV2D:
                if (hw.has_winograd_support) {
                    return select_winograd_layout(hw);
                } else if (hw.vector_width == 4) {
                    return NC4HW4;
                } else if (hw.vector_width == 8) {
                    return NC8HW8;
                }
                break;
                
            case MATMUL:
                // 矩阵乘法偏好行主序或列主序
                return hw.prefers_row_major ? ROW_MAJOR : COL_MAJOR;
                
            case ELEMENTWISE:
                // 逐元素操作偏好连续内存布局
                return CONTIGUOUS;
        }
        return NCHW;  // 默认布局
    }
    
    // 布局转换的代价估计
    float estimate_layout_conversion_cost(LayoutFormat from, LayoutFormat to, 
                                        TensorShape shape) {
        // 计算数据重排的内存访问代价
        size_t total_elements = shape.total_elements();
        
        if (is_simple_transpose(from, to)) {
            // 简单转置：2倍内存访问
            return total_elements * 2 * sizeof(float);
        } else if (requires_padding(from, to)) {
            // 需要填充：额外的内存和计算开销
            float padding_ratio = calculate_padding_ratio(from, to, shape);
            return total_elements * (1 + padding_ratio) * 2 * sizeof(float);
        } else {
            // 复杂重排：可能需要多次pass
            return total_elements * 4 * sizeof(float);
        }
    }
    
    // 全局布局优化
    void optimize_global_layout(ComputeGraph& graph) {
        // 构建布局传播图
        std::map<Node*, LayoutFormat> node_layouts;
        std::map<Edge*, float> conversion_costs;
        
        // 为每个节点选择候选布局
        for (auto& node : graph.nodes) {
            auto candidates = get_layout_candidates(node);
            node_layouts[&node] = select_best_layout(candidates);
        }
        
        // 最小化总的转换代价
        optimize_conversion_costs(graph, node_layouts, conversion_costs);
        
        // 插入必要的布局转换节点
        insert_layout_conversion_nodes(graph, node_layouts);
    }
};
```