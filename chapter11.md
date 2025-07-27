# 第11章：性能优化技术

## 11.1 性能分析与建模

### 11.1.1 NPU性能瓶颈分析

NPU性能优化是一个系统工程，需要从算法、编译器、硬件等多个层面进行协同优化。理解性能瓶颈是优化的第一步。

**NPU常见性能瓶颈：**

1. **计算瓶颈（Compute Bound）**
   - MAC阵列利用率不足
   - 数据类型不匹配（如FP32在INT8硬件上）
   - 算法本身的计算复杂度过高

2. **内存瓶颈（Memory Bound）**
   - 片上内存容量限制
   - 内存带宽不足
   - 数据复用率低

3. **通信瓶颈（Communication Bound）**
   - 多核间数据传输开销
   - 主机与NPU间的PCIe带宽
   - 网络通信延迟（分布式训练）

```python
# NPU性能分析框架
class NPUPerformanceAnalyzer:
    def __init__(self, npu_spec):
        self.compute_ops_per_sec = npu_spec.peak_ops  # TOPS
        self.memory_bandwidth = npu_spec.memory_bw    # GB/s
        self.on_chip_memory = npu_spec.sram_size      # MB
        
    def analyze_workload(self, model):
        """分析模型的计算和内存特征"""
        analysis = {}
        
        for layer in model.layers:
            # 计算操作数
            compute_ops = self.calculate_ops(layer)
            
            # 内存访问量
            memory_access = self.calculate_memory_access(layer)
            
            # 计算强度（OI: Operational Intensity）
            operational_intensity = compute_ops / memory_access
            
            # 性能上界分析
            compute_bound_perf = compute_ops / self.compute_ops_per_sec
            memory_bound_perf = memory_access / self.memory_bandwidth
            
            # roofline模型预测
            predicted_perf = max(compute_bound_perf, memory_bound_perf)
            
            analysis[layer.name] = {
                'ops': compute_ops,
                'memory_access': memory_access,
                'operational_intensity': operational_intensity,
                'bottleneck': 'compute' if compute_bound_perf > memory_bound_perf else 'memory',
                'predicted_time': predicted_perf
            }
        
        return analysis
    
    def roofline_model(self, operational_intensity):
        """Roofline性能模型"""
        # 计算屋顶线
        compute_roof = self.compute_ops_per_sec
        memory_roof = self.memory_bandwidth * operational_intensity
        
        # 性能上界由较小者决定
        performance_ceiling = min(compute_roof, memory_roof)
        
        return performance_ceiling
```

### 11.1.2 性能建模技术

```cpp
// 分析工具：Roofline模型实现
class RooflineModel {
    struct HardwareSpec {
        double peak_compute;     // FLOPS
        double peak_bandwidth;   // Bytes/sec
        double l1_bandwidth;     // L1缓存带宽
        double l2_bandwidth;     // L2缓存带宽
    };
    
    HardwareSpec hw_spec;
    
public:
    // 计算给定工作负载的性能上界
    double predict_performance(double ops, double bytes_accessed) {
        double operational_intensity = ops / bytes_accessed;
        
        // 多级存储的Roofline模型
        double l1_roof = hw_spec.l1_bandwidth * operational_intensity;
        double l2_roof = hw_spec.l2_bandwidth * operational_intensity;
        double ddr_roof = hw_spec.peak_bandwidth * operational_intensity;
        
        // 性能由最严格的约束决定
        double memory_roof = std::max({l1_roof, l2_roof, ddr_roof});
        double performance = std::min(hw_spec.peak_compute, memory_roof);
        
        return performance;
    }
    
    // 识别性能瓶颈
    std::string identify_bottleneck(double ops, double bytes_accessed) {
        double oi = ops / bytes_accessed;
        double ridge_point = hw_spec.peak_compute / hw_spec.peak_bandwidth;
        
        if (oi < ridge_point) {
            return "Memory Bound";
        } else {
            return "Compute Bound";
        }
    }
};
```

## 11.2 算法层优化

### 11.2.1 模型压缩技术

```python
# 模型压缩技术实现
class ModelCompression:
    
    @staticmethod
    def quantize_weights(model, bits=8):
        """权重量化"""
        for layer in model.layers:
            if hasattr(layer, 'weight'):
                # 计算量化参数
                w_min = layer.weight.min()
                w_max = layer.weight.max()
                scale = (w_max - w_min) / (2**bits - 1)
                zero_point = -w_min / scale
                
                # 执行量化
                w_quantized = torch.round(layer.weight / scale + zero_point)
                w_quantized = torch.clamp(w_quantized, 0, 2**bits - 1)
                
                # 存储量化参数
                layer.weight_scale = scale
                layer.weight_zero_point = zero_point
                layer.weight = w_quantized
    
    @staticmethod
    def prune_weights(model, sparsity_ratio=0.5):
        """权重剪枝"""
        for layer in model.layers:
            if hasattr(layer, 'weight'):
                # 计算重要性分数（这里使用L1范数）
                importance = torch.abs(layer.weight)
                
                # 确定剪枝阈值
                threshold = torch.quantile(importance, sparsity_ratio)
                
                # 创建掩码
                mask = importance > threshold
                
                # 应用剪枝
                layer.weight = layer.weight * mask
                layer.weight_mask = mask
    
    @staticmethod
    def knowledge_distillation(teacher_model, student_model, data_loader, 
                             temperature=3.0, alpha=0.7):
        """知识蒸馏"""
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kd = nn.KLDivLoss(reduction='batchmean')
        
        for batch_idx, (data, target) in enumerate(data_loader):
            # 教师模型输出（不更新梯度）
            with torch.no_grad():
                teacher_output = teacher_model(data)
                teacher_prob = F.softmax(teacher_output / temperature, dim=1)
            
            # 学生模型输出
            student_output = student_model(data)
            student_log_prob = F.log_softmax(student_output / temperature, dim=1)
            student_prob = F.softmax(student_output, dim=1)
            
            # 组合损失：硬标签损失 + 软标签损失
            hard_loss = criterion_ce(student_output, target)
            soft_loss = criterion_kd(student_log_prob, teacher_prob) * (temperature ** 2)
            
            total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

### 11.2.2 神经架构搜索(NAS)

```python
# 面向NPU的神经架构搜索
class NPUAwareNAS:
    def __init__(self, npu_constraints):
        self.max_latency = npu_constraints.max_latency
        self.max_memory = npu_constraints.max_memory
        self.preferred_ops = npu_constraints.efficient_ops
    
    def search_architecture(self, search_space, num_iterations=1000):
        """搜索最优架构"""
        best_arch = None
        best_score = float('-inf')
        
        for iteration in range(num_iterations):
            # 采样候选架构
            candidate_arch = self.sample_architecture(search_space)
            
            # 评估架构性能
            accuracy = self.evaluate_accuracy(candidate_arch)
            latency = self.estimate_latency(candidate_arch)
            memory = self.estimate_memory(candidate_arch)
            
            # 检查约束
            if latency > self.max_latency or memory > self.max_memory:
                continue
            
            # 计算综合评分
            score = self.calculate_score(accuracy, latency, memory)
            
            if score > best_score:
                best_score = score
                best_arch = candidate_arch
        
        return best_arch
    
    def estimate_latency(self, architecture):
        """基于NPU特性估计延迟"""
        total_latency = 0
        
        for layer in architecture.layers:
            # 获取算子类型
            op_type = layer.op_type
            
            if op_type in self.preferred_ops:
                # NPU高效算子
                base_latency = self.get_base_latency(op_type, layer.params)
                efficiency_factor = 1.0
            else:
                # 需要CPU fallback的算子
                base_latency = self.get_cpu_latency(op_type, layer.params)
                efficiency_factor = 10.0  # CPU fallback开销大
            
            # 考虑数据传输开销
            data_transfer_cost = self.estimate_data_transfer(layer)
            
            layer_latency = base_latency * efficiency_factor + data_transfer_cost
            total_latency += layer_latency
        
        return total_latency
    
    def calculate_score(self, accuracy, latency, memory):
        """多目标优化评分函数"""
        # 归一化各个指标
        norm_accuracy = accuracy / 100.0  # 假设精度在0-100之间
        norm_latency = self.max_latency / latency  # 延迟越小越好
        norm_memory = self.max_memory / memory     # 内存越小越好
        
        # 加权求和
        score = 0.6 * norm_accuracy + 0.3 * norm_latency + 0.1 * norm_memory
        
        return score
```

## 11.3 编译器优化

### 11.3.1 循环优化技术

```python
# 编译器循环优化
class LoopOptimizer:
    
    @staticmethod
    def tile_loops(loop_nest, tile_sizes):
        """循环分块优化"""
        # 将大循环分解为小的tile，提高缓存局部性
        optimized_code = f"""
        for (int i_outer = 0; i_outer < M; i_outer += {tile_sizes[0]}):
            for (int j_outer = 0; j_outer < N; j_outer += {tile_sizes[1]}):
                for (int k_outer = 0; k_outer < K; k_outer += {tile_sizes[2]}):
                    for (int i = i_outer; i < min(i_outer + {tile_sizes[0]}, M); i++):
                        for (int j = j_outer; j < min(j_outer + {tile_sizes[1]}, N); j++):
                            for (int k = k_outer; k < min(k_outer + {tile_sizes[2]}, K); k++):
                                C[i][j] += A[i][k] * B[k][j];
        """
        return optimized_code
    
    @staticmethod
    def vectorize_loops(loop, vector_width):
        """循环向量化"""
        if loop.is_vectorizable():
            # 将标量操作转换为向量操作
            vectorized_ops = []
            for i in range(0, loop.trip_count, vector_width):
                chunk_size = min(vector_width, loop.trip_count - i)
                vectorized_ops.append(f"vec_op({chunk_size}, &data[{i}])")
            return vectorized_ops
        return loop.original_ops
    
    @staticmethod
    def unroll_loops(loop, unroll_factor):
        """循环展开"""
        unrolled_body = []
        for i in range(unroll_factor):
            # 复制循环体，更新索引
            body_copy = loop.body.copy()
            body_copy.update_indices(i)
            unrolled_body.append(body_copy)
        
        return unrolled_body
```

### 11.3.2 指令级并行优化

```cpp
// 指令调度优化
class InstructionScheduler {
    struct Instruction {
        OpCode opcode;
        std::vector<int> src_regs;
        int dst_reg;
        int latency;
        int issue_cycle;
    };
    
    // 数据依赖图
    class DependencyGraph {
        std::vector<Instruction> instructions;
        std::vector<std::vector<int>> dependencies;
        
    public:
        void add_dependency(int src_inst, int dst_inst) {
            dependencies[src_inst].push_back(dst_inst);
        }
        
        std::vector<int> get_ready_instructions(int current_cycle) {
            std::vector<int> ready;
            for (int i = 0; i < instructions.size(); i++) {
                if (is_ready(i, current_cycle)) {
                    ready.push_back(i);
                }
            }
            return ready;
        }
    };
    
    // 列表调度算法
    std::vector<Instruction> schedule_instructions(
        const std::vector<Instruction>& input_instructions) {
        
        DependencyGraph dep_graph(input_instructions);
        std::vector<Instruction> scheduled;
        std::vector<bool> scheduled_mask(input_instructions.size(), false);
        
        int current_cycle = 0;
        while (scheduled.size() < input_instructions.size()) {
            // 获取当前周期可调度的指令
            auto ready_insts = dep_graph.get_ready_instructions(current_cycle);
            
            // 选择优先级最高的指令
            if (!ready_insts.empty()) {
                int selected = select_highest_priority(ready_insts);
                
                auto& inst = input_instructions[selected];
                inst.issue_cycle = current_cycle;
                scheduled.push_back(inst);
                scheduled_mask[selected] = true;
            }
            
            current_cycle++;
        }
        
        return scheduled;
    }
    
private:
    int select_highest_priority(const std::vector<int>& candidates) {
        // 优先级启发式：
        // 1. 关键路径上的指令优先
        // 2. 延迟长的指令优先
        // 3. 依赖关系少的指令优先
        
        int best_candidate = candidates[0];
        int highest_priority = calculate_priority(best_candidate);
        
        for (int i = 1; i < candidates.size(); i++) {
            int priority = calculate_priority(candidates[i]);
            if (priority > highest_priority) {
                highest_priority = priority;
                best_candidate = candidates[i];
            }
        }
        
        return best_candidate;
    }
};
```

## 11.4 硬件协同优化

### 11.4.1 软硬件协同设计

```cpp
// 软硬件协同优化框架
class HardwareSoftwareCoDesign {
    struct HardwareConfig {
        int mac_array_size;
        int on_chip_memory_kb;
        int memory_bandwidth_gbps;
        std::vector<OpType> supported_ops;
    };
    
    struct SoftwareConfig {
        int batch_size;
        int tile_size;
        SchedulingPolicy policy;
        std::vector<OptimizationPass> passes;
    };
    
public:
    // 联合优化硬件配置和软件策略
    std::pair<HardwareConfig, SoftwareConfig> co_optimize(
        const WorkloadCharacteristics& workload,
        const DesignConstraints& constraints) {
        
        HardwareConfig best_hw_config;
        SoftwareConfig best_sw_config;
        double best_efficiency = 0.0;
        
        // 搜索硬件配置空间
        for (auto& hw_config : generate_hw_candidates(constraints)) {
            // 为每个硬件配置优化软件
            auto sw_config = optimize_software(hw_config, workload);
            
            // 评估整体效率
            double efficiency = evaluate_efficiency(hw_config, sw_config, workload);
            
            if (efficiency > best_efficiency) {
                best_efficiency = efficiency;
                best_hw_config = hw_config;
                best_sw_config = sw_config;
            }
        }
        
        return {best_hw_config, best_sw_config};
    }
    
private:
    SoftwareConfig optimize_software(const HardwareConfig& hw_config,
                                   const WorkloadCharacteristics& workload) {
        SoftwareConfig sw_config;
        
        // 基于硬件特性选择最优的tile size
        sw_config.tile_size = select_optimal_tile_size(
            hw_config.on_chip_memory_kb, workload.tensor_sizes);
        
        // 选择最优的batch size
        sw_config.batch_size = select_optimal_batch_size(
            hw_config.mac_array_size, workload.parallelism);
        
        // 配置编译器优化pass
        sw_config.passes = configure_optimization_passes(
            hw_config.supported_ops, workload.operation_types);
        
        return sw_config;
    }
    
    double evaluate_efficiency(const HardwareConfig& hw_config,
                             const SoftwareConfig& sw_config,
                             const WorkloadCharacteristics& workload) {
        // 计算硬件利用率
        double compute_utilization = calculate_compute_utilization(
            hw_config, sw_config, workload);
        
        // 计算内存效率
        double memory_efficiency = calculate_memory_efficiency(
            hw_config, sw_config, workload);
        
        // 综合评分
        return 0.6 * compute_utilization + 0.4 * memory_efficiency;
    }
};
```