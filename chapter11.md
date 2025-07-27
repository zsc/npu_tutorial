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

## 11.4 数据流优化

### 11.4.1 内存访问模式优化

内存访问是NPU性能的关键瓶颈之一。优化数据流模式可以显著提升性能。

```python
# 数据流优化分析工具
class DataFlowOptimizer:
    def __init__(self, npu_spec):
        self.npu_spec = npu_spec
        self.memory_hierarchy = {
            'l0_buffer': {'size_kb': 32, 'bandwidth_gbps': 2048, 'latency_cycles': 1},
            'l1_cache': {'size_kb': 512, 'bandwidth_gbps': 1024, 'latency_cycles': 3},
            'l2_cache': {'size_kb': 2048, 'bandwidth_gbps': 512, 'latency_cycles': 10},
            'hbm': {'size_gb': 16, 'bandwidth_gbps': 1024, 'latency_cycles': 100}
        }
    
    def analyze_conv2d_dataflow(self, layer_params):
        """分析卷积层的数据流模式"""
        
        # 提取层参数
        N, H, W, C = layer_params['input_shape']
        K, R, S = layer_params['kernel_shape'] 
        OH, OW = layer_params['output_shape'][1:3]
        
        # 计算内存访问量
        input_bytes = N * H * W * C * 2  # FP16
        weight_bytes = K * R * S * C * 2
        output_bytes = N * OH * OW * K * 2
        
        # 分析不同数据流模式
        dataflow_patterns = {
            'weight_stationary': self.analyze_weight_stationary(layer_params),
            'input_stationary': self.analyze_input_stationary(layer_params),
            'output_stationary': self.analyze_output_stationary(layer_params),
            'row_stationary': self.analyze_row_stationary(layer_params)
        }
        
        return {
            'memory_footprint': {
                'input_mb': input_bytes / (1024**2),
                'weight_mb': weight_bytes / (1024**2),
                'output_mb': output_bytes / (1024**2),
                'total_mb': (input_bytes + weight_bytes + output_bytes) / (1024**2)
            },
            'dataflow_analysis': dataflow_patterns,
            'recommendation': self.recommend_dataflow(dataflow_patterns)
        }
    
    def analyze_weight_stationary(self, layer_params):
        """分析权重固定数据流"""
        N, H, W, C = layer_params['input_shape']
        K, R, S = layer_params['kernel_shape']
        OH, OW = layer_params['output_shape'][1:3]
        
        # 权重在PE阵列中保持不变，输入和输出流动
        weight_reuse = N * OH * OW  # 每个权重被重用的次数
        input_reads = N * H * W * C  # 输入读取次数
        output_writes = N * OH * OW * K  # 输出写入次数
        
        # 内存层次访问分析
        l0_accesses = input_reads + output_writes
        l1_accesses = weight_reuse * R * S * C * K // 64  # 假设64个PE
        
        energy_per_access = {
            'l0': 0.2,  # pJ per access
            'l1': 2.0,
            'l2': 10.0,
            'hbm': 50.0
        }
        
        total_energy = (l0_accesses * energy_per_access['l0'] + 
                       l1_accesses * energy_per_access['l1'])
        
        return {
            'pattern': 'weight_stationary',
            'weight_reuse': weight_reuse,
            'total_memory_accesses': l0_accesses + l1_accesses,
            'energy_pj': total_energy,
            'memory_bandwidth_efficiency': self.calculate_bandwidth_efficiency(
                layer_params, 'weight_stationary')
        }
    
    def analyze_input_stationary(self, layer_params):
        """分析输入固定数据流"""
        N, H, W, C = layer_params['input_shape']
        K, R, S = layer_params['kernel_shape']
        OH, OW = layer_params['output_shape'][1:3]
        
        # 输入在PE阵列中保持不变，权重和输出流动
        input_reuse = K  # 每个输入特征被多个输出通道重用
        weight_reads = K * R * S * C
        output_writes = N * OH * OW * K
        
        l0_accesses = weight_reads + output_writes
        l1_accesses = N * H * W * C
        
        energy_per_access = {'l0': 0.2, 'l1': 2.0, 'l2': 10.0, 'hbm': 50.0}
        total_energy = (l0_accesses * energy_per_access['l0'] + 
                       l1_accesses * energy_per_access['l1'])
        
        return {
            'pattern': 'input_stationary',
            'input_reuse': input_reuse,
            'total_memory_accesses': l0_accesses + l1_accesses,
            'energy_pj': total_energy,
            'memory_bandwidth_efficiency': self.calculate_bandwidth_efficiency(
                layer_params, 'input_stationary')
        }
    
    def analyze_output_stationary(self, layer_params):
        """分析输出固定数据流"""
        N, H, W, C = layer_params['input_shape']
        K, R, S = layer_params['kernel_shape']
        OH, OW = layer_params['output_shape'][1:3]
        
        # 输出累积在PE阵列中，输入和权重流动
        output_reuse = R * S * C  # 每个输出被累积的次数
        input_reads = N * H * W * C
        weight_reads = K * R * S * C
        
        l0_accesses = input_reads + weight_reads
        l1_accesses = N * OH * OW * K
        
        energy_per_access = {'l0': 0.2, 'l1': 2.0, 'l2': 10.0, 'hbm': 50.0}
        total_energy = (l0_accesses * energy_per_access['l0'] + 
                       l1_accesses * energy_per_access['l1'])
        
        return {
            'pattern': 'output_stationary',
            'output_reuse': output_reuse,
            'total_memory_accesses': l0_accesses + l1_accesses,
            'energy_pj': total_energy,
            'memory_bandwidth_efficiency': self.calculate_bandwidth_efficiency(
                layer_params, 'output_stationary')
        }
    
    def analyze_row_stationary(self, layer_params):
        """分析行固定数据流（Eyeriss风格）"""
        N, H, W, C = layer_params['input_shape']
        K, R, S = layer_params['kernel_shape']
        OH, OW = layer_params['output_shape'][1:3]
        
        # 卷积窗口的一行在PE阵列中保持不变
        row_reuse = OW  # 行内重用
        col_reuse = R   # 列间重用
        
        # 更复杂的重用模式，综合考虑多维重用
        total_reuse = row_reuse * col_reuse
        
        # 估算内存访问（简化模型）
        input_reads = N * H * W * C / total_reuse
        weight_reads = K * R * S * C / col_reuse
        output_writes = N * OH * OW * K
        
        l0_accesses = input_reads + weight_reads + output_writes
        l1_accesses = l0_accesses * 0.1  # 假设90%的访问命中L0
        
        energy_per_access = {'l0': 0.2, 'l1': 2.0, 'l2': 10.0, 'hbm': 50.0}
        total_energy = (l0_accesses * energy_per_access['l0'] + 
                       l1_accesses * energy_per_access['l1'])
        
        return {
            'pattern': 'row_stationary',
            'total_reuse': total_reuse,
            'total_memory_accesses': l0_accesses + l1_accesses,
            'energy_pj': total_energy,
            'memory_bandwidth_efficiency': self.calculate_bandwidth_efficiency(
                layer_params, 'row_stationary')
        }
    
    def calculate_bandwidth_efficiency(self, layer_params, pattern):
        """计算内存带宽效率"""
        N, H, W, C = layer_params['input_shape']
        K, R, S = layer_params['kernel_shape']
        OH, OW = layer_params['output_shape'][1:3]
        
        # 计算理论计算量
        ops = N * OH * OW * K * R * S * C * 2  # MAC操作数
        
        # 根据数据流模式估算内存访问量
        if pattern == 'weight_stationary':
            memory_accesses = N * H * W * C + N * OH * OW * K
        elif pattern == 'input_stationary':
            memory_accesses = K * R * S * C + N * OH * OW * K
        elif pattern == 'output_stationary':
            memory_accesses = N * H * W * C + K * R * S * C
        else:  # row_stationary
            memory_accesses = (N * H * W * C + K * R * S * C + N * OH * OW * K) * 0.6
        
        # 计算算术强度（Operational Intensity）
        arithmetic_intensity = ops / (memory_accesses * 2)  # bytes per op
        
        # 基于roofline模型计算带宽效率
        peak_bandwidth = self.memory_hierarchy['l1_cache']['bandwidth_gbps']
        peak_compute = self.npu_spec.get('peak_ops_per_sec', 1e12)
        
        # 计算性能上界
        compute_bound_perf = peak_compute
        memory_bound_perf = peak_bandwidth * 1e9 * arithmetic_intensity / 8  # ops/sec
        
        bandwidth_efficiency = min(compute_bound_perf, memory_bound_perf) / peak_compute
        
        return {
            'arithmetic_intensity': arithmetic_intensity,
            'bandwidth_efficiency': bandwidth_efficiency,
            'bottleneck': 'compute' if compute_bound_perf < memory_bound_perf else 'memory'
        }
    
    def recommend_dataflow(self, dataflow_patterns):
        """推荐最优数据流模式"""
        
        # 综合评分函数
        def calculate_score(pattern_analysis):
            energy_score = 1.0 / (pattern_analysis['energy_pj'] / 1e6 + 1)  # 归一化能耗
            bandwidth_score = pattern_analysis['memory_bandwidth_efficiency']['bandwidth_efficiency']
            return 0.6 * energy_score + 0.4 * bandwidth_score
        
        scores = {}
        for pattern_name, analysis in dataflow_patterns.items():
            scores[pattern_name] = calculate_score(analysis)
        
        # 选择最佳模式
        best_pattern = max(scores, key=scores.get)
        
        return {
            'recommended_pattern': best_pattern,
            'scores': scores,
            'rationale': self.get_pattern_rationale(best_pattern)
        }
    
    def get_pattern_rationale(self, pattern):
        """获取模式选择的理由"""
        rationales = {
            'weight_stationary': "适合权重较小、特征图较大的层，减少权重重载开销",
            'input_stationary': "适合输入通道数较少的层，最大化输入重用",
            'output_stationary': "适合卷积核较大的层，减少部分和的传输",
            'row_stationary': "适合大多数卷积层，平衡各维度的数据重用"
        }
        return rationales.get(pattern, "未知模式")

# 使用示例
def optimize_resnet_dataflow():
    npu_spec = {
        'peak_ops_per_sec': 1e12,  # 1 TOPS
        'memory_bandwidth': 1024,  # GB/s
        'pe_array_size': (16, 16)
    }
    
    optimizer = DataFlowOptimizer(npu_spec)
    
    # ResNet-50第一层卷积
    layer1_params = {
        'input_shape': (1, 224, 224, 3),
        'kernel_shape': (64, 7, 7),
        'output_shape': (1, 112, 112, 64)
    }
    
    # ResNet-50中间层卷积
    layer2_params = {
        'input_shape': (1, 56, 56, 64),
        'kernel_shape': (128, 3, 3),
        'output_shape': (1, 28, 28, 128)
    }
    
    print("=== ResNet-50 数据流优化分析 ===")
    
    for i, params in enumerate([layer1_params, layer2_params], 1):
        print(f"\n--- 第{i}层分析 ---")
        analysis = optimizer.analyze_conv2d_dataflow(params)
        
        print(f"内存占用: {analysis['memory_footprint']['total_mb']:.1f} MB")
        
        # 显示各种数据流模式的结果
        for pattern_name, pattern_analysis in analysis['dataflow_analysis'].items():
            print(f"\n{pattern_name}:")
            print(f"  能耗: {pattern_analysis['energy_pj']/1e6:.2f} μJ")
            print(f"  带宽效率: {pattern_analysis['memory_bandwidth_efficiency']['bandwidth_efficiency']:.2%}")
            print(f"  瓶颈: {pattern_analysis['memory_bandwidth_efficiency']['bottleneck']}")
        
        # 推荐模式
        rec = analysis['recommendation']
        print(f"\n推荐模式: {rec['recommended_pattern']}")
        print(f"理由: {rec['rationale']}")

if __name__ == "__main__":
    optimize_resnet_dataflow()
```

## 11.5 练习题

### 练习题11.1：Roofline模型分析
**题目：** 使用Roofline模型分析一个NPU系统的性能瓶颈。给定NPU峰值计算能力为2 TOPS，内存带宽为1TB/s，分析ResNet-50第一层卷积的性能特征。

<details>
<summary>参考答案</summary>

```python
# Roofline模型分析
import numpy as np

class RooflineAnalyzer:
    def __init__(self, peak_performance_tops, memory_bandwidth_gbps):
        self.peak_performance = peak_performance_tops * 1e12  # ops/sec
        self.memory_bandwidth = memory_bandwidth_gbps * 1e9   # bytes/sec
        
    def analyze_workload(self, ops, bytes_transferred):
        """分析工作负载的性能特征"""
        
        # 计算算术强度
        operational_intensity = ops / bytes_transferred  # ops/byte
        
        # 计算Roofline模型的性能上界
        memory_bound_performance = self.memory_bandwidth * operational_intensity
        compute_bound_performance = self.peak_performance
        
        actual_performance = min(memory_bound_performance, compute_bound_performance)
        
        # 确定瓶颈类型
        if memory_bound_performance < compute_bound_performance:
            bottleneck = "Memory Bound"
            efficiency = memory_bound_performance / self.peak_performance
        else:
            bottleneck = "Compute Bound"
            efficiency = 1.0
        
        # 计算ridge point
        ridge_point = self.peak_performance / self.memory_bandwidth
        
        return {
            'operational_intensity': operational_intensity,
            'performance_ops_per_sec': actual_performance,
            'bottleneck': bottleneck,
            'efficiency': efficiency,
            'ridge_point': ridge_point
        }
    
    def analyze_conv2d_layer(self, input_shape, weight_shape, stride=1, padding=0):
        """分析卷积层的Roofline特征"""
        
        N, H, W, C = input_shape
        K, R, S, C = weight_shape
        
        # 计算输出尺寸
        OH = (H + 2*padding - R) // stride + 1
        OW = (W + 2*padding - S) // stride + 1
        
        # 计算操作数（MAC operations）
        ops = N * OH * OW * K * R * S * C * 2
        
        # 计算数据传输量（考虑数据重用）
        input_bytes = N * H * W * C * 2  # FP16
        weight_bytes = K * R * S * C * 2
        output_bytes = N * OH * OW * K * 2
        
        # 考虑数据重用的实际传输量
        input_reuse = K
        weight_reuse = N * OH * OW
        
        actual_input_bytes = input_bytes / input_reuse
        actual_weight_bytes = weight_bytes / weight_reuse
        total_bytes = actual_input_bytes + actual_weight_bytes + output_bytes
        
        return self.analyze_workload(ops, total_bytes)

# ResNet-50第一层分析
analyzer = RooflineAnalyzer(peak_performance_tops=2, memory_bandwidth_gbps=1000)

# ResNet-50第一层参数
input_shape = (1, 224, 224, 3)
weight_shape = (64, 7, 7, 3)
stride = 2
padding = 3

result = analyzer.analyze_conv2d_layer(input_shape, weight_shape, stride, padding)

print("=== ResNet-50第一层Roofline分析 ===")
print(f"算术强度: {result['operational_intensity']:.2f} ops/byte")
print(f"性能上界: {result['performance_ops_per_sec']/1e12:.2f} TOPS")
print(f"瓶颈类型: {result['bottleneck']}")
print(f"硬件效率: {result['efficiency']:.1%}")
print(f"Ridge Point: {result['ridge_point']:.2f} ops/byte")

# 优化建议
if result['bottleneck'] == "Memory Bound":
    print(f"\n=== 优化建议 ===")
    print("1. 增加数据重用，减少内存访问")
    print("2. 使用更高效的数据排列")
    print("3. 考虑算子融合减少中间结果存储")
    print(f"4. 需要将算术强度提高到 {result['ridge_point']:.2f} ops/byte 以上")
else:
    print(f"\n=== 优化建议 ===")
    print("1. 提高计算并行度")
    print("2. 优化算法降低计算复杂度")
    print("3. 考虑使用更低精度计算")
```

**分析结果：**
- ResNet-50第一层的算术强度约为 14.2 ops/byte
- 由于算术强度高于ridge point (2.0 ops/byte)，该层是计算受限的
- 硬件效率为100%，说明能充分利用计算资源
- 优化重点应放在提高计算并行度和降低计算复杂度

</details>

### 练习题11.2：数据流优化
**题目：** 为一个16×16的MAC阵列设计最优的数据流模式，分析不同数据流模式的能耗和性能特征。

<details>
<summary>参考答案</summary>

```python
# 数据流优化分析
class DataflowOptimizer:
    def __init__(self, pe_array_size=(16, 16)):
        self.pe_rows, self.pe_cols = pe_array_size
        self.memory_hierarchy = {
            'RF': {'capacity_kb': 2, 'energy_pj_per_access': 0.1},
            'PE_Buffer': {'capacity_kb': 8, 'energy_pj_per_access': 0.5},
            'NoC': {'energy_pj_per_hop': 0.3},
            'GLB': {'capacity_kb': 512, 'energy_pj_per_access': 5.0},
            'DRAM': {'capacity_gb': 16, 'energy_pj_per_access': 200}
        }
    
    def analyze_weight_stationary(self, conv_params):
        """分析权重固定数据流"""
        N, H, W, C = conv_params['input_shape']
        K, R, S = conv_params['kernel_shape']
        OH, OW = conv_params['output_shape'][1:3]
        
        # 权重在PE中保持不变，输入和输出数据流动
        weight_reuse = N * OH * OW
        input_temporal_reuse = K
        output_spatial_reuse = R * S * C
        
        # 内存访问模式
        weight_accesses = {
            'GLB': K * R * S * C // (self.pe_rows * self.pe_cols),
            'PE_Buffer': K * R * S * C,
            'RF': weight_reuse * K * R * S * C // (self.pe_rows * self.pe_cols)
        }
        
        input_accesses = {
            'GLB': N * H * W * C,
            'NoC': N * H * W * C * self.pe_cols,
            'PE_Buffer': N * H * W * C * K // (self.pe_rows * self.pe_cols),
            'RF': N * H * W * C * K
        }
        
        output_accesses = {
            'RF': N * OH * OW * K * R * S * C // (self.pe_rows * self.pe_cols),
            'PE_Buffer': N * OH * OW * K,
            'GLB': N * OH * OW * K,
        }
        
        # 计算总能耗
        total_energy = self.calculate_energy(weight_accesses, input_accesses, output_accesses)
        
        return {
            'dataflow': 'Weight Stationary',
            'reuse_factors': {
                'weight_reuse': weight_reuse,
                'input_temporal_reuse': input_temporal_reuse,
                'output_spatial_reuse': output_spatial_reuse
            },
            'total_energy_pj': total_energy,
            'pe_utilization': min(K, self.pe_rows) * min(C, self.pe_cols) / (self.pe_rows * self.pe_cols)
        }
    
    def calculate_energy(self, weight_accesses, input_accesses, output_accesses):
        """计算总能耗"""
        total_energy = 0
        
        # 权重访问能耗
        for level, accesses in weight_accesses.items():
            if level in self.memory_hierarchy:
                energy_per_access = self.memory_hierarchy[level]['energy_pj_per_access']
                total_energy += accesses * energy_per_access
        
        # 输入访问能耗
        for level, accesses in input_accesses.items():
            if level in self.memory_hierarchy:
                energy_per_access = (self.memory_hierarchy[level]['energy_pj_per_access'] 
                                   if level != 'NoC' 
                                   else self.memory_hierarchy[level]['energy_pj_per_hop'])
                total_energy += accesses * energy_per_access
        
        # 输出访问能耗
        for level, accesses in output_accesses.items():
            if level in self.memory_hierarchy:
                energy_per_access = self.memory_hierarchy[level]['energy_pj_per_access']
                total_energy += accesses * energy_per_access
        
        return total_energy
    
    def compare_dataflows(self, conv_params):
        """比较不同数据流模式"""
        
        ws_analysis = self.analyze_weight_stationary(conv_params)
        
        print("=== 数据流模式分析 ===")
        print(f"模式: {ws_analysis['dataflow']}")
        print(f"总能耗: {ws_analysis['total_energy_pj']/1e6:.2f} μJ")
        print(f"PE利用率: {ws_analysis['pe_utilization']:.1%}")
        print(f"主要重用因子: {max(ws_analysis['reuse_factors'].items(), key=lambda x: x[1])}")

# 测试分析
optimizer = DataflowOptimizer(pe_array_size=(16, 16))

test_case = {
    'input_shape': (1, 56, 56, 64),
    'kernel_shape': (128, 3, 3),
    'output_shape': (1, 56, 56, 128)
}

optimizer.compare_dataflows(test_case)
```

**分析结果：**

不同数据流模式的特点：

1. **Weight Stationary**：适合权重较小的层
   - 权重重用率高
   - 减少权重加载开销

2. **Output Stationary**：适合大卷积核
   - 输出累积减少部分和传输
   - 适合深度可分离卷积

3. **Row Stationary**：平衡各维度重用
   - 适合大多数常规卷积层
   - Eyeriss采用的方案

选择标准：
- 小卷积核（1×1）：Weight Stationary
- 大卷积核（7×7）：Row Stationary  
- 深度卷积：Output Stationary

</details>