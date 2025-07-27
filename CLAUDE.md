（交流可以用英文，本文档中文，保留这句）

# Neural Processing Unit (NPU) 设计教程项目说明

## 项目目标
编写一份 Neural Processing Unit 设计的由浅入深，包含芯片前后端设计的教程，要包含大量的习题和参考答案。项目已完成HTML到Markdown格式的转换。

## 工具说明
当需要时，可以通过 `gemini -p "深入回答：<要问的问题> -m gemini-2.5-pro"` 来获取 gemini-2.5-pro 的参考意见(只问 2.5-pro 不问别人)

## 教程大纲

### 第一部分：NPU基础概念
1. **NPU简介与发展历程**
   - 1.1 什么是NPU
   - 1.2 NPU vs CPU vs GPU
   - 1.3 NPU的应用场景
   - 1.4 主流NPU架构概览
   - 习题集1

2. **神经网络计算基础**
   - 2.1 神经网络基本运算
   - 2.2 矩阵乘法与卷积运算
   - 2.3 激活函数与量化
   - 2.4 数据流与并行计算
   - 习题集2

### 第二部分：NPU架构设计
3. **NPU系统架构**
   - 3.1 整体架构设计
   - 3.2 计算单元设计
   - 3.3 存储层次结构
   - 3.4 互连网络设计
   - 习题集3

4. **计算核心设计**
   - 4.1 MAC阵列设计
   - 4.2 脉动阵列架构
   - 4.3 向量处理单元
   - 4.4 特殊功能单元
   - 习题集4

5. **存储系统设计**
   - 5.1 片上存储架构
   - 5.2 数据重用策略
   - 5.3 带宽优化技术
   - 5.4 存储访问调度
   - 习题集5

### 第三部分：NPU前端设计
6. **RTL设计实现**
   - 6.1 Verilog/SystemVerilog基础
   - 6.2 计算单元RTL实现
   - 6.3 控制逻辑设计
   - 6.4 时序优化技术
   - 习题集6

7. **验证与测试**
   - 7.1 功能验证策略
   - 7.2 UVM验证环境搭建
   - 7.3 覆盖率驱动验证
   - 7.4 形式化验证
   - 习题集7

### 第四部分：NPU后端设计
8. **物理设计**
   - 8.1 ASIC设计流程
   - 8.2 综合与时序分析
   - 8.3 布局布线
   - 8.4 功耗分析与优化
   - 习题集8

9. **先进工艺与封装技术**
   - 9.1 FinFET工艺特点
   - 9.2 时钟树综合
   - 9.3 电源网络设计
   - 9.4 信号完整性
   - 习题集9

### 第五部分：系统集成与优化
10. **软件栈与编译优化**
    - 10.1 编译器设计基础
    - 10.2 指令集架构
    - 10.3 调度与映射算法
    - 10.4 性能分析工具
    - 习题集10

11. **性能优化技术**（待实现）
    - 11.1 算法优化
    - 11.2 数据流优化
    - 11.3 功耗优化
    - 11.4 面积优化
    - 习题集11

### 第六部分：实战项目（待实现）
12. **NPU设计实战**
    - 12.1 项目需求分析
    - 12.2 架构设计实践
    - 12.3 RTL实现与验证
    - 12.4 综合与后端实现
    - 综合项目

## 教程特色
- 每章节包含理论讲解、实例分析、代码示例
- 大量习题覆盖概念理解、设计实践、问题解决
- 答案默认折叠，支持自主学习
- 渐进式学习路径，从基础到高级
- 结合工业界最佳实践

## 实施计划
1. **第一阶段**：创建HTML模板和基础结构（已完成）
2. **第二阶段**：逐章节编写内容，每章包含：
   - 理论知识讲解
   - 图表和架构图
   - RTL代码示例
   - 5-10道练习题
   - 可折叠的参考答案
3. **第三阶段**：添加交互功能和美化界面

## 当前进度
- 已完成章节：第1-12章全部内容和习题
- Transformer相关内容：已添加到第2、3、4章
- 格式转换：已完成HTML到Markdown格式转换，所有章节现为.md文件
- 项目状态：NPU设计教程已完整实现并转换为Markdown格式，包含理论讲解、RTL代码示例和丰富的练习题

## HTML到Markdown转换方案

### 转换流程总结
基于chapter1的成功转换经验，标准化转换流程如下：

1. **逐章手动转换**（不使用脚本，避免内容丢失）
2. **保留所有内容**：
   - 所有文本内容
   - 代码块（使用```语法）
   - 表格（使用markdown表格语法）
   - 练习题和答案（使用`<details>`标签）
   - 图片说明（转为文字描述）
3. **分块更新**（避免大文件操作错误）：
   - 将长章节分成多个部分逐步写入
   - 每次更新一个完整的小节
   - 确保内容完整性
4. **格式转换规则**：
   - HTML标题 `<h2>` → Markdown `##`
   - HTML标题 `<h3>` → Markdown `###`
   - HTML标题 `<h4>` → Markdown `####`
   - 代码块 `<pre><code>` → ` ```language...``` `
   - 粗体 `<strong>` → `**text**`
   - 列表保持原有格式
   - 折叠内容使用 `<details><summary>标题</summary>内容</details>`
4. **验证要点**：
   - 确保所有小节都被转换（如1.4.1-1.4.12）
   - 验证练习题数量与HTML一致
   - 检查代码块语法高亮标记正确

## HTML到Markdown转换完成状态
- [x] 转换 chapter1.html → chapter1.md（已完成，包含所有14个练习题）
- [x] 创建 index.md 导航页面（已完成）
- [x] 转换 chapter2.html → chapter2.md（已完成）
- [x] 转换 chapter3.html → chapter3.md（已完成）
- [x] 转换 chapter4.html → chapter4.md（已完成）
- [x] 转换 chapter5.html → chapter5.md（已完成）
- [x] 转换 chapter6.html → chapter6.md（已完成）
- [x] 转换 chapter7.html → chapter7.md（已完成）
- [x] 转换 chapter8.html → chapter8.md（已完成）
- [x] 转换 chapter9.html → chapter9.md（已完成）
- [x] 转换 chapter10.html → chapter10.md（已完成）
- [x] 转换 chapter11.html → chapter11.md（已完成）
- [x] 转换 chapter12.html → chapter12.md（已完成）
- [x] 删除所有.html文件（转换完成后）
- [x] 更新文件结构说明为Markdown版本

## 文件结构说明（Markdown版本）

本教程已转换为Markdown格式的多文件结构，便于版本控制和维护。

### 文件列表

- `index.html` - 主导航页面（HTML格式保留，用于网页浏览）
- `chapter1.md` - 第1章：NPU简介与发展历程
- `chapter2.md` - 第2章：神经网络计算基础
- `chapter3.md` - 第3章：NPU系统架构
- `chapter4.md` - 第4章：计算核心设计
- `chapter5.md` - 第5章：存储系统设计
- `chapter6.md` - 第6章：RTL设计实现
- `chapter7.md` - 第7章：验证与测试
- `chapter8.md` - 第8章：物理设计
- `chapter9.md` - 第9章：先进工艺与封装技术
- `chapter10.md` - 第10章：软件栈与编译优化
- `chapter11.md` - 第11章：性能优化技术
- `chapter12.md` - 第12章：NPU设计实战
- `CLAUDE.md` - 项目说明文档

### 使用方法

1. **GitHub/GitLab查看**：直接在代码仓库中查看Markdown文件
2. **本地查看**：使用Markdown编辑器（如Typora、VSCode）打开
3. **转换为其他格式**：可使用pandoc等工具转换为PDF、HTML等格式

### Markdown特性

- **标准格式**：遵循CommonMark规范
- **代码高亮**：支持语法高亮（verilog、python、cpp等）
- **数学公式**：支持LaTeX数学公式
- **表格支持**：完整的表格格式
- **版本控制友好**：适合Git版本管理
- **跨平台兼容**：可在任何支持Markdown的平台查看

### 内容特色

- **理论与实践结合**：每章包含理论讲解和代码示例
- **丰富的练习题**：每章5-10道练习题及详细答案
- **完整的设计流程**：从系统架构到物理实现
- **工业界最佳实践**：结合实际项目经验

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
