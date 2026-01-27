在推荐系统中，归一化层通过稳定训练过程提升模型性能。以下是常见归一化层的原理、公式、输入输出维度及PyTorch实现，结合推荐系统场景说明：

### **1. Batch Normalization (BN)**
**原理**：对每个批次的同一通道（特征）进行归一化，解决内部协变量偏移问题。  
**公式**：  
<img width="1510" height="776" alt="image" src="https://github.com/user-attachments/assets/f575ae19-b3c1-4100-b49d-c6ee66f9e36f" />

**输入输出维度**：  
- 输入：`(batch_size, num_features, seq_length)`（如用户行为序列，`num_features`为特征数，`seq_length`为序列长度）。  
- 输出：维度与输入相同。  
**推荐系统场景**：  
- 适用于用户行为序列建模（如RNN/Transformer的输入层），但需批次大小稳定。若序列长度差异大，效果可能下降。  
**PyTorch实现**：  
```python
import torch
import torch.nn as nn

bn = nn.BatchNorm1d(num_features=64)  # 对64个特征归一化
input = torch.randn(32, 64, 10)      # (batch_size=32, num_features=64, seq_length=10)
output = bn(input.permute(0, 2, 1))  # 调整维度为 (32, 10, 64) 以匹配BN输入要求
output = output.permute(0, 2, 1)     # 恢复原维度
```

### **2. Layer Normalization (LN)**
**原理**：对单个样本的所有特征进行归一化，不依赖批次大小。  
**公式**：  
<img width="1608" height="382" alt="image" src="https://github.com/user-attachments/assets/df408511-91d8-4996-84f2-25201dfb74fb" />
 
**输入输出维度**：  
- 输入：`(batch_size, num_features, seq_length)`。  
- 输出：维度与输入相同。  
**推荐系统场景**：  
- 适用于Transformer或RNN的序列建模（如用户行为序列），因序列长度可变，LN更稳定。  
**PyTorch实现**：  
```python
ln = nn.LayerNorm(normalized_shape=64)  # 对64个特征归一化
input = torch.randn(32, 64, 10)       # (batch_size=32, num_features=64, seq_length=10)
output = ln(input)  # 直接应用，无需维度调整
```

### **3. Instance Normalization (IN)**
**原理**：对每个样本的每个通道单独归一化，常用于风格迁移。  
**公式**：  
<img width="844" height="199" alt="截屏2026-01-27 15 05 28" src="https://github.com/user-attachments/assets/c0d2b4a6-a360-463f-8c35-e706026eb801" />

**输入输出维度**：  
- 输入：`(batch_size, num_features, seq_length)`（若视为图像，`seq_length`为高度/宽度）。  
- 输出：维度与输入相同。  
**推荐系统场景**：  
- 较少用于推荐系统，但可用于用户-物品交互特征的局部归一化（如图像类商品推荐）。  
**PyTorch实现**：  
```python
in_norm = nn.InstanceNorm1d(num_features=64)  # 对64个特征归一化
input = torch.randn(32, 64, 10)              # (batch_size=32, num_features=64, seq_length=10)
output = in_norm(input.permute(0, 2, 1))      # 调整维度为 (32, 10, 64)
output = output.permute(0, 2, 1)             # 恢复原维度
```

### **4. Group Normalization (GN)**
**原理**：将通道分组后归一化，平衡BN和LN的优缺点。  
**公式**：  
 <img width="800" height="195" alt="截屏2026-01-27 15 05 35" src="https://github.com/user-attachments/assets/364d3d31-65db-4165-b258-12a5c987fbb6" />

**输入输出维度**：  
- 输入：`(batch_size, num_features, seq_length)`。  
- 输出：维度与输入相同。  
**推荐系统场景**：  
- 适用于小批次或动态图任务（如目标检测），在推荐系统中可替代BN处理变长序列。  
**PyTorch实现**：  
```python
gn = nn.GroupNorm(num_groups=8, num_channels=64)  # 将64通道分为8组
input = torch.randn(32, 64, 10)                  # (batch_size=32, num_features=64, seq_length=10)
output = gn(input)  # 直接应用
```

### **推荐系统中的选择建议**
1. **Transformer模型**：优先用`LN`（如BERT4Rec），因序列长度不固定。  
2. **CNN特征提取**：若批次稳定，可用`BN`；否则用`GN`。  
3. **小批次训练**：`LN`或`GN`更稳健。  
4. **风格迁移类推荐**：如图像商品推荐，可尝试`IN`。  

### **关键区别总结**
| 归一化层 | 归一化维度       | 适用场景                     | 推荐系统典型用途           |
|----------|------------------|------------------------------|----------------------------|
| BN       | 批次内同通道     | 固定批次大小的CNN            | 用户行为序列的CNN特征提取   |
| LN       | 单样本所有特征   | RNN/Transformer/变长序列    | 用户行为序列建模           |
| IN       | 单样本单通道     | 风格迁移/生成模型            | 图像类商品推荐的特征处理   |
| GN       | 分组通道         | 小批次或动态图任务           | 替代BN处理变长序列         |
