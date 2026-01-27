在 PyTorch 中，`nn.Embedding` 是一个用于存储**固定大小的嵌入矩阵（Embedding Matrix）**的层，常用于自然语言处理（NLP）和推荐系统等任务。它将离散的整数索引（如单词、ID）映射为密集的向量表示（即嵌入向量）。以下是详细说明和示例：

---

## **1. `nn.Embedding` 的核心功能**
- **作用**：将整数索引（如单词 ID）转换为对应的密集向量（嵌入向量）。
- **输入**：形状为 `(batch_size, sequence_length)` 的整数张量（每个元素是索引）。
- **输出**：形状为 `(batch_size, sequence_length, embedding_dim)` 的浮点张量（嵌入向量）。
- **可学习参数**：嵌入矩阵的权重（`weight`），可通过反向传播更新。

---

## **2. 主要参数**
```python
torch.nn.Embedding(
    num_embeddings,   # 词汇表大小（即最大索引 + 1）
    embedding_dim,    # 嵌入向量的维度
    padding_idx=None, # 指定填充索引（该索引的嵌入向量会被强制设为0，且不参与梯度更新）
    max_norm=None,    # 嵌入向量的最大范数（可选，用于归一化）
    norm_type=2.0,    # 范数类型（默认L2）
    scale_grad_by_freq=False, # 是否按词频缩放梯度
    sparse=False      # 是否使用稀疏梯度（加速大规模嵌入）
)
```

---

## **3. 详细用法**
### **(1) 基本用法**
```python
import torch
import torch.nn as nn

# 定义嵌入层：词汇表大小=10，嵌入维度=3
embedding = nn.Embedding(num_embeddings=10, embedding_dim=3)

# 输入：2个样本，每个样本是长度为4的索引序列
input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])

# 获取嵌入向量
output = embedding(input)
print(output.shape)  # 输出: torch.Size([2, 4, 3])
```

### **(2) 使用 `padding_idx`**
```python
# 指定索引0为填充符（嵌入向量始终为0，且不更新）
embedding = nn.Embedding(num_embeddings=10, embedding_dim=3, padding_idx=0)

input = torch.LongTensor([[0, 2, 0], [1, 0, 3]])  # 0是填充符
output = embedding(input)
print(output[0, 0])  # 输出: tensor([0., 0., 0.], grad_fn=<SelectBackward0>)
```

### **(3) 手动初始化嵌入矩阵**
```python
embedding = nn.Embedding(num_embeddings=5, embedding_dim=2)
# 手动设置嵌入矩阵的权重（例如用预训练词向量）
embedding.weight.data = torch.randn(5, 2)  # 随机初始化

input = torch.LongTensor([1, 3, 4])
output = embedding(input)
print(output)
```

### **(4) 在模型中使用（如文本分类）**
```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, hidden_dim)
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        # 对序列取平均（简化操作）
        pooled = embedded.mean(dim=1)  # (batch_size, embed_dim)
        return self.fc(pooled)

model = TextClassifier(vocab_size=1000, embed_dim=50, hidden_dim=10)
input = torch.randint(0, 1000, (32, 10))  # 32个样本，每个样本10个词
output = model(input)
print(output.shape)  # 输出: torch.Size([32, 10])
```

---

## **4. 关键注意事项**
1. **索引范围**：输入索引必须在 `[0, num_embeddings-1]` 范围内，否则会报错。
2. **梯度更新**：嵌入矩阵的权重默认参与反向传播（除非 `sparse=True`）。
3. **稀疏梯度**：对于大规模词汇表（如 >10万），设置 `sparse=True` 可加速优化。
4. **与 `nn.Linear` 的区别**：
   - `Embedding` 的输入是离散索引，`Linear` 的输入是连续值。
   - `Embedding` 的权重形状为 `(num_embeddings, embedding_dim)`，而 `Linear` 的权重形状为 `(in_features, out_features)`。

---

## **5. 完整示例：词向量训练**
```python
import torch
import torch.nn as nn
import torch.optim as optim

# 模拟数据：4个单词的词汇表，嵌入维度=2
vocab_size = 4
embed_dim = 2
embedding = nn.Embedding(vocab_size, embed_dim)

# 模拟训练数据：输入是索引序列，目标是让相邻词的嵌入向量接近
data = torch.LongTensor([[0, 1], [1, 2], [2, 3]])  # 训练对
labels = torch.FloatTensor([[1.0], [1.0], [1.0]])   # 相似度标签（这里简化）

optimizer = optim.SGD(embedding.parameters(), lr=0.1)
criterion = nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    # 获取嵌入向量
    embeddings = embedding(data)  # (3, 2, 2)
    # 计算相邻词的嵌入向量相似度（这里用点积简化）
    sim = (embeddings[:, 0] * embeddings[:, 1]).sum(dim=1).unsqueeze(1)
    loss = criterion(sim, labels)
    loss.backward()
    optimizer.step()

# 查看训练后的嵌入向量
print(embedding.weight)
```

---

## **6. 常见问题**
### **(1) 如何加载预训练词向量（如 Word2Vec）？**
```python
import numpy as np

# 假设预训练词向量是一个 (vocab_size, embed_dim) 的numpy数组
pretrained_weights = np.load("word2vec.npy")
embedding = nn.Embedding.from_pretrained(
    torch.FloatTensor(pretrained_weights),
    freeze=False  # 是否冻结权重（不更新）
)
```

### **(2) 如何处理 OOV（未登录词）？**
- 方法1：扩大 `num_embeddings`，并为 OOV 分配一个专用索引。
- 方法2：使用 `<UNK>` 标记的嵌入向量表示未知词。

---

## **总结**
- `nn.Embedding` 是 PyTorch 中处理离散索引到连续向量的核心层。
- 关键参数：`num_embeddings`（词汇表大小）、`embedding_dim`（向量维度）、`padding_idx`（填充符）。
- 典型应用：NLP 中的词嵌入、推荐系统中的用户/物品 ID 嵌入。
- 可通过 `from_pretrained` 加载预训练权重，或手动初始化嵌入矩阵。

通过合理使用 `nn.Embedding`，可以高效地将离散特征转换为密集向量，为下游任务（如分类、聚类）提供输入。 🚀
