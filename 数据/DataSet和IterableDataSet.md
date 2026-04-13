markdown
Copy Code
# PyTorch 中 IterableDataset 与 Dataset 的对比指南

## 🧩 核心区别
| &zwnj;**特性**&zwnj;               | &zwnj;**Dataset (Map-style)**&zwnj;              | &zwnj;**IterableDataset (Stream-style)**&zwnj;       |
|------------------------|--------------------------------------|------------------------------------------|
| &zwnj;**数据访问方式**&zwnj;       | 索引随机访问 (`dataset[i]`)          | 顺序迭代访问 (`iter(dataset)`)           |
| &zwnj;**必须实现的方法**&zwnj;     | `__len__`, `__getitem__`             | `__iter__`                               |
| &zwnj;**内存占用**&zwnj;           | 通常需预加载数据                     | 可动态生成数据，内存效率更高             |
| &zwnj;**DataLoader 的 shuffle**&zwnj; | 支持（通过`shuffle=True`）           | &zwnj;**不支持**&zwnj;（需在数据源内部实现乱序）     |
| &zwnj;**分布式训练**&zwnj;         | 需用`DistributedSampler`分片         | 需在`__iter__`中根据`worker_id`手动分片  |

## 📌 使用场景
### Dataset 适用场景
- 静态数据集（如图像/表格数据）
- 需随机访问的预加载数据
- 中小规模数据集（内存可容纳）

### IterableDataset 适用场景
- 流式数据（如日志/传感器流）
- 超大规模数据（TB级）
- 实时生成数据（如在线增强）
- 无法全量加载的场景

## 💻 代码示例

### 1. Dataset 基础用法
```python
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data  # 预加载数据

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # 随机访问

# 使用示例
data = [torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([5, 6])]
dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

2. IterableDataset 基础用法
python
Copy Code
from torch.utils.data import IterableDataset, DataLoader

class StreamDataset(IterableDataset):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        for i in range(self.start, self.end):
            yield torch.tensor([i])  # 顺序生成数据

# 使用示例（注意shuffle无效）
dataset = StreamDataset(0, 10)
dataloader = DataLoader(dataset, batch_size=3)

3. IterableDataset 分布式训练
python
Copy Code
import torch.distributed as dist

class DistributedStreamDataset(IterableDataset):
    def __init__(self, start, end, num_workers=1, rank=0):
        self.start = start
        self.end = end
        self.num_workers = num_workers
        self.rank = rank  # 当前进程ID

    def __iter__(self):
        # 计算当前进程的数据范围
        worker_size = (self.end - self.start) // self.num_workers
        worker_start = self.start + self.rank * worker_size
        worker_end = worker_start + worker_size
        
        for i in range(worker_start, worker_end):
            yield torch.tensor([i])

# 分布式初始化（示例）
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# 每个进程创建独立数据集
dataset = DistributedStreamDataset(0, 100, num_workers=world_size, rank=rank)
dataloader = DataLoader(dataset, batch_size=10)

⚠️ 关键注意事项

访问方式限制‌

Dataset 必须实现 __getitem__
IterableDataset 禁止实现 __getitem__

分布式差异‌

Dataset → 自动分片 (DistributedSampler)
IterableDataset → 手动分片 (worker_id控制)

性能优化‌

IterableDataset 配合 num_workers>0 实现并行预加载
流式数据需在数据源内部实现乱序

选择原则‌：静态/随机访问 → Dataset；动态/流式数据 → IterableDataset

text
Copy Code
