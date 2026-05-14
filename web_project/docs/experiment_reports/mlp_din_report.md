# Deep MLP + DIN 实验报告

## 实验目的

在精排模型中引入用户行为序列特征，通过 DIN (Deep Interest Network) 注意力机制对用户历史行为进行自适应加权，捕捉用户在不同候选物品下的动态兴趣，以替代 DeepFM 的静态特征交叉范式。

- 基线模型：DeepFM（9 特征，3 epoch），AUC=0.8369 / gAUC=0.8444
- 实验模型：Deep MLP + DIN（16 静态特征 + hist_movie_id 序列，10 epoch，early stopping）

---

## 模型架构

```
输入层 (17 特征)
  │
  ├── 16 静态特征 ──→ Embedding ──→ Concat/Flatten ──→ DNN([128,64,32]) ──→ MLP logit
  │   (user_id, gender, age, occupation, zip_code, activity_bucket,
  │    movie_id, genres, isAdult, startYear, genre_count,
  │    popularity_bucket, quality_bucket, runtime_bucket,
  │    movie_age_bucket, director_bucket)
  │
  ├── Linear 一阶项 ──→ 16×Embedding(1) ──→ Add ──→ Linear logit
  │
  └── hist_movie_id ──→ Embedding (与 movie_id 共享词表) ──→ DIN Attention ──→ Dense(64) ──→ DIN logit
        (序列, 最大长度 10)        ↑
                            query = movie_id (candidate item)
                            keys  = hist_movie_id (user behavior sequence)
                            [q; k; q-k; q*k] → FFN([80,40], ReLU) → Dense(1) → softmax → weighted sum
                                                                        │
                Add(MLP + Linear + DIN) → Sigmoid ←─────────────────────┘
```

### DIN 注意力机制

`DinAttentionLayer` 对用户行为序列中的每个历史物品计算与候选物品的关联权重：

1. 将候选物品 `movie_id` 与历史序列 `hist_movie_id` 映射到同一 embedding 空间（共享词表）
2. 对序列中每个位置，构造交互特征 `[q, k, q-k, q*k]`（共 4×D 维）
3. 通过两层 FFN ([80, 40], ReLU) → Dense(1) 得到未归一化的注意力分数
4. padding 位置（0）做 mask 处理（score = -1e9）
5. softmax 归一化后对 keys 加权求和

这使模型能根据**候选物品的不同**，对用户历史行为赋予不同的权重，实现动态兴趣建模。

---

## 新增特征

| 特征 | 类型 | 说明 |
|------|------|------|
| `hist_movie_id` | 序列 (max_len=10) | 用户按时间序此前交互过的 movie_id，左 padding，与当前 target movie_id 共享 embedding |

其余 16 个静态特征与 DeepFM 特征增强实验保持一致。

### 序列构建方式

- ratings 按 user_id 分组，每组按 timestamp 排序
- 对每条记录，取该用户此前看过的所有 encoded movie_id（**排除当前目标电影**，防止 label leakage）
- 左 padding 到 10（padding_value=0）
- 负样本继承对应正样本的 hist_movie_id（同一用户同时刻的历史相同）

---

## 实验设置

| 配置项 | DeepFM (基线) | MLP+DIN |
|--------|-------------|---------|
| 特征数 | 9 | 16 静态 + 1 序列 |
| 嵌入维度 | 16 | 16 |
| DNN 结构 | [128, 64, 32] | [128, 64, 32] |
| 注意力结构 | — | FFN([80, 40]), ReLU |
| Batch size | 128 | 512 |
| Epochs | 3 | 10 (early stopping, patience=2) |
| 优化器 | Adam lr=0.001 | Adam lr=0.001 |
| 序列长度 | — | 10 |
| 训练设备 | 本地 CPU | AutoDL (RTX 4090 D) |

数据处理保持一致：时间序 80/20 划分，1:1 困难负样本 + 2:1 随机负样本。

---

## 实验结果

### 对比汇总

| 指标 | DCN (3 epoch) | DeepFM (3 epoch) | MLP+DIN |
|------|:----------:|:--------------:|:------:|
| AUC | 0.8312 | 0.8369 | **0.8703** |
| gAUC | 0.8409 | 0.8444 | **0.8722** |

MLP+DIN 在 AUC 和 gAUC 上均显著优于基线：
- AUC 提升 **+0.0334**（+4.0%）vs DeepFM
- gAUC 提升 **+0.0278**（+3.3%）vs DeepFM

### 训练过程

```
Epoch    train_auc    val_auc     val_loss
────────────────────────────────────────────
  1      0.8647       0.8600      0.4259
  2      0.8941       0.8652      0.4162
  3      0.9062       0.8673      0.4167
  4      0.9137       0.8699      0.4114
  5      0.9191       0.8700      0.4170  ← 最佳验证 AUC（early stopping 恢复到此权重）
  6      0.9230       0.8692      0.4187
  7      0.9264       0.8691      0.4187  ← early stopping 触发
```

- **收敛速度快**：epoch 5 即达到峰值，无需 10 epoch
- **过拟合控制好**：val_auc 在峰值后仅轻微下降（0.8700→0.8691），train-val gap 约 0.05
- **训练总耗时**：约 2 分钟（RTX 4090 D，~188 万训练样本，batch_size=512）

---

## 分析

### 1. 为什么 MLP+DIN 优于 DeepFM？

**DeepFM** 的 FM 组件做**静态**的 pairwise 特征交叉，无论候选物品是什么，用户 ID 与物品 ID 等特征的交叉权重是固定的。这意味着 DeepFM 学到的用户兴趣是"全局平均"的。

**DIN** 的注意力机制让用户行为序列的权重**随候选物品变化**：
- 候选是《变形金刚》→ 历史中动作片、科幻片的权重提高
- 候选是《真爱至上》→ 历史中爱情片、喜剧片的权重提高
- 候选是《教父》→ 历史中犯罪片、经典片的权重提高

这种**动态兴趣建模**是精排 AUC 从 0.84 提升到 0.87 的核心原因。

### 2. 训练效率

- MLP+DIN 仅 0.35M 参数（DeepFM 约 0.30M），参数量接近
- 5 epoch 收敛，比 DeepFM 的 3 epoch 稍多，但得益于 GPU 并行，总耗时仅约 2 分钟
- Early stopping 有效防止过拟合，最终权重为 epoch 5 而非 epoch 7

### 3. 共享 Embedding 的效果

`hist_movie_id` 与 `movie_id` 共享同一张 embedding 表，使 attention 在统一语义空间中计算。这不仅减少了参数量，还让梯度同时更新静态特征和序列特征的表示，促进相互学习。

---

## 改进建议

1. **增加更多序列特征**：当前仅有 `hist_movie_id`，可加入 `hist_genres`、`hist_startYear` 等，或使用多序列拼接的 multi-head attention
2. **增大序列长度**：当前 max_len=10，部分活跃用户历史超过 10，可尝试 20-30
3. **DIEN 替代 DIN**：在 DIN 基础上加入 GRU 建模行为序列的演化趋势（兴趣进化）
4. **特征交叉补充**：可在 DIN 输出后加入一个浅层 FM，补充静态特征的二阶交叉（当前纯 MLP 缺乏显式特征交叉）
5. **在线验证**：建议将 MLP+DIN 模型部署上线，观察线上 CTR/CVR 指标

---

## 结论

Deep MLP + DIN 架构在 MovieLens-1M 数据集上取得了显著效果提升：

- **AUC 从 0.8369 提升至 0.8703（+0.0334）**
- **gAUC 从 0.8444 提升至 0.8722（+0.0278）**

DIN 注意力机制通过动态兴趣建模，有效弥补了 DeepFM 静态特征交叉的局限。该架构简洁高效（0.35M 参数，5 epoch 收敛），适合作为精排模型的升级方案。
