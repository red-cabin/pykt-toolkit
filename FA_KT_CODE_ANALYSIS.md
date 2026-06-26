# FA_KT 代码结构与设计思路分析

本文档分析 `pykt/models/fa_kt.py` 中几个核心概念的代码对应关系：

- 时间编码器
- 练习编码器
- 多尺度频率分解
- 具备频率识别功能的路由器
- 异质性专家混合模型

结论先放前面：`fa_kt.py` 的主体思路是把“练习/知识点交互序列”和“时间间隔序列”分别编码，再用门控融合；每条编码路径内部可以先做三频带分解增强，再进入 MoE Transformer 层，由路由器在 CNN、LSTM、Attention、Mamba 四类异质专家之间自适应选择。

## 1. 整体数据流

核心模型类是 `FA_KT`：

- 代码位置：`pykt/models/fa_kt.py:280`
- 前向入口：`FA_KT.forward()`，`pykt/models/fa_kt.py:399`

训练时，`train_model.py` 会把普通序列特征 `dcur` 和时间 gap 特征 `dgaps` 一起传入 FA_KT：

- `pykt/models/train_model.py:88`
- `pykt/models/train_model.py:200`

整体流程：

```text
dcur:
  qseqs / cseqs / rseqs
      |
      v
练习/知识点编码 base_emb()
      |
      v
self.model(q_embed_data, qa_embed_data)
      |
      v
d_output

dgaps:
  rgaps / sgaps / pcounts
      |
      v
timeGap()
      |
      v
self.model2(temb, qa_embed_data)
      |
      v
t_output

d_output + t_output
      |
      v
门控融合 w = sigmoid(c_weight(d_output) + t_weight(t_output))
      |
      v
final_output
      |
      v
MLP 输出预测概率
```

最终预测代码在：

- 时间 embedding：`pykt/models/fa_kt.py:431`
- 双路径编码：`pykt/models/fa_kt.py:434` 到 `pykt/models/fa_kt.py:435`
- 内容/时间门控融合：`pykt/models/fa_kt.py:443` 到 `pykt/models/fa_kt.py:444`
- 输出层预测：`pykt/models/fa_kt.py:449` 到 `pykt/models/fa_kt.py:451`

## 2. 时间编码器

### 对应代码

时间编码器主要对应 `timeGap` 类：

- 类定义：`pykt/models/fa_kt.py:706`
- 线性映射：`pykt/models/fa_kt.py:721`
- 前向编码：`pykt/models/fa_kt.py:723`
- FA_KT 中实例化：`pykt/models/fa_kt.py:367`
- FA_KT 中调用：`pykt/models/fa_kt.py:431`

时间 gap 数据来自 `DktForgetDataset`：

- gap 字段创建：`pykt/datasets/dkt_forget_dataloader.py:138`
- gap 特征计算：`pykt/datasets/dkt_forget_dataloader.py:203`
- `rgaps`：同一知识点/题目距离上次出现的时间间隔
- `sgaps`：当前交互距离上一条交互的时间间隔
- `pcounts`：当前知识点/题目历史练习次数

### 设计思路

`timeGap` 并不是直接使用连续时间戳，而是使用离散化后的三类行为时间特征：

```text
rgap   = repeated gap，同一 skill/question 距离上次出现的间隔
sgap   = sequence gap，当前交互距离上一交互的间隔
pcount = past count，该 skill/question 过去出现次数
```

在 `DktForgetDataset.calC()` 中，这些值会通过 `log2()` 做压缩：

- `pykt/datasets/dkt_forget_dataloader.py:199`
- `pykt/datasets/dkt_forget_dataloader.py:216`
- `pykt/datasets/dkt_forget_dataloader.py:223`
- `pykt/datasets/dkt_forget_dataloader.py:228`

这样做的意义是减少极端时间间隔的数值影响。例如 5 分钟和 10 分钟应有差异，但 10000 分钟和 10010 分钟的差异不应被模型过度放大。

`timeGap.forward()` 会把三类离散特征转成 one-hot，再拼接：

```text
one_hot(rgap) + one_hot(sgap) + one_hot(pcount)
```

然后通过一个无 bias 的线性层映射到 `d_model` 维：

```text
tg_emb = Linear(num_rgap + num_sgap + num_pcount, d_model)
```

因此，时间编码器的作用是把“遗忘、间隔、重复练习次数”转成和题目/知识点 embedding 同维度的时间表示。

## 3. 练习编码器

### 对应代码

练习编码器主要由 `FA_KT.base_emb()` 和若干 embedding 层组成：

- embedding 初始化：`pykt/models/fa_kt.py:315` 到 `pykt/models/fa_kt.py:329`
- `base_emb()`：`pykt/models/fa_kt.py:390`
- 前向中构造序列：`pykt/models/fa_kt.py:400` 到 `pykt/models/fa_kt.py:407`
- 前向中调用：`pykt/models/fa_kt.py:415`

关键 embedding：

```text
self.q_embed        # 知识点/概念 embedding
self.qa_embed       # 答题交互 embedding
self.difficult_param
self.q_embed_diff
self.qa_embed_diff
```

### 设计思路

FA_KT 中虽然变量名有 `q_data`，但实际传入的是 `cseqs`，也就是知识点/概念序列：

```python
q_data = torch.cat((c[:, 0:1], cshft), dim=1)
```

所以这里的“练习编码器”更准确地说，是“概念/练习交互编码器”。

`base_emb()` 输出两类表示：

```text
q_embed_data  = 当前概念/练习本身的表示
qa_embed_data = 当前概念/练习 + 回答结果的交互表示
```

如果 `separate_qa=True`：

```text
qa_id = q_id + n_question * response
```

也就是把“答对某题”和“答错某题”看成两个不同交互 token。

如果 `separate_qa=False`：

```text
qa_embed = answer_embed(response) + q_embed
```

也就是题目/概念表示和答题结果表示相加。

当 `n_pid > 0` 时，还会加入题目难度调制：

- `difficult_param(pid_data)`
- `q_embed_diff(q_data)`
- `q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data`

这相当于让同一个知识点在不同题目难度下拥有不同表现。

## 4. 双编码路径：练习路径与时间路径

### 对应代码

FA_KT 中有两个 `Architecture`：

- `self.model`：练习/概念路径，`pykt/models/fa_kt.py:341`
- `self.model2`：时间路径，`pykt/models/fa_kt.py:368`

前向调用：

- `d_output, d_stats = self.model(q_embed_data, qa_embed_data)`，`pykt/models/fa_kt.py:434`
- `t_output, t_stats = self.model2(temb, qa_embed_data)`，`pykt/models/fa_kt.py:435`

融合：

- `w = sigmoid(c_weight(d_output) + t_weight(t_output))`，`pykt/models/fa_kt.py:443`
- `final_output = w * d_output + (1 - w) * t_output`，`pykt/models/fa_kt.py:444`

### 设计思路

`self.model` 关注学生在知识点/练习序列上的学习状态演化。

`self.model2` 关注时间行为对学习状态的影响，例如：

- 很久没练同一知识点，可能遗忘
- 连续刷题，短期状态变化更密集
- 某知识点练习次数越多，掌握概率可能越高

最后用一个逐位置、逐维度的门控权重 `w` 进行融合：

```text
final = w * content_state + (1 - w) * time_state
```

这比简单相加更灵活，因为模型可以在不同学生、不同时间步、不同隐藏维度上决定更依赖内容路径还是时间路径。

## 5. 多尺度频率分解

### 对应代码

多尺度频率分解主要对应：

- `ThreeBandFrequencyLayer`：`pykt/models/fa_kt.py:560`
- 低通滤波器初始化：`pykt/models/fa_kt.py:630`
- 在 `Architecture` 中实例化：`pykt/models/fa_kt.py:493`
- 在 `Architecture.forward()` 中调用：`pykt/models/fa_kt.py:510` 到 `pykt/models/fa_kt.py:511`

注意：只有当 `emb_type` 包含 `"band"` 时才会启用：

```python
elif emb_type.find("band") != -1:
    self.freq_enhancer = ThreeBandFrequencyLayer(...)
```

### 设计思路

`ThreeBandFrequencyLayer` 用两个因果深度卷积构造两级低通滤波：

```text
input
  |
  |-- lpf_stage1 --> 较平滑表示
  |-- lpf_stage2 --> 更低频表示
```

然后拆成三个频带：

```text
high_freq = input - lpf1_out
mid_freq  = lpf1_out - lpf2_out
low_freq  = lpf2_out
```

直观理解：

- 高频：短期快速波动，例如临时答题状态、局部噪声、突然变化
- 中频：中等跨度内的学习变化
- 低频：长期稳定趋势，例如较慢变化的掌握状态

每个频带会先按标准差归一化，再乘以可学习参数 `gammas`：

```text
high_freq = gamma_h * normalized_high
mid_freq  = gamma_m * normalized_mid
low_freq  = gamma_l * normalized_low
```

之后用可学习权重 `band_weights` 做 softmax 融合：

```text
combined = w_h * high + w_m * mid + w_l * low
```

最后通过 dropout、残差连接和 LayerNorm：

```text
output = LayerNorm(Dropout(combined) + input)
```

这个模块的目的，是在进入 Transformer/MoE 前，把序列表示按时间变化尺度拆开，让模型更容易捕获短期波动与长期趋势。

## 6. 具备频率识别功能的路由器

### 对应代码

路由器在 `MoETransformerLayer` 中：

- 类定义：`pykt/models/fa_kt.py:60`
- gate 定义：`pykt/models/fa_kt.py:106`
- router logits：`pykt/models/fa_kt.py:251`
- routing probs：`pykt/models/fa_kt.py:252`
- 置信度计算：`pykt/models/fa_kt.py:124`
- 自适应专家数量：`pykt/models/fa_kt.py:137`
- 专家选择与加权融合：`pykt/models/fa_kt.py:162`

### 设计思路

路由器的直接输入是当前层的 `query`：

```text
router_logits = gate(query)
routing_probs = softmax(router_logits)
```

`gate` 是一个线性层：

```text
Linear(d_model, num_experts)
```

它会为每个时间步输出每个专家的选择概率。

这里需要特别注意：代码里没有单独写一个 `FrequencyRouter` 类，也没有把 `high_freq / mid_freq / low_freq` 三个频带直接输入 router。所谓“具备频率识别功能”在当前代码中的实现更像是间接实现：

```text
原始 embedding
  -> ThreeBandFrequencyLayer 频率增强
  -> MoETransformerLayer
  -> gate(query) 基于增强后的 query 选择专家
```

也就是说，如果 `emb_type` 包含 `"band"`，`query` 已经包含三频带融合后的信息，router 可以基于这种频率增强表示学习专家选择策略。但它不是显式频带分类器。

### 自适应专家数量

路由器不是固定只选 top-1 或 top-k，而是先根据 routing 分布的熵计算置信度：

```text
entropy = -sum(p * log(p))
confidence = 1 - normalized_entropy
```

直观解释：

- 如果路由概率很集中，说明模型很确定该用哪个专家，选择较少专家。
- 如果路由概率很分散，说明当前时间步模式不清晰，选择更多专家共同处理。

阈值默认是：

```text
[0.8, 0.6, 0.4]
```

对应逻辑：

- 高置信度：选 `min_experts`
- 中等置信度：选 2 个专家
- 较低置信度：选 3 个专家
- 很低置信度：选 `max_experts`

这部分对应 `determine_expert_count()`。

## 7. 异质性专家混合模型

### 对应代码

四类专家：

- `CnnExpert`：`pykt/models/fa_kt.py:11`
- `LstmExpert`：`pykt/models/fa_kt.py:22`
- `AttentionExpert`：`pykt/models/fa_kt.py:31`
- `MambaExpert`：`pykt/models/fa_kt.py:51`

专家集合：

- `self.expert_dict`：`pykt/models/fa_kt.py:98`

专家输出计算：

- attention 专家：`pykt/models/fa_kt.py:183`
- 非 attention 专家：`pykt/models/fa_kt.py:185`

top-k 专家选择与融合：

- `torch.topk()`：`pykt/models/fa_kt.py:191`
- 权重 mask 与归一化：`pykt/models/fa_kt.py:193` 到 `pykt/models/fa_kt.py:198`
- 取出被选专家输出：`pykt/models/fa_kt.py:203`
- 加权求和：`pykt/models/fa_kt.py:205`

### 四类专家的分工

| 专家 | 代码类 | 主要建模能力 |
| --- | --- | --- |
| CNN | `CnnExpert` | 局部连续模式、短窗口变化 |
| LSTM | `LstmExpert` | 顺序依赖、递推式状态演化 |
| Attention | `AttentionExpert` | 长距离依赖、历史交互检索 |
| Mamba | `MambaExpert` | 状态空间序列建模，偏长序列效率与动态记忆 |

### 设计思路

KT 序列中的行为模式并不单一：

- 有些预测依赖最近几次练习，CNN 更合适。
- 有些依赖顺序状态积累，LSTM 更合适。
- 有些需要回看很远的相关知识点，Attention 更合适。
- 有些需要高效处理长序列状态，Mamba 更合适。

因此 FA_KT 使用异质专家混合，而不是多个同构 FFN 专家。这样每个专家拥有不同归纳偏置，路由器可以按时间步选择最合适的组合。

## 8. 单专家消融与关闭 MoE

FA_KT 支持通过 `emb_type` 控制专家结构：

- 包含 `onlyattn`：只用 Attention 专家
- 包含 `onlylstm`：只用 LSTM 专家
- 包含 `onlymamba`：只用 Mamba 专家
- 包含 `onlycnn`：只用 CNN 专家
- 包含 `nomoe`：关闭 MoE，使用默认 Attention 专家

对应代码：

- 单专家类型解析：`pykt/models/fa_kt.py:333` 到 `pykt/models/fa_kt.py:340`
- 关闭 MoE：`pykt/models/fa_kt.py:347` 到 `pykt/models/fa_kt.py:348`
- 单专家初始化：`pykt/models/fa_kt.py:78` 到 `pykt/models/fa_kt.py:93`
- 非 MoE 默认专家：`pykt/models/fa_kt.py:110`

这套开关主要用于消融实验，比较不同专家或 MoE 机制本身的贡献。

## 9. 模块对应关系总表

| 论文/设计概念 | 主要代码位置 | 核心作用 |
| --- | --- | --- |
| 时间编码器 | `timeGap`，`pykt/models/fa_kt.py:706` | 将 `rgap/sgap/pcount` one-hot 拼接后映射到 `d_model` |
| 时间特征来源 | `DktForgetDataset.calC()`，`pykt/datasets/dkt_forget_dataloader.py:203` | 从 timestamps 计算重复间隔、序列间隔、历史练习次数 |
| 练习编码器 | `FA_KT.base_emb()`，`pykt/models/fa_kt.py:390` | 编码知识点/练习及其答题结果 |
| 双路径编码 | `self.model` 与 `self.model2`，`pykt/models/fa_kt.py:341`、`pykt/models/fa_kt.py:368` | 分别处理练习内容序列和时间 gap 序列 |
| 内容/时间融合 | `c_weight/t_weight`，`pykt/models/fa_kt.py:443` | 动态融合内容状态与时间状态 |
| 多尺度频率分解 | `ThreeBandFrequencyLayer`，`pykt/models/fa_kt.py:560` | 分解高频、中频、低频学习状态变化 |
| 低通滤波器 | `init_lowpass_filter()`，`pykt/models/fa_kt.py:630` | 用 sinc + window 初始化低通卷积核 |
| 频率增强入口 | `Architecture.forward()`，`pykt/models/fa_kt.py:510` | 当 `emb_type` 含 `band` 时启用 |
| 路由器 | `self.gate`，`pykt/models/fa_kt.py:106` | 根据 query 输出专家概率 |
| 自适应专家数 | `compute_confidence_metrics()`，`pykt/models/fa_kt.py:124` | 用路由熵估计置信度，决定选几个专家 |
| 异质专家池 | `expert_dict`，`pykt/models/fa_kt.py:98` | CNN/LSTM/Attention/Mamba 四类专家 |
| 专家融合 | `adaptive_expert_selection()`，`pykt/models/fa_kt.py:162` | top-k 专家输出加权求和 |

## 10. 一句话总结

`FA_KT` 的设计核心是：用练习编码器建模学生在知识点交互上的状态，用时间编码器建模遗忘和重复练习行为；在每条路径内部，通过多尺度频率分解增强短期/中期/长期变化特征，再交给异质 MoE 层按时间步动态选择 CNN、LSTM、Attention、Mamba 专家，最后用门控方式融合内容状态和时间状态完成知识追踪预测。

## 11. 重点拆解：如何把三个模块学会并拆出来用

下面重点讲三个最值得拆出来复用的模块：

- 时间编码器：把离散时间行为特征变成 dense embedding。
- 内容/时间双分支门控融合：把内容状态和时间状态动态合并。
- 三频带频率分解：把序列表示拆成高频、中频、低频变化，再融合增强。

建议阅读顺序：

1. 先看 `pykt/datasets/dkt_forget_dataloader.py:199` 到 `pykt/datasets/dkt_forget_dataloader.py:230`，理解时间特征是怎么从原始时间戳构造出来的。
2. 再看 `pykt/models/fa_kt.py:706` 到 `pykt/models/fa_kt.py:735`，理解 `timeGap` 如何把 `rgap/sgap/pcount` 转成 embedding。
3. 然后看 `pykt/models/fa_kt.py:367` 到 `pykt/models/fa_kt.py:381`，理解 FA_KT 如何定义时间分支和融合门控。
4. 接着看 `pykt/models/fa_kt.py:399` 到 `pykt/models/fa_kt.py:450`，沿着 forward 跑一遍完整数据流。
5. 最后看 `pykt/models/fa_kt.py:560` 到 `pykt/models/fa_kt.py:628`，理解三频带分解模块如何独立工作。

### 11.1 时间编码器怎么设计

时间编码器不是直接喂原始时间戳，而是先把时间行为变成三个离散特征。

原始交互序列大概是：

```text
skill:      s1, s2, s1, s3, s1
timestamp: 10, 20, 80, 90, 300
```

模型真正使用的是：

```text
rgap:   当前 skill 距离上一次出现过了多久
sgap:   当前交互距离上一条交互过了多久
pcount: 当前 skill 之前练过几次
```

对应代码在 `DktForgetDataset.calC()`：

```python
curRepeatedGap = self.log2((t - dlastskill[s]) / 1000 / 60) + 1
curLastGap = self.log2((t - pret) / 1000 / 60) + 1
past_counts.append(self.log2(dcount[s]))
```

这里有两个关键设计点。

第一，使用相对时间，而不是绝对时间戳。绝对时间戳本身通常没有意义，真正有意义的是“间隔多久没有练”“上一题到这一题隔了多久”。

第二，使用 `log2(t + 1)` 压缩数值。教育行为里的时间间隔通常长尾很严重：几分钟、几小时、几天都可能出现。如果直接用分钟数，极大值会压过其他信息。`log2` 可以保留顺序关系，同时减小极端值影响。

离散特征进入 `timeGap` 后，处理方式是：

```text
rgap id    -> one-hot
sgap id    -> one-hot
pcount id  -> one-hot
拼接       -> [B, L, num_rgap + num_sgap + num_pcount]
Linear     -> [B, L, d_model]
```

对应代码：

```python
rgap = self.rgap_eye[rgap]
sgap = self.sgap_eye[sgap]
pcount = self.pcount_eye[pcount]
tg = torch.cat(infs, -1)
tg_emb = self.time_emb(tg)
```

如果你想拆出来用，最小版本可以写成这样：

```python
class TimeGapEncoder(nn.Module):
    def __init__(self, num_rgap, num_sgap, num_pcount, d_model):
        super().__init__()
        self.register_buffer("rgap_eye", torch.eye(num_rgap))
        self.register_buffer("sgap_eye", torch.eye(num_sgap))
        self.register_buffer("pcount_eye", torch.eye(num_pcount))
        self.proj = nn.Linear(num_rgap + num_sgap + num_pcount, d_model, bias=False)

    def forward(self, rgap, sgap, pcount):
        x = torch.cat([
            self.rgap_eye[rgap],
            self.sgap_eye[sgap],
            self.pcount_eye[pcount],
        ], dim=-1)
        return self.proj(x)
```

输入输出形状：

```text
rgap/sgap/pcount: [batch_size, seq_len]
time_embedding:   [batch_size, seq_len, d_model]
```

学习时你要重点理解三个问题：

- 为什么不用绝对时间，而用间隔时间？
- 为什么要把时间间隔离散化并做 one-hot？
- 为什么最后映射到和内容 embedding 一样的 `d_model`？

答案是：这样时间特征才能像普通 token embedding 一样，直接进入后续 Transformer、LSTM、MoE 或其他序列模型。

### 11.2 内容/时间双分支门控融合怎么设计

FA_KT 没有把时间 embedding 简单加到内容 embedding 后只走一条网络，而是设计了两条分支：

```text
内容分支:
q_embed_data + qa_embed_data -> self.model -> d_output

时间分支:
temb + qa_embed_data         -> self.model2 -> t_output
```

对应代码：

```python
d_output, d_stats = self.model(q_embed_data, qa_embed_data)
t_output, t_stats = self.model2(temb, qa_embed_data)
```

这里的 `q_embed_data` 是知识点/练习内容表示，`qa_embed_data` 是带答题结果的交互表示，`temb` 是时间编码器输出。

为什么要双分支？

因为内容信息和时间信息表达的是两种不同因素：

- 内容分支回答“学生在这些知识点上表现如何”。
- 时间分支回答“遗忘、间隔、重复练习对当前状态有什么影响”。

如果直接相加：

```text
q_embed_data + temb
```

模型很早就把两种信息混在一起了，后续很难区分预测到底更应该依赖内容还是时间。

双分支的好处是先让两种因素各自充分建模：

```text
d_output = content_encoder(content)
t_output = time_encoder(time)
```

然后再融合。

FA_KT 的融合不是固定比例，而是逐位置、逐维度的门控融合：

```python
w = torch.sigmoid(self.c_weight(d_output) + self.t_weight(t_output))
final_output = w * d_output + (1 - w) * t_output
```

可以把 `w` 理解为“当前位置上内容分支的可信度/占比”：

```text
w 接近 1：更相信内容分支
w 接近 0：更相信时间分支
w 在中间：两者混合
```

而且 `w` 的形状是：

```text
[batch_size, seq_len, d_model]
```

这意味着它不是一个全局标量，而是每个学生、每个时间步、每个隐藏维度都有自己的融合比例。这是这个设计最灵活的地方。

拆出来复用时，最小门控融合模块可以写成：

```python
class DualBranchGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.content_gate = nn.Linear(d_model, d_model)
        self.time_gate = nn.Linear(d_model, d_model)

    def forward(self, content_state, time_state):
        w = torch.sigmoid(self.content_gate(content_state) + self.time_gate(time_state))
        fused = w * content_state + (1.0 - w) * time_state
        return fused, w
```

你可以把它用在任何两个同形状序列状态之间：

```text
content_state: [B, L, D]
time_state:    [B, L, D]
fused:         [B, L, D]
gate w:        [B, L, D]
```

学习这个模块时建议你做三个小实验：

1. 固定 `w=0.5`，比较简单平均融合的效果。
2. 改成 `w = sigmoid(Linear(concat(content, time)))`，比较 concat gate 和当前加法 gate。
3. 可视化 `w.mean(dim=-1)`，看不同时间步模型更依赖内容还是时间。

这能帮你真正理解门控融合不是装饰，而是在做动态信息选择。

### 11.3 三频带频率分解模块怎么设计

三频带频率分解对应 `ThreeBandFrequencyLayer`。

它的输入是一个序列表示：

```text
input_tensor: [batch_size, seq_len, hidden_size]
```

模块先把维度换成 Conv1d 需要的格式：

```python
x = input_tensor.permute(0, 2, 1)
```

变成：

```text
[batch_size, hidden_size, seq_len]
```

然后用两个深度卷积做低通滤波：

```python
lpf1_out = self.lpf_stage1(x)
lpf2_out = self.lpf_stage2(lpf1_out)
```

这里的卷积是 depthwise causal convolution：

```python
CausalConv1d(hidden_size, hidden_size, kernel_size, groups=hidden_size)
```

三个关键点：

- `causal`：只看当前和过去，不偷看未来，适合知识追踪这种时序预测。
- `groups=hidden_size`：每个隐藏维度单独滤波，不混合通道。
- `kernel_size`：控制平滑窗口大小，窗口越大，保留的变化越慢。

然后分解成三个频带：

```python
high_freq = input_tensor - lpf1_out
mid_freq = lpf1_out - lpf2_out
low_freq = lpf2_out
```

这个设计很像信号处理里的多尺度分解：

```text
原始信号 = 快速变化 + 中等变化 + 缓慢趋势
```

具体理解：

- `lpf1_out` 是第一次低通后的平滑信号。
- `input - lpf1_out` 剩下的是被低通滤掉的快速变化，所以是高频。
- `lpf2_out` 是更强低通后的慢变化，所以是低频。
- `lpf1_out - lpf2_out` 是介于两者之间的变化，所以是中频。

之后代码做了每个频带的标准差归一化：

```python
high_freq = high_freq / hf_std
mid_freq = mid_freq / mf_std
low_freq = low_freq / lf_std
```

这样做是为了避免某个频带因为数值尺度更大而天然压过其他频带。

然后每个频带有一个可学习强度参数：

```python
high_freq = self.gammas[0] * high_freq
mid_freq = self.gammas[1] * mid_freq
low_freq = self.gammas[2] * low_freq
```

最后用 softmax 权重融合：

```python
weights = F.softmax(self.band_weights, dim=0)
combined = weights[0] * high_freq + weights[1] * mid_freq + weights[2] * low_freq
output = self.layer_norm(self.out_dropout(combined) + input_tensor)
```

这里有两层可学习控制：

- `gammas`：控制每个频带内部的放大/缩小。
- `band_weights`：控制三个频带最终融合比例。

为什么要残差连接？

因为频率分解是增强模块，不应该强迫模型完全依赖分解结果。残差让模型保留原始表示：

```text
output = normalized(frequency_enhancement + original_input)
```

如果你想拆出来用，最小使用方式是：

```python
freq_layer = ThreeBandFrequencyLayer(
    dropout=0.1,
    hidden_size=d_model,
    kernel_sizes=[3, 15],
)

enhanced_x, (high, mid, low) = freq_layer(x)
```

输入输出：

```text
x:          [B, L, D]
enhanced_x: [B, L, D]
high/mid/low: each [B, L, D]
```

学习这个模块时建议你按下面顺序拆：

1. 先只实现 `CausalConv1d`，输入一个简单序列，看输出是否没有未来信息泄露。
2. 再实现一个一级低通：`smooth = lpf(x)`，观察 `x - smooth` 是否捕捉到快速变化。
3. 再加第二级低通，构造 `high/mid/low`。
4. 最后加入 `gammas`、`band_weights`、dropout、residual、LayerNorm。

一个容易踩的点：`kernel_size1` 和 `kernel_size2` 应该有明显尺度差异。例如 `[3, 15]` 比 `[3, 5]` 更容易形成短期/长期差异。如果两个 kernel 太接近，中频和低频的区分会变弱。

### 11.4 三个模块如何组合成自己的模型

如果你想把这三块拆出去，用到自己的知识追踪模型或其他时序模型里，可以按这个顺序搭：

```text
1. 内容 embedding
   item_id / concept_id / response -> content_emb

2. 时间 embedding
   rgap / sgap / pcount -> time_emb

3. 两条分支
   content_state = encoder(content_emb, interaction_emb)
   time_state    = encoder(time_emb, interaction_emb)

4. 可选频率增强
   content_emb = freq_layer(content_emb)
   time_emb    = freq_layer(time_emb)

5. 门控融合
   fused_state = gate(content_state, time_state)

6. 输出预测
   pred = sigmoid(MLP(concat(fused_state, content_emb + time_emb)))
```

最小可复用骨架：

```python
class ContentTimeModel(nn.Module):
    def __init__(self, num_items, num_rgap, num_sgap, num_pcount, d_model, encoder):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, d_model)
        self.answer_emb = nn.Embedding(2, d_model)
        self.time_encoder = TimeGapEncoder(num_rgap, num_sgap, num_pcount, d_model)
        self.freq = ThreeBandFrequencyLayer(0.1, d_model, [3, 15])
        self.content_encoder = encoder
        self.time_branch_encoder = copy.deepcopy(encoder)
        self.gate = DualBranchGate(d_model)
        self.out = nn.Linear(d_model * 2, 1)

    def forward(self, item, answer, rgap, sgap, pcount):
        item_emb = self.item_emb(item)
        interaction_emb = item_emb + self.answer_emb(answer)
        time_emb = self.time_encoder(rgap, sgap, pcount)

        item_enhanced, _ = self.freq(item_emb)
        time_enhanced, _ = self.freq(time_emb)

        content_state = self.content_encoder(item_enhanced, interaction_emb)
        time_state = self.time_branch_encoder(time_enhanced, interaction_emb)

        fused, gate = self.gate(content_state, time_state)
        logits = self.out(torch.cat([fused, item_enhanced + time_enhanced], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits), gate
```

这里的 `encoder` 可以先不用 MoE，直接用你熟悉的 LSTM 或 Transformer。等你把时间编码、频率分解、门控融合跑通后，再考虑接 MoE。

### 11.5 推荐学习路线

第一阶段：只学时间特征。

- 读 `DktForgetDataset.calC()`。
- 自己用一条手写序列算出 `rgap/sgap/pcount`。
- 把它们喂给 `timeGap`，确认输出形状是 `[B, L, D]`。

第二阶段：只学双分支融合。

- 用两个随机 tensor 模拟 `d_output` 和 `t_output`。
- 实现 `DualBranchGate`。
- 打印 `w.mean()`、`w.min()`、`w.max()`，观察门控范围。

第三阶段：只学三频带分解。

- 构造一个混合信号：慢趋势 + 快速震荡 + 噪声。
- 喂给 `ThreeBandFrequencyLayer`。
- 分别画出或打印 `high/mid/low` 的均值、标准差。

第四阶段：组合。

- 先用 `timeGap + DualBranchGate` 接一个简单 LSTM。
- 再加入 `ThreeBandFrequencyLayer`。
- 最后再回到 FA_KT 的完整 `Architecture` 和 MoE。

最重要的学习原则：不要一开始就读完整 FA_KT。它把时间编码、频率增强、MoE、Transformer、门控融合放在一起，信息密度很高。你应该先把每个模块当成独立积木跑通，再回来看它们如何组合。
