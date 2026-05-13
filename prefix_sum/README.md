# Prefix Sum

这目录里的实现大致都在走同一条线：

- 块内先做 scan
- 每块记录一个 `block_sum`
- 再把 `block_sum` 做一次 scan
- 最后把前面所有块的偏移加回去

其中：

- `ref1.cu`：经典 Blelloch 方向，结构直观，但写法偏硬
- `ref2.cu`：递归 scan 的思路是对的，代码更接近“能跑的完整答案”
- `ref4.cu`：Kogge-Stone 路线，块内逻辑更短，比较适合第一次手撕
- `ref3.cu` / `ref5.cu` / `ref6.cu`：都在做不同程度的块内/递归/工程化变体
- `solution.cu`：目录里现有的整理版，保留作参考

## 推荐的面试手撕版本

我建议准备两版：

- [solution_interview.cu](/Users/gary/interview/hand-written/prefix_sum/solution_interview.cu)：Blelloch 版
- [solution_interview_kogge_stone.cu](/Users/gary/interview/hand-written/prefix_sum/solution_interview_kogge_stone.cu)：Kogge-Stone 版

如果面试时间很紧、你想先写一版最顺手的，优先用 Kogge-Stone。
如果面试官更在意“标准并行 scan 模板”和工作量复杂度，再讲 Blelloch。

## Blelloch 版的特点

Blelloch 版只保留 3 个关键点：

1. 每个 block 处理 `2 * 256` 个元素
2. 块内用 Blelloch scan 算出局部前缀和，并顺手记录块总和
3. 递归扫描 `block_sums`，再把偏移量加回原数组

核心流程很固定：

```text
block scan -> scan block sums -> add offsets
```

## Kogge-Stone 版的特点

Kogge-Stone 版也保留同样的跨 block 主线：

```text
block scan -> scan block sums -> add offsets
```

但块内 scan 的实现更直接：

1. 第 1 轮看前 `1` 个元素
2. 第 2 轮看前 `2` 个元素
3. 第 3 轮看前 `4` 个元素
4. 一直扩到 `8/16/32/...`

它天然更像“逐轮传播前缀信息”。

## 为什么这两版都适合面试

- 主线固定：都是“扫描数组、扫描块和、加偏移”
- 不太 naive：都处理了跨 block 偏移，不是只做局部 scan
- 好回答追问：都能自然讲到递归、inclusive/exclusive、边界处理、复杂度

区别主要在块内 scan：

- Kogge-Stone：代码更短，更适合第一次手撕
- Blelloch：结构更标准，总工作量更优

## 记忆口诀

- Kogge-Stone：每轮看前 `1/2/4/8/...` 个
- Blelloch：先 upsweep，再 downsweep
- 跨 block：先 scan 块和，再加偏移

## 这版的定位

这两版都是“面试正确版 + 轻量优化意识”，不是最终工程最优版。

- 时间复杂度：`O(N)`
- 额外空间：`O(N / 512)` 级别的块和数组，递归层数很少
- 优点：结构清楚、可扩展、比纯朴素版本更像 GPU 写法
- 缺点：递归和多次 kernel launch 仍然有开销；如果追求极致性能，通常会走更复杂的 look-back / CUB 路线

块内 scan 的细节差异是：

- Blelloch：总工作量 `O(n)`，但代码更绕，而且天然先得到 exclusive scan
- Kogge-Stone：代码更直观，天然更容易直接得到 inclusive scan，但块内总工作量是 `O(n log n)`

## 面试时建议怎么讲

你可以直接说：

1. prefix sum 先做块内扫描
2. 每个块最后一个元素就是这个块的总和
3. 把所有块总和再做一次扫描，得到每块该加的偏移
4. 把偏移加回每个块的元素

如果面试官追问为什么要递归，你就回答：因为“块和数组”本身也可能很长，不能假设它一定能一次放进一个 block 里扫完。

如果面试官追问“你为什么选 Kogge-Stone / Blelloch”，你可以这样答：

- 我先用 Kogge-Stone 写一版最稳的手撕答案，因为块内逻辑更短
- 如果继续追问复杂度和标准模板，我再补 Blelloch，说清楚它的 upsweep / downsweep 和 exclusive scan
