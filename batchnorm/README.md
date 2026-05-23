# BatchNorm

这目录里的 `ref` 大致可以分成三类：

- `ref1.cu`：三阶段实现，先算 `mean`，再算 `var`，最后做 normalize。直观，但要额外开临时 buffer，而且输入会读 3 遍。
- `ref3.cu`：把 `sum` 和 `sum_sq` 合到一轮统计里，方向是对的，但最后只有 `thread 0` 串行写整个 channel，GPU 并行度太差。
- `ref5.cu` / `ref6.cu`：开始往 tiled / vectorize / benchmark 方向走，更像性能版，不适合第一次手撕。

## 推荐的面试手撕版本

直接使用 [solution_interview.cu](/Users/gary/interview/hand-written/batchnorm/solution_interview.cu)。

这版只保留 3 个关键点：

1. 一个 block 处理一个 channel
2. 第一轮遍历同时累计 `sum` 和 `sum_sq`
3. 第二轮遍历做 normalize，并行写回输出

核心公式也固定：

```cpp
mean = E[x]
var = E[x^2] - E[x]^2
y = gamma * (x - mean) / sqrt(var + eps) + beta
```

## 为什么这版适合面试

- 比三阶段更像样：只需要一个 kernel，不用额外申请 `mean/var` buffer
- 比优化版更稳：没有 atomic、没有 tile、没有向量化，现场不容易写乱
- 逻辑完整：既包含块内归约，也包含 `gamma/beta` 仿射变换
- 方便应对追问：可以自然展开到 `rsqrtf`、`fmaxf`、数值稳定性、训练态和推理态差异

## 记忆口诀

- 一个 block 管一个 channel
- 先做 `sum + sum_sq`
- 再算 `mean + inv_std`
- 最后并行写 `output`

## 手写时建议的讲法

可以直接对面试官这样说：

1. 输入按 `N x C` 展平存，batchnorm 要按 channel 统计
2. 我先让 `blockIdx.x = channel`
3. 块内线程按 `n += blockDim.x` 扫这个 channel 的所有 batch 元素
4. 第一遍顺手把 `sum` 和 `sum_sq` 都算出来，避免把输入读 3 遍
5. 用块内归约得到 `mean` 和 `var`
6. 第二遍再把 `(x - mean) * inv_std * gamma + beta` 写回

## 这版的定位

这版是“面试正确版 + 有一点优化意识”，不是最终工程最优版。

- 时间复杂度：`O(N * C)`
- 额外空间：`O(1)`，只用了少量 shared memory
- 优点：代码短、好记、公式完整、比最 naive 的三 kernel 版更进一步
- 缺点：每个 channel 都要读输入两遍；而且一个 block 只负责一个 channel，适合讲清楚，不一定是最快实现

## 如果面试官继续追问优化

你可以顺着往下答：

- 先把两次统计合并成一次：`sum + sum_sq`
- 再考虑 warp shuffle / block reduce，减少 shared memory 开销
- 如果 `N` 很大，可以做多 block 统计 + 二次归约
- 如果 `C` 连续、布局友好，可以继续做向量化加载
- 真正工程里还要区分 training 和 inference，对 running mean / running var 单独处理
