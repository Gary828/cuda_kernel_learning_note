# BatchNorm 面试追问
welford算法伪代码提供一下


• 可以，最常用的是这个版本：

  count = 0
  mean = 0
  M2 = 0

  for x in data:
      count += 1
      delta = x - mean
      mean += delta / count
      delta2 = x - mean
      M2 += delta * delta2

  variance = M2 / count      # population variance
  # variance = M2 / (count-1)  # sample variance

  如果要并行合并两个部分结果：

  merge(a, b):
      delta = b.mean - a.mean
      count = a.count + b.count
      mean = a.mean + delta * b.count / count
      M2 = a.M2 + b.M2 + delta * delta * a.count * b.count / count
      return (count, mean, M2)

  如果你要，我可以再给你一版“BatchNorm/LayerNorm 面试里能直接背的 CUDA 伪代码”。

 
 
## Q1：BatchNorm 在这个实现里是按哪个维度做归一化？

按 channel。

输入按 `N x C` 展平后，对固定的 `c`，需要统计所有 batch 样本上的：

- `mean[c]`
- `var[c]`

所以这版让一个 block 负责一个 channel，是最容易讲清楚的映射。

## Q2：为什么推荐“一个 block 处理一个 channel”？

因为逻辑最顺。

BatchNorm 的核心就是“对某个 channel 做一维归约”，所以最自然的并行方式就是：

- `blockIdx.x -> channel`
- `threadIdx.x -> 扫 batch 维`

这样公式、索引和归约关系都很好讲。

## Q3：为什么不是三段式 `mean -> var -> normalize`？

三段式当然能写，而且最直观。

但它有两个问题：

- 需要额外分配 `mean/var` 全局 buffer
- 输入通常会被读 3 遍

面试里更好的答案是把统计阶段收敛成：

```cpp
sum += x;
sumsq += x * x;
```

这样可以直接通过 `E[x^2] - E[x]^2` 算方差。

## Q4：`var = E[x^2] - E[x]^2` 为什么成立？

因为：

```cpp
Var(X) = E[X^2] - (E[X])^2
```

所以只要同时拿到：

- `sum(x)`
- `sum(x^2)`

就能在一次统计后得到方差。

## Q5：为什么这版还是要读输入两遍？

因为第一遍只是在做统计，第二遍才知道最终的 `mean` 和 `var`，所以归一化写回必须等统计结束后再做。

因此最自然的结构就是：

1. 第一遍：统计 `sum` 和 `sum_sq`
2. 块内归约出 `mean` 和 `inv_std`
3. 第二遍：做 normalize

这已经比三遍读取更好，但还不是单遍输出。

## Q6：为什么要写 `fmaxf(var, 0.0f)`？

因为浮点误差可能让：

```cpp
E[x^2] - E[x]^2
```

出现一个非常小的负数，比如 `-1e-7`。

如果直接拿去 `rsqrtf(var + eps)`，就可能出 NaN。先 clamp 到 0，是很常见的数值保护。

## Q7：为什么用 `rsqrtf(var + eps)`，而不是 `1.0f / sqrtf(var + eps)`？

两个原因：

- 写法更符合 CUDA 常见优化习惯
- 一般会更快一些

面试里直接说“先算 `inv_std`，后面重复使用”就够了。

## Q8：`eps` 是干什么的？

防止分母接近 0。

当某个 channel 的数据几乎一样时，`var` 会非常小，甚至接近 0。这时：

```cpp
sqrt(var)
```

会不稳定，所以要加一个很小的 `eps`。

## Q9：`gamma` 和 `beta` 的作用是什么？

它们是每个 channel 的可学习仿射参数：

```cpp
y = gamma * x_hat + beta
```

BatchNorm 不是只做零均值、单位方差，还会再给网络一个可学习的缩放和平移自由度。

## Q10：为什么块内归约适合用 warp shuffle？

因为这是 block 内求和，属于很标准的 reduction。

用 warp shuffle 可以先做 warp 内求和，再把每个 warp 的结果写到 shared memory，最后再做一次 warp 级归约。这样代码不长，而且比纯 shared memory reduction 更像样。

## Q11：为什么不直接让 `thread 0` 扫完整个 channel？

那就太 naive 了。

虽然逻辑也对，但会有两个问题：

- 一个 channel 基本只用到一个线程，GPU 并行度太差
- 面试官通常会继续追问“那你为什么不用块内并行归约”

所以更稳的版本是：线程先各自累加，再做 block reduction。

## Q12：这版最大的性能瓶颈是什么？

主要还是内存访问。

因为每个 channel 至少要扫两遍输入：

- 一遍统计
- 一遍归一化

所以这题通常不是算术太重，而是读写流量比较大。

## Q13：如果 `N` 非常大，一个 block 处理一个 channel 还够吗？

不一定。

当 `N` 很大时，一个 block 扫完整个 channel 可能不够快。这时可以继续拆成：

1. 多个 block 共同统计同一个 channel 的 partial sum / partial sumsq
2. 再做一次全局归约得到最终 mean / var
3. 最后做 normalize

但这已经超过“面试手撕正确版”的复杂度了。

## Q14：如果面试官问 training 和 inference 的区别，怎么答？

training 时，当前 batch 会现算：

- batch mean
- batch var

同时更新 running mean / running var。

inference 时，一般不再用当前输入现算统计量，而是直接使用训练阶段累计好的 running mean / running var。

也就是说：

- training：有实时统计
- inference：直接查历史统计

## Q15：这版和 LayerNorm 的主要区别是什么？

归约维度不同。

- BatchNorm：对每个 channel，在 batch 维上做统计
- LayerNorm：对每个样本，在特征维上做统计

所以虽然公式很像，但线程映射通常会不同。

## Q16：如果输入布局不是 `N x C`，而是 `N x H x W x C` 怎么办？

本质不变，还是“按 channel 归约”，只是统计范围会扩成：

```cpp
N * H * W
```

也就是每个 channel 要跨所有样本和空间位置去算 mean / var。

## Q17：如果面试官问反向传播，需要会到什么程度？

至少要能讲清三件事：

- `dgamma = sum(dy * x_hat)`
- `dbeta = sum(dy)`
- `dx` 不能只看当前位置，它和同一个 channel 的整体统计量有关

不一定非要现场把完整 backward 推公式写完，但要知道 backward 也是一个“按 channel 的归约问题”。

## Q18：如果继续优化，这题下一步往哪走？

- 让统计和归约写得更 cache / warp friendly
- 对大 `N` 做多 block 分阶段归约
- 利用更连续的内存布局做向量化加载
- 训练态下把 forward / stats / running update 做更好的 kernel 组织

面试里最重要的是先说明：我现在先写的是“正确且容易扩展”的版本，不是假装一步写到最优。
