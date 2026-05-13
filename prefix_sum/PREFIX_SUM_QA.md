# Prefix Sum 面试追问

## Q1：这版做的是 inclusive scan 还是 exclusive scan？

默认是 inclusive scan。

也就是输出满足：

```text
out[i] = input[0] + input[1] + ... + input[i]
```

如果要 exclusive scan，只要把块内结果从“当前值 + 前缀”改成“前缀本身”，再把起始值处理好就行。

## Q2：Kogge-Stone 和 Blelloch 的核心区别是什么？

Kogge-Stone 是“逐轮传播前缀信息”，Blelloch 是“先建和树，再把和树改造成前缀树”。

更具体一点：

- Kogge-Stone：第 1 轮看前 `1` 个，第 2 轮看前 `2` 个，第 3 轮看前 `4` 个……
- Blelloch：先 upsweep 求总和，再 downsweep 求前缀和

## Q3：两者在 inclusive / exclusive 上有什么区别？

Kogge-Stone 更容易直接写成 inclusive scan。

Blelloch 的标准 downsweep 模板天然先得到 exclusive scan，所以如果题目要 inclusive scan，通常最后还要把当前位置原值再加回去。

也就是说：

- Kogge-Stone：更适合直接手写 inclusive
- Blelloch：更适合解释标准并行 scan 模板

## Q4：为什么推荐“2 个元素/线程，256 线程/块”？

因为这样每个 block 正好处理 512 个元素，比较容易记：

- 线程数不算太大，方便写块内同步
- 每个线程处理 2 个元素，吞吐比一线程一元素更好
- 512 是 2 的幂，适合 Blelloch scan 的树形结构

如果你走 Kogge-Stone 版，常见配置也可以直接用 `256` 线程、一线程一个元素，代码会更短。

## Q5：为什么块内用 Blelloch 或 Kogge-Stone，而不是直接顺序扫？

顺序扫当然最简单，但太 naive。

这两种并行 scan 的好处是：

- 块内是 `O(log n)` 轮同步
- 结构经典，面试容易讲
- 适合 GPU 的并行归约/扫描思维

其中：

- Kogge-Stone：代码更短，但总工作量更大
- Blelloch：代码更绕，但总工作量更优

## Q6：为什么还要扫 `block_sums`？

因为每个 block 只能得到自己的局部前缀和。

如果不处理 `block_sums`，那第二个 block 的结果只会从 0 开始，不知道前面所有 block 的总和是多少，所以必须把前面 block 的累积和加回来。

## Q7：为什么说这题可以递归？

因为 `block_sums` 也是一个数组。

如果它的长度还很大，就继续：

1. 对 `block_sums` 做 scan
2. 再把结果加回原数组

这就是“scan 的 scan”。

## Q8：为什么 `add_offsets` 里用 `scanned_block_sums[blockIdx.x - 1]`？

因为当前 block 需要加的是“前面所有 block 的和”。

`scanned_block_sums[b]` 是前 `b+1` 个 block 的总前缀，所以当前 block `b` 应该拿 `b-1` 那个位置的值。

## Q9：为什么要先做 upsweep，再做 downsweep？

这是 Blelloch scan 的标准套路。

- upsweep：先把整棵树的总和建起来
- downsweep：把总和分发成前缀和

这样能在块内用并行方式完成 scan。

## Q10：为什么 Kogge-Stone 不需要 downsweep？

因为 Kogge-Stone 不是“先建树再回填”的思路。

它每一轮都直接把前缀信息传播到当前位置，所以最后自然就得到 scan 结果了，不需要再来一轮树的回填。

## Q11：为什么最后还要把当前值 `+ input[i]`？

因为 downsweep 得到的是 exclusive scan。

而我们这里想要的是 inclusive scan，所以要把当前位置原值加回去。

如果你写的是 Kogge-Stone inclusive 版，这一步通常不需要。

## Q12：如果面试官要求 exclusive scan，怎么改？

很简单：

- Blelloch：保留 downsweep 后的 `temp[pos]`，不再加 `input[i]`
- Kogge-Stone：可以额外开一个位置右移，或者先保留旧值再构造 exclusive 结果

也就是：

```cpp
output[i] = temp[i];
```

## Q13：Kogge-Stone 和 Blelloch 的复杂度差异是什么？

如果看并行轮数，两者都是 `O(log n)` 级别。

但如果看总工作量：

- Blelloch：`O(n)`
- Kogge-Stone：`O(n log n)`

所以面试里常见说法是：

- Kogge-Stone 更好写
- Blelloch 更省工作量

## Q14：这版的时间复杂度是多少？

理论上是 `O(N)`。

虽然块内有 `log` 轮同步，但每个元素总体只参与常数次工作，所以总工作量仍然是线性的。

如果你严格讨论“块内 Kogge-Stone 的总工作量”，那它会比 Blelloch 更大。

## Q15：为什么不是单 block 扫完整个数组？

因为数组可能很大。

单 block 会把并行度压得太低，而且 shared memory 也放不下太多元素。分块扫描再合并块和，是更通用的做法。

## Q16：为什么不是每个线程直接做前缀累加？

那会变成串行思路。

prefix sum 的关键就是把“前面所有元素的累积”转成树形并行计算，而不是让一个线程从头加到尾。

## Q17：这版最大的性能瓶颈是什么？

主要是多次 kernel launch 和递归带来的开销。

如果是工程最优，通常会进一步做：

- 更深的块内融合
- warp-level scan
- decoupled look-back
- 或直接用 CUB

如果你写的是 Kogge-Stone，块内重复加法更多，这也是它比 Blelloch 更吃亏的地方。

## Q18：为什么不用 CUB？

如果是工程代码，CUB 很好。

但面试手撕通常想看你是否理解：

- 块内 scan
- 块和归约
- 跨块 offset 回填

所以这里更适合手写一个可解释的版本，而不是直接搬库。

## Q19：如果 `N` 不是 block 大小或 512 的倍数怎么办？

在 load 时对越界元素补 `0`。

这样不会影响前缀和的正确性，最后写回时再做边界判断即可。

## Q20：如果输入和输出是同一个指针，可以做吗？

通常可以。

因为每个 block 会先把数据读进 shared memory，再写回结果；跨 block 之间处理的是不同区间，逻辑上是安全的。

## Q21：如果要支持别的运算，不是加法，能不能做？

可以，但前提是这个运算满足结合律。

scan 本质上依赖“前缀可合并”，所以像加法、乘法、max 这类结合运算都可以；如果不满足结合律，就不能直接这么做。

## Q22：如果面试官问你这题和 reduce 有什么关系？

scan 可以看成“带保留中间结果的 reduce”。

- reduce 只要最终总和
- scan 要每个位置的前缀结果

所以 scan 比 reduce 更强一点，但底层的树形归约思想是同一类。
