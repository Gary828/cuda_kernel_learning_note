# TopK Selection 面试追问

## Q1：为什么不能只在一个 block 里做 bitonic sort？

因为那只能排这个 block 内部的数据，拿不到全局 top-k。

如果 `N > blockDim.x`，不同 block 之间完全没有合并，结果最多只是“每个 block 的局部 top-k”，不是整个输入上的 top-k。

## Q2：为什么 bitonic 这里要补到 2 的幂？

经典 bitonic sorting network 是按 2 的幂长度构造的。

如果 `N` 不是 2 的幂，最简单的做法就是补到 `next_pow2(N)`，补进去的元素用 `-INF`，这样不会干扰最大的 `k` 个值。

## Q3：为什么 padding 要填 `-INF`？

因为我们要做 top-k 最大值选择。

补进去的元素必须保证永远不可能进入最终答案，所以要选一个比真实数据都小的哨兵值。对 `float` 来说，最直接就是 `-CUDART_INF_F`。

## Q4：`peer = idx ^ stride` 这句是什么意思？

这是 bitonic network 的固定配对方式。

`stride` 对应当前阶段的比较距离，`idx ^ stride` 会把 `idx` 的某一位翻转，从而找到这一轮需要比较的 partner。

## Q5：为什么要判断 `peer > idx`？

为了避免一对元素被处理两次。

`idx` 和 `peer` 会互相指向对方，如果两个线程都做 swap，结果会重复甚至写乱。所以通常只保留一边，比如 `peer > idx`。

## Q6：为什么推荐“升序排完，再从尾部倒着取 top-k”？

因为这是标准 bitonic 模板，最稳。

你当然也可以把比较方向改成最终降序，然后直接拷贝前 `k` 个。但现场手写时，最容易记错的是“当前 stage 到底是升序还是降序”。保留标准升序模板，再单独做一个 gather，逻辑更稳。

## Q7：这版的复杂度是多少？

- 时间复杂度：`O(N log^2 N)`
- 额外空间：`O(N)`

原因是 bitonic sort 本质上就是完整排序，不是 selection-only 算法。

## Q8：如果 `k` 很小，这版有什么问题？

会做很多没必要的工作。

例如 `N = 1e6, k = 10`，完整排序全量数据明显浪费。工程里更常见的思路是：

- radix select
- heap / quickselect
- block 内局部 top-k + 全局 merge

## Q9：重复值会不会出错？

不会。

这版是对数组位置做排序，不是“按值去重”。所以像 `[7, 7, 7, 7]` 这种输入，取 `k = 3` 时会正确输出 `[7, 7, 7]`。

## Q10：如果面试官要求同时返回 value 和 index，怎么改？

把 `float` 改成 `(value, index)` 对即可。

比较时先比 `value`，如果需要稳定 tie-break，再比 `index`。swap 时把 value 和 index 一起交换。

## Q11：如果面试官问为什么不用 shared memory，怎么回答？

可以直接说：这题我先写“全局正确版”，优先保证跨 block 正确性。

shared memory 更适合做 block 内局部排序。要把它扩展成全局 top-k，还需要多阶段 merge，代码复杂度会明显上升。面试里如果先求稳，直接写全局 bitonic 更容易一次写对。

## Q12：这版还有哪些可以优化的点？

- `copy_and_pad` 和排序前几轮可以考虑融合
- block 内小段排序可以放到 shared memory
- 如果只要 top-k，不一定要完整排序
- 可以减少 kernel launch 次数，做分阶段 merge

## Q13：如果输入里本身就有 `-INF` 怎么办？

也没问题。

padding 的 `-INF` 只是额外增加了一些最小值，最多和真实输入里的 `-INF` 混在一起，但它们都不可能进入“最大 top-k”，所以不影响结果。

## Q14：如果要输出的是未排序 top-k，能更快吗？

通常可以。

因为“selection”和“sort”不是一回事。只要 top-k 集合，不要求内部有序时，可以用更便宜的选择算法，避免完整排序的 `O(N log^2 N)` 开销。

## Q15：如果面试官追问这版最大的工程缺点是什么？

最大的缺点不是 kernel 逻辑本身，而是它做了全量排序。

这在“只要很小的 k”时性价比不高；另外 bitonic 需要很多轮 kernel launch，launch overhead 也比较重。所以这版更适合作为面试手撕正确版，而不是最终工程最优版。
