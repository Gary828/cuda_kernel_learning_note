# Kogge-Stone vs Blelloch

这份文档专门对比 prefix sum 里最常见的两条并行 scan 路线：

- Kogge-Stone
- Blelloch

建议把它当成面试速记材料来用。

---

## 1. 两者分别在做什么

### Kogge-Stone

Kogge-Stone 的思路是：

- 第 1 轮，看前 `1` 个元素
- 第 2 轮，看前 `2` 个元素
- 第 3 轮，看前 `4` 个元素
- 第 4 轮，看前 `8` 个元素

也就是每一轮都把“更远距离的前缀信息”继续往后传播。

它的特点是：

- 代码短
- 很容易直接写成 inclusive scan
- 总工作量偏大

### Blelloch

Blelloch 的思路是：

1. `upsweep`：先把数组构造成一棵求和树
2. `downsweep`：再把这棵求和树改造成前缀和树

它的特点是：

- 是经典标准模板
- 总工作量更优
- 天然先得到 exclusive scan
- 代码比 Kogge-Stone 更绕

---

## 2. 一个 8 元素的完整例子

假设输入是：

```text
[1, 2, 3, 4, 5, 6, 7, 8]
```

目标：

- inclusive scan：

```text
[1, 3, 6, 10, 15, 21, 28, 36]
```

- exclusive scan：

```text
[0, 1, 3, 6, 10, 15, 21, 28]
```

---

## 3. Kogge-Stone 手推

Kogge-Stone 一般直接做 inclusive scan。

### 初始状态

```text
[1, 2, 3, 4, 5, 6, 7, 8]
```

### 第 1 轮：offset = 1

每个位置都加上前面 `1` 个位置的值：

```text
[1, 1+2, 2+3, 3+4, 4+5, 5+6, 6+7, 7+8]
=
[1, 3, 5, 7, 9, 11, 13, 15]
```

注意这里并行实现里不能直接原地顺着改，必须先读旧值再同步。

### 第 2 轮：offset = 2

每个位置都加上前面 `2` 个位置的结果：

```text
[1, 3, 1+5, 3+7, 5+9, 7+11, 9+13, 11+15]
=
[1, 3, 6, 10, 14, 18, 22, 26]
```

### 第 3 轮：offset = 4

每个位置都加上前面 `4` 个位置的结果：

```text
[1, 3, 6, 10, 1+14, 3+18, 6+22, 10+26]
=
[1, 3, 6, 10, 15, 21, 28, 36]
```

这就已经是 inclusive scan。

### Kogge-Stone 的直觉

它不是先建树再回填，而是：

- 每一轮都直接让当前位置拿到更长前缀的信息
- 直到完整前缀被传播到所有位置

---

## 4. Blelloch 手推

Blelloch 分两步：

- upsweep
- downsweep

### 4.1 Upsweep

初始状态：

```text
[1, 2, 3, 4, 5, 6, 7, 8]
```

#### 第 1 轮：offset = 1

更新位置 `1, 3, 5, 7`

```text
temp[1] += temp[0]
temp[3] += temp[2]
temp[5] += temp[4]
temp[7] += temp[6]
```

结果：

```text
[1, 3, 3, 7, 5, 11, 7, 15]
```

#### 第 2 轮：offset = 2

更新位置 `3, 7`

```text
temp[3] += temp[1]
temp[7] += temp[5]
```

结果：

```text
[1, 3, 3, 10, 5, 11, 7, 26]
```

#### 第 3 轮：offset = 4

更新位置 `7`

```text
temp[7] += temp[3]
```

结果：

```text
[1, 3, 3, 10, 5, 11, 7, 36]
```

现在最右边 `36` 就是整段总和。

### 4.2 Downsweep

先把根节点置成 `0`：

```text
[1, 3, 3, 10, 5, 11, 7, 0]
```

这一步意味着：整个数组左边没有任何元素，所以最开始的前缀基准是 `0`。

#### 第 1 轮：offset = 4

处理位置 `7`

```text
val = temp[3] = 10
temp[3] = temp[7] = 0
temp[7] = temp[7] + val = 10
```

结果：

```text
[1, 3, 3, 0, 5, 11, 7, 10]
```

#### 第 2 轮：offset = 2

处理位置 `3, 7`

对 `idx = 3`：

```text
val = temp[1] = 3
temp[1] = temp[3] = 0
temp[3] = 0 + 3 = 3
```

对 `idx = 7`：

```text
val = temp[5] = 11
temp[5] = temp[7] = 10
temp[7] = 10 + 11 = 21
```

结果：

```text
[1, 0, 3, 3, 5, 10, 7, 21]
```

#### 第 3 轮：offset = 1

处理位置 `1, 3, 5, 7`

对 `idx = 1`：

```text
val = temp[0] = 1
temp[0] = temp[1] = 0
temp[1] = 0 + 1 = 1
```

对 `idx = 3`：

```text
val = temp[2] = 3
temp[2] = temp[3] = 3
temp[3] = 3 + 3 = 6
```

对 `idx = 5`：

```text
val = temp[4] = 5
temp[4] = temp[5] = 10
temp[5] = 10 + 5 = 15
```

对 `idx = 7`：

```text
val = temp[6] = 7
temp[6] = temp[7] = 21
temp[7] = 21 + 7 = 28
```

最终得到：

```text
[0, 1, 3, 6, 10, 15, 21, 28]
```

这就是 exclusive scan。

如果要 inclusive scan，再把原始输入加回去：

```text
[0, 1, 3, 6, 10, 15, 21, 28]
+
[1, 2, 3, 4, 5, 6, 7, 8]
=
[1, 3, 6, 10, 15, 21, 28, 36]
```

---

## 5. 两者的 inclusive / exclusive 差异

### Kogge-Stone

最自然的是直接写成 inclusive scan。

如果面试官要 exclusive：

- 可以右移一位
- 或者额外保留上一轮结果

### Blelloch

最自然的是先得到 exclusive scan。

如果面试官要 inclusive：

- 最后再加回原始输入

这是两者在面试里最容易被问到的区别。

---

## 6. 复杂度对比

假设块内处理 `n` 个元素。

### Kogge-Stone

- 并行轮数：`O(log n)`
- 总工作量：`O(n log n)`

原因是：

- 每一轮基本都有很多线程参与
- 总共做 `log n` 轮
- 所以重复计算更多

### Blelloch

- 并行轮数：`O(log n)`
- 总工作量：`O(n)`

原因是：

- upsweep 和 downsweep 各做 `log n` 轮
- 但每一轮活跃线程数在递减或递增
- 总体加法次数是线性的

### 结论

- 如果问“谁更好写”：Kogge-Stone
- 如果问“谁总工作量更优”：Blelloch

---

## 7. 实现层面的区别

### Kogge-Stone 的实现特征

代码通常长这样：

```cpp
for (int offset = 1; offset < blockDim.x; offset <<= 1) {
    float val = (tid >= offset) ? temp[tid - offset] : 0.0f;
    __syncthreads();
    temp[tid] += val;
    __syncthreads();
}
```

特点是：

- 每一轮都直接往当前位置累加
- 实现短
- 容易理解
- 但重复读写更多

### Blelloch 的实现特征

upsweep：

```cpp
for (int offset = 1; offset < n; offset <<= 1) {
    int idx = (tid + 1) * (offset << 1) - 1;
    if (idx < n) {
        temp[idx] += temp[idx - offset];
    }
    __syncthreads();
}
```

downsweep：

```cpp
for (int offset = n >> 1; offset > 0; offset >>= 1) {
    int idx = (tid + 1) * (offset << 1) - 1;
    if (idx < n) {
        float val = temp[idx - offset];
        temp[idx - offset] = temp[idx];
        temp[idx] += val;
    }
    __syncthreads();
}
```

特点是：

- 更像树结构算法
- 模板性很强
- 更适合解释标准并行 scan

---

## 8. 在面试里怎么选

### 如果你想先快速写对

先写 Kogge-Stone。

理由：

- 逻辑更直观
- 每一轮只是“看前 `1/2/4/8...` 个”
- 更不容易把 upsweep / downsweep 写混

### 如果面试官更看重“标准答案”

写 Blelloch。

理由：

- 是经典并行 scan 模板
- 更容易顺着讲到 exclusive / inclusive
- 更容易展开到工作量复杂度

### 更稳的答法

你可以这样说：

> 我先写一版 Kogge-Stone，因为它块内逻辑更短，更适合现场快速写对。  
> 如果继续追问更标准的 scan 模板和工作量复杂度，我再补 Blelloch，说明它的 upsweep / downsweep 和 exclusive scan。

---

## 9. 面试高频问答

### Q1：为什么 Kogge-Stone 更容易直接做 inclusive？

因为它每一轮都直接把更长前缀累加到当前位置，最后自然就是 inclusive 结果。

### Q2：为什么 Blelloch 天然先得到 exclusive？

因为 downsweep 前会先把根节点设成 `0``，然后把“父前缀”一路向下传播，所以第 `i` 个位置最终拿到的是它左边所有元素的和。

### Q3：为什么 Blelloch 工作量更优？

因为它不是每轮所有位置都重复算一遍前缀，而是先汇总成树，再把前缀信息分发下去，总体加法次数更少。

### Q4：那为什么还要学 Kogge-Stone？

因为它好写、好记、好讲。

很多面试场景里，先写一版 Kogge-Stone 比一上来写 Blelloch 更稳。

### Q5：工程里会直接手写这两种吗？

有时会，但更常见的是：

- warp-level scan
- block-level scan 融合优化
- decoupled look-back
- 直接用 CUB

面试里更重要的是你是否真正理解了 scan 的并行结构。

---

## 10. 一句话总结

- Kogge-Stone：更短、更直观、容易直接写 inclusive，但总工作量更大
- Blelloch：更标准、更省总工作量、天然先得到 exclusive，但实现更绕

如果只准备一版手撕答案，优先准备 Kogge-Stone。  
如果想应对后续追问，再把 Blelloch 的 upsweep / downsweep 补齐。
