# pyglass

## Features

- Supports multiple graph algorithms, like [**HNSW**](https://github.com/nmslib/hnswlib) and [**NSG**](https://github.com/ZJULearning/nsg).
- Sophisticated memory management and data structure design, very low memory footprint.
- It's high performant.

## Key points

在构图上：

使用[**HNSW**](https://github.com/nmslib/hnswlib) 或 [**NSG**](https://github.com/ZJULearning/nsg) 算法进行构图

在搜索上：

- 减少内存占用

  使用标量量化技术（SQ8、SQ4）

  SQ8：使用uint8型数据（1字节）量化表示浮点数数据（4字节），减小内存占用

  量化方法：

  - 归一化：对向量每个维度进行归一化，将原来的float数据范围压缩到 0 到 1

  - scale：乘以255即满足 uint8 的表示范围，用 8 bit 量化浮点数（也可只用 4 bit，即SQ4）

    > 浮点数取值空间映射到整数取值空间中

  ```C++
  for (int64_t i = 0; i < n; ++i) {
     for (int64_t j = 0; j < d; ++j) {
        mx[j] = std::max(mx[j], data[i * d + j]); // 第 j 个维度的最大值
        mi[j] = std::min(mi[j], data[i * d + j]); // 第 j 个维度的最小值
     }
  }
  for (int64_t j = 0; j < d; ++j) {
    dif[j] = mx[j] - mi[j]; // 第 j 个维度的最大差值
  }
  float x = (from[j] - mi[j]) / dif[j]; // 归一化，x 取值范围为 0 到 1
  uint8_t y = x * 255; // scale x 为 y， 取值范围为 0 到 255，即 uint8_t 取值范围
  to[j] = y;
  ```

  距离计算：

  将量化后的数据 scale 回原来的浮点数取值空间和query点进行距离计算

- 提升搜索性能

  使用 prefetch 技术，引入两个参数：

  - po（prefetch out-neighbors）：预取邻居数量
  - pl（prefetch lines）：预取数据行数（一行为 64B）

  ```C++
  graph.prefetch(u, graph_po); // 预取点 u 的邻居 id，graph_po 是预取行数
  for (int i = 0; i < po; ++i) { // 预取量化后的数据， po 是预取邻居数，pl 是预取数据行数
      int to = graph.at(u, i);
      computer.prefetch(to, pl);
  ```

  在搜索前，选择最优的 prefetch 参数 po、pl

  在 base 数据集内随机抽取一定数量的 point，如1000个，作为优化阶段的query集，统计不同 prefetch 参数下搜索完query集内所有query点的时间（会使用 omp 并行进行搜索），从而选出用时最短的 prefetch 参数作为搜索阶段的最优参数

