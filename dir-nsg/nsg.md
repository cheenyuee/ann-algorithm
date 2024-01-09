## 参数说明

构造：

- `L` （构造时的搜索队列大小）controls the quality of the NSG, the larger the better.

- `R` （最大出度）controls the index size of the graph, the best R is related to the intrinsic dimension of the dataset.

- `C` controls the maximum candidate pool size during NSG contruction.

  > 参数`C`控制**最大 neighbor candidate 数量**：
  >
  > - NSG 通过对 neighbor candidate 进行剪枝来选择 neighbor
  > - NSG 的 neighbor candidate 来自搜索时的所有途经访问点集合
  > - 剪枝时最多考虑前`C`个距离最小的途经访问点，即所有途经访问点集合的子集，避免剪枝时间过长

搜索：

- `SEARCH_L` controls the quality of the search results, the larger the better but slower. The `SEARCH_L` cannot be samller than the `SEARCH_K`