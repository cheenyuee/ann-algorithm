[如何快速读懂开源代码？ (rstk.cn)](https://www.rstk.cn/news/8978.html?action=onClick)

[Idea查看源码，类的结构快捷键_类结构快捷键_好想学习呀的博客-CSDN博客](https://blog.csdn.net/weixin_46351306/article/details/113712965)

[百度自研高性能ANN检索引擎，开源了-开源中国 (erlo.vip)](https://www.erlo.vip/share/15/101177.html)

[各项结果排名第一！百度内容技术架构团队在BigANN中斩获佳绩 (baidu.com)](https://baijiahao.baidu.com/s?id=1724979311345232141&wfr=spider&for=pc)

## Introduction

Puck 支持余弦相似度、L2 (Euclidean) 和 IP (Inner Product, conditioned)。当两个向量归一化时，L2 距离等于 2 - 2 * cos。IP2COS 是一种将 IP 距离转换为 cos 距离的变换方法。搜索结果中的距离值始终为 L2。

Puck 使用压缩向量（PQ 之后）代替原始向量，默认情况下，内存消耗仅为原始向量的 1/4。

Tinker 需要保存相似点的关系，默认情况下内存消耗比原始向量多（小于 Nmslib）

##### Puck

包括反转索引和数据集多级量化的两层架构设计

> 由于使用 IVF，内存使用较少，适合billion级别大数据集

- 采用二层倒排索引架构，能更敏感的感知数据分布，从而非常高效的分割子空间，减少搜索范围；同时采用共享二级类聚中心的方式，大幅减少训练时间
- 训练时采用启发式迭代的方法，不断优化一二级类聚中心，通过等价空间变换，训练获得更好的数据分布描述
- 采用多层级量化加速查找，优先通过大尺度量化的小特征快速找到候选集，再通过稍大一些的量化特征二次查找
- 在各个检索环节打磨极致的剪枝， 针对loss函数，通过多种公式变化，最大程度减少在线检索计算量，缩短计算时间
- 严格的内存cacheline对齐和紧致排列，最大程度降低cache miss
- 支持大尺度的量化，单实例支持尽可能多的数据，针对大尺度量化定向优化，减少量化损失; 同时支持非均匀量化，更加适应各种纬度的特征

**Puck 功能拓展**

- 实时插入：支持无锁结构的实时插入，做到数据的实时更新。
- 条件查询：支持检索过程中的条件查询，从底层索引检索过程中就过滤掉不符合要求的结果，解决多路召回归并经常遇到的截断问题，更好满足组合检索的要求。
- 分布式建库：索引的构建过程支持分布式扩展，全量索引可以通过 map-reduce 一起建库，无需按分片 build，大大加快和简化建库流程。
- 自适应参数：ANN 方法检索参数众多，应用起来有不小门槛，不了解技术细节的用户并不容易找到最优参数，Puck 提供参数自适应功能，在大部分情况下使用默认参数即可得到很好效果 。

##### Tinker

比 Puck 消耗更多内存，但搜索性能比 Puck 更好。

> 适合较小数据集（如 10M、100M）

## 跑代码

使用sift数据集

https://github.com/erikbern/ann-benchmarks

在gflags内修改 feature_dim 参数（sift是128）

```
DEFINE_int32(feature_dim, 128, "feature dim");
```

run

```shell
# 指定 DMKLROOT
cmake -DCMAKE_BUILD_TYPE=Release -DMKLROOT=/opt/intel/oneapi/mkl/latest -DBLA_VENDOR=Intel10_64lp_seq -DBLA_STATIC=ON -B build .
# 把 sh 改为 bash
bash script/puck_train_control.sh -t -b
```

