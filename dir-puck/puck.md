[百度自研高性能ANN检索引擎](https://www.erlo.vip/share/15/101177.html)

[百度内容技术架构团队在BigANN中斩获佳绩](https://baijiahao.baidu.com/s?id=1724979311345232141&wfr=spider&for=pc)

## Introduction

#### 距离计算

Puck/Tinker 支持余弦相似度、L2 (Euclidean) 和 IP (Inner Product, conditioned)，但 Tinker 底层的 HNSW 图搜索始终使用的L2 (Euclidean) 距离。

- 当需要计算IP距离时，将 ip2cos 参数设为True：

  IP2COS 是一种将 IP 距离转换为 cos 距离的变换方法。当两个向量归一化时，L2 距离等于 2 - 2 * cos，因此可以将IP距离计算转换为L2距离计算，搜索结果中的距离值始终为 L2。

  > **前提条件：向量数据的L2范数小于等于1**，Yandex text2image 数据集具有此特征

- 当需要计算余弦距离时，将 whether_norm 参数设为True：

  余弦距离与规范化后的L2距离等价，这也是为什么搜索时要对query 进行 normalization，然后使用 L2 距离在图上搜索。

**puck-in-mem 文件**

根据距离度量确定数据的转换方式

```python
				#根据距离计算方式调整
        whether_norm = False
        py_puck_api.update_gflag('whether_norm', 'false')
        ip2cos = 0
        if ds.distance() == "angular":
            whether_norm = True # 余弦距离或角距离
            py_puck_api.update_gflag('whether_norm', 'true')
        elif ds.distance() == "ip":
            ip2cos = 1	# 内积距离
        py_puck_api.update_gflag('ip2cos', '%d'%(ip2cos))
```

将转换后的数据保存在 all_feature_file 文件中，转换后的数据直接使用L2距离进行计算

```python
				all_feature_file = open("%s/all_data.feat.bin"%(self.index_name(dataset)), 'wb')
        add_part=100000
        for xblock in ds.get_dataset_iterator(bs=add_part):
            for x in xblock:
                feat = x
                if(whether_norm):# 用规范化后的L2距离实现余弦距离或角距离计算
                    feat = feat / np.sqrt(np.dot(feat, feat))
                elif(ip2cos > 0):# 用 ip2cos 后的L2距离实现内积距离计算
                    norm = np.dot(feat, feat)
                    if norm > 1.0: # 前提条件
                        print("not support, please contact yinjie06")
                        return False
                    feat = np.append(feat, math.sqrt(1.0 - norm)) # 处理后会增加一个维度
                buf = struct.pack('i', len(feat))
                all_feature_file.write(buf)
                buf = struct.pack('f' * len(feat), *feat)
                all_feature_file.write(buf)
        all_feature_file.close()
```

这里python文件的query函数没有对query向量做处理，处理过程放在了底层C++代码 normalization 函数中

```python
def query(self, X, topK):
  	n, d = X.shape
    self.index.search(n, swig_ptr(X), topK, swig_ptr(self.res[0]), swig_ptr(self.res[1]))
```

**HierarchicalClusterIndex::normalization 函数**

搜索时也对query向量做对应处理

```c++
const float* HierarchicalClusterIndex::normalization(SearchContext* context, const float* feature) {
    SearchCellData& search_cell_data = context->get_search_cell_data();
    if (_conf.ip2cos == 1) {
      	// 用 ip2cos 后的L2距离实现内积距离计算，
        uint32_t dim = _conf.feature_dim - 1; // feature_dim 是 ip2cos 增加后的维度，减1即实际维度
        memset(search_cell_data.query_norm, 0, _conf.feature_dim);
        memcpy(search_cell_data.query_norm, feature, sizeof(float) * dim);
      	// 返回向量实际上是在 query 的特征向量末尾添加了一维，值设为0
        return search_cell_data.query_norm;
    } else if (_conf.whether_norm) {
      	// 用规范化后的L2距离实现余弦距离或角距离计算
        memcpy(search_cell_data.query_norm, feature, sizeof(float) * _conf.feature_dim);
        float norm = cblas_snrm2(_conf.feature_dim, search_cell_data.query_norm, 1); // 计算L2范数
        if (norm < 1e-6) {
            LOG(ERROR) << "query norm is " << norm << ", could not be normalize";
            return nullptr;
        }
        cblas_sscal(_conf.feature_dim, 1.0 / norm, search_cell_data.query_norm, 1);//规范化，即乘1.0 / norm
      	// 返回向量是对 query 的特征向量做规范化
        return search_cell_data.query_norm;
    }
    return feature;
}
```

> 参考 https://developer.apple.com/documentation/accelerate/1513250-cblas_snrm2
>
> cblas：C Interface to Basic Linear Algebra Subprograms，即C语言的 BLAS 线性代数计算库
>
> cblas_snrm2 函数：计算L2范数（开方后的值），s代表单精度single precision，nrm2代表L2范数
>
> cblas_sscal 函数：向量所有维度同乘一个值，s代表单精度，scal代表同乘一个标量值，即向量长度变大或变小

#### Puck

包括反转索引和数据集多级量化的两层架构设计，使用压缩向量（PQ 之后）代替原始向量，默认情况下，内存消耗仅为原始向量的 1/4。

> 由于使用 IVF，内存使用较少，适合billion级别大数据集

- 采用**二层倒排索引架构**，能更敏感的感知数据分布，从而非常高效的分割子空间，减少搜索范围；

  同时采用**共享二级类聚中心**的方式，大幅减少训练时间

- 训练时采用启发式迭代的方法，不断优化一二级类聚中心，通过等价空间变换，训练获得更好的数据分布描述

- 采用多层级量化加速查找，优先通过大尺度量化的小特征快速找到候选集，再通过稍大一些的量化特征二次查找

- 在各个检索环节打磨极致的剪枝， 针对loss函数，通过多种公式变化，最大程度减少在线检索计算量，缩短计算时间

- 严格的内存cacheline对齐和紧致排列，最大程度降低cache miss

- 支持大尺度的量化，单实例支持尽可能多的数据，针对大尺度量化定向优化，减少量化损失; 同时支持非均匀量化，更加适应各种纬度的特征

**Puck 功能拓展**

- 实时插入：支持无锁结构的实时插入，做到数据的实时更新

#### Tinker

Tinker 需要保存相似点的关系，比 Puck 消耗更多内存（但小于 Nmslib），但搜索性能比 Puck 更好

> 适合较小数据集（如 10M、100M）

**Search**

数据是明码读的，不能直接读sift中的query文件，query文件格式参考 init-feature-example 

```shell
ln -s build_tools/puck_index . # 建议一个软连接

# puck
./bin/search_client /home/cy/repo/puck/tools/demo/init-feature-example RECALL_FILE_NAME --flagfile=conf/puck.conf

# tinker
./bin/search_client /home/cy/repo/puck/tools/demo/init-feature-example RECALL_FILE_NAME --flagfile=conf/tinker.conf
```

## HierarchicalClusterIndex

HierarchicalClusterIndex 是升级版 IVF ，具有两层聚类

> hierarchical k-means ：递归地在产生的簇中运行 k-means
>
> Hierarchical Cluster Index：这里的索引具有两层，与 Hierarchical k-means 不同的是，hierarchical_cluster_index 的二级聚类是对残差进行聚类，而不是在一级聚类产生的簇内进行聚类

##### 索引训练

- coarse_cluster

  粗粒度的一级聚类，对向量使用 k-means 得到一级聚类中心（k 等于 coarse_cluster_count）

- fine_cluster：

  细粒度的二级聚类，对残差向量（向量减去对应的一级聚类中心）使用 k-means 得到二级聚类中心（k 等于 fine_cluster_count）

  二级聚类把相近的残差向量聚集到了一起，从而形成了两层的索引结构

  > 如何理解残差向量相近？
  >
  > 残差向量相近意味着不同原向量虽然可能位于不同的一级 cluster 中，但与对应一级聚类中心的相对位置差不多，因此有：
  >
  > - 二级 cluster 中的残差向量可能来自不同的一级 cluster（采用共享二级类聚中心的方式，大幅减少训练时间）
  > - 一级 cluster 中的向量可能去往不同的二级 cluster（体现出了与 hierarchical k-means 类似的层次结构，即二层倒排索引架构，高效的分割子空间，减少搜索范围）
  >
  > 每个二级 cluster 由 coarse_cluster_count  个一级 cluster 共享，实际产生了 coarse_cluster_count * fine_cluster_count 个二级聚类中心

主要函数：

HierarchicalClusterIndex::train 函数：训练码本（计算一二级聚类中心）

HierarchicalClusterIndex::build 函数：建库（计算样本最近的1个聚类中心，即样本所属的二级 cluster/cell）

HierarchicalClusterIndex::single_build 函数：实时入库，用于插入1个新样本

```C++
// HierarchicalClusterIndex::train 函数
		for (int ite = 0; ite < FLAGS_kmeans_iterations_count; ++ite) {
        //kmeans聚类得到一级聚类中心
        float err = kmeans_cluster.kmeans(_conf.feature_dim, kmenas_point_cnt, _conf.coarse_cluster_count,
                                          train_vocab.get(),
                                          coarse_init_vocab.get(), nullptr, cluster_assign.get());
        //计算残差（为了进行二级聚类）
        memcpy(train_vocab.get(), kmeans_train_vocab, sizeof(float) * kmenas_point_cnt * _conf.feature_dim);
        for (uint32_t i = 0; i < kmenas_point_cnt; ++i) {
            int assign_id = cluster_assign.get()[i];
            cblas_saxpy(_conf.feature_dim, -1.0, coarse_init_vocab.get() + assign_id * _conf.feature_dim, 1,
                        train_vocab.get() + i * _conf.feature_dim, 1);
        }

        //残差向量kmeans聚类得到二级聚类中心
        err = kmeans_cluster.kmeans(_conf.feature_dim, kmenas_point_cnt, _conf.fine_cluster_count,
                                    train_vocab.get(),
                                    fine_init_vocab.get(), nullptr, cluster_assign.get());
        //计算残差
        memcpy(train_vocab.get(), kmeans_train_vocab, sizeof(float) * kmenas_point_cnt * _conf.feature_dim);
        for (uint32_t i = 0; i < kmenas_point_cnt; ++i) {
            int assign_id = fine_vocab_assign.get()[i];
            cblas_saxpy(_conf.feature_dim, -1.0, _fine_vocab + assign_id * _conf.feature_dim, 1,
                        train_vocab.get() + i * _conf.feature_dim, 1);
        }
    }
```

> 注意：当数据量较少时，使用全部数据进行训练，如果数据量太大，随机抽样选择部分数据进行训练

##### 向量搜索

- 计算 query 与所有一级聚类中心的距离并排序（堆排）

  > 对应 search_nearest_coarse_cluster函数

- 计算 query 与 search_coarse_count 个最近的一级聚类中心下的所有二级聚类中心的距离并排序

  > 对应 search_nearest_fine_cluster函数
  >
  > 总共 search_coarse_count * fine_cluster_count 个二级聚类中心

- 按距离从小到大访问二级聚类中心，将对应二级 cluster 中的点加入最大堆中（使用最大堆是为了进行剪枝），直到访问完 neighbors_count 个点

  > 访问过程中维护一个最大堆，堆的大小为 k，堆顶保存当前 topk 的最大距离，只有当前访问点的距离小于堆顶距离，才会将访问点加入最大堆中，从而进行剪枝，访问完即求得 topk

  最后对结果队列进行排序（堆排）

  > 对应 flat_topN_points 函数

```C++
    for (uint32_t l = 0; l < _conf.search_coarse_count; ++l) {
        int coarse_id = coarse_tag[l];
        //计算query与当前一级聚类中心下cell的距离
				//当前搜索的一级 cluster
        for (uint32_t idx = 0; idx < _conf.fine_cluster_count; ++idx) {
 						//当前搜索的一级 cluster 下的二级 cluster
        }
    }
```

## PuckIndex

puckIndex 是升级版 IVF + PQ

##### 索引训练

- 升级版 IVF：训练一个 HierarchicalClusterIndex 
- 升级版 PQ：训练两个PQ
  - _filter_quantization：大尺度量化，用来过滤求 filter_topk（提高速度）
  - _pq_quantization：小尺度量化，用来对过滤出的小规模样本重新排序求 topk（提高精度）

##### 向量搜索

- 计算 query 与所有一级聚类中心的距离并排序（堆排）

- 遍历与 query 最近的 search_coarse_count 个一级聚类中心下的所有二级聚类中心

  > 总共 search_coarse_count * fine_cluster_count 个二级聚类中心

- 将所有二级聚类中心下的所有点都加入 filter 队列

- 对 filter 队列进行排序（使用 _filter_quantization 的PQ距离），求 filter_topk

- 将 filter_topk 个点加入结果队列进行重排（使用 _pq_quantization 的PQ距离），求 topk

```c++
    //通常取1/4量化，对过滤出的小规模样本重新排序时使用
    std::unique_ptr<Quantization> _pq_quantization;
    //大尺度量化，用来过滤
    std::unique_ptr<Quantization> _filter_quantization;
```

## TinkerIndex

TinkerIndex 是升级版 HNSW

##### 索引训练

- 训练一个 HierarchicalClusterIndex，用于确定图搜索的 entry points
- HNSW：训练一个图，用于贪心搜索

##### 向量搜索

- 计算 query 与所有一级聚类中心的距离并排序（堆排）

- 遍历与 query 最近的 search_coarse_count 个一级聚类中心下的所有二级聚类中心，找到距离 query 最近的 cell

  > 总共 search_coarse_count * fine_cluster_count 个cell

- 将 nearest_cell 内下的所有点设为 entry points，加入搜索队列

  > 注意：从代码上看，如果数据集较小， cell 过多，**可能导致找不到 ep**，此时将无法进行搜索

- 在 HNSW 的最底层 level0 上进行贪心搜索，返回 TopK

## 文件结构

- tools
  - train.cpp：用于 train
  - build.cpp：用于 build
  - demo/conf：用于配置训练参数
  - script
    - initProcessData.py：将明码数据转换为二进制形式的数据处理脚本
    - puck_train_control.sh：训练脚本（调用 train.cpp 和 build.cpp）

- demo
  - insert_demo.cpp：用于测试 puck::RealtimeInsertPuckIndex
  - search_client.cpp：用于 query
  - conf：用于配制查询参数
- test：使用 gtest 进行测试

## 代码运行

**修改 feature_dim 参数（sift是128）**

```
DEFINE_int32(feature_dim, 128, "feature dim");
```

```
--feature_dim=128
```

**Build project**

```shell
# 指定 DMKLROOT
cmake -DCMAKE_BUILD_TYPE=Release -DMKLROOT=/opt/intel/oneapi/mkl/latest -DBLA_VENDOR=Intel10_64lp_seq -DBLA_STATIC=ON -B build .

cd build && make && make install # make install 将生成 output 目录
```

**Train and build**

使用sift的base数据

```shell
# 把 sift 数据拷贝到指定目录下
cd output/build_tools
mkdir puck_index
cp ~/data/siftsmall/siftsmall_base.fvecs puck_index/all_data.feat.bin
# 可以在 puck_train_control.sh 文件内设置 conf_file 文件路径，在 conf_file 内配置训练参数
# 如设置 conf_file=conf/tinker_train.conf 来使用 tinker 索引
bash script/puck_train_control.sh -t -b
```

**search**

```shell
cd output/
ln -s build_tools/puck_index .
./bin/search_client ../../tools/demo/init-feature-example RECALL_FILE_NAME --flagfile=conf/puck.conf # 使用 puck
./bin/search_client ../../tools/demo/init-feature-example RECALL_FILE_NAME --flagfile=conf/tinker.conf # 使用 tinker
```

> ## 在比赛框架运行Puck和Tinker
>
>在 big-ann-benchmarks/neurips23/ood/ 内添加 puck 文件夹，放三个文件：
> 
>- Dockerfile：参考 ann-benchmarks/install/Dockerfile.puck_inmem，改成 `FROM neurips23`
> - config.yaml：参考 puck 仓库的 ann-benchmarks/algos.yaml
>  - index_type:1 表示puck（默认）
>   - index_type:2 表示tinker
>- puck_inmem.py：参考 puck 仓库 ann-benchmarks/benchmark/algorithms/puck_inmem.py
> 
> 改动地方见照片

