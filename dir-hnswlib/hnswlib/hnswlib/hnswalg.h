#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>

/**
 * 写在前面：
 * Internal_Id: 指 label 是 data_level0_memory_ 中第几个元素
 * 标记删除：指标记某个点被被删除，但仍然被保留在图上，搜索时可以进行扩展与访问，只是不会被加入结果队列(搜索结果将不会包括被标记删除的点)
 * Filter：构图完全与Filter无关，仅在查询时在加入结果队列时进行条件判断来实现 Filter
 * 剪枝算法：HNSW 使用的剪枝算法和 NSG 是相同的，见 mutuallyConnectNewElement 函数，注意正向边剪枝和反向边剪枝所用的最大出度是不同的(M_ 和 maxM_/maxM0_)
 * 锁操作：构图时有写操作需要使用锁，查询时只有读操作不需要使用锁
 */
namespace hnswlib {
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;

    template<typename dist_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t> {
    public:
        static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
        static const unsigned char DELETE_MARK = 0x01;

        size_t max_elements_{0}; // 最大可容纳的元素数目
        mutable std::atomic<size_t> cur_element_count{0};  // current number of elements，当前元素数目
        size_t size_data_per_element_{0}; // 一个元素的长度(字节数)，使用方法见 get_linklist0 函数
        size_t size_links_per_element_{0}; // 一个点保存一层邻居所需要的字节数，使用方法见 get_linklist 函数
        mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements，当前被标记删除的元素数目
        size_t M_{0}; // 用户设置的最大出度参数，是构图时添加正向边时的最大出度
        size_t maxM_{0}; // 构图时第0层以上添加反向边时的最大出度，等于 M_
        size_t maxM0_{0}; // 构图时第0层添加反向边时的最大出度，等于 M_ * 2
        size_t ef_construction_{0}; // 构图时的搜索队列大小
        size_t ef_{0}; // 查询时的搜索队列大小

        double mult_{0.0}, revSize_{0.0}; // mult_ 是计算层高时用到的常数，revSize_ 好像没有用
        int maxlevel_{0}; // 当前的最大层高，也是入口点的层高(将被构造函数初始化为 -1，此时还没有任何点被插入)

        VisitedListPool *visited_list_pool_{nullptr};

        /**
         * 用 label_op_locks_[getLabelOpMutex(label)]，对 label 上锁(锁的数量为 MAX_LABEL_OPERATION_LOCKS)
         * 作用：保证同时只有一个线程在对某个 label 操作，例如，不能插入某个 label 时又在删除这个 label，只有插入完成后才能进行删除操作
         */
        // Locks operations with element by label value
        mutable std::vector<std::mutex> label_op_locks_; //

        std::mutex global; // 全局锁，用于插入操作更新入口点时对整个图进行上锁

        /**
         * 用 link_list_locks_[Internal_Id] 对 Internal_Id 节点上锁(锁的数量为 max_elements)，包括：
         * 对 linkLists_[Internal_Id] 上锁、
         * 对 data_level0_memory_ + internal_id * size_data_per_element_ 上锁
         * 作用：避免对 Internal_Id 节点进行 check 时邻居被修改，保证多线程插入时对某个点的互斥访问
         */
        std::vector<std::mutex> link_list_locks_; // 细粒度锁，用于对某个点上锁（读写某个点时使用）

        tableint enterpoint_node_{0}; // 保持入口点位于最高层(将被构造函数初始化为 -1，此时还没有任何点被插入)

        size_t size_links_level0_{0}; // level0 层一个元素的 links 信息长度，包括 2 字节邻居数，2 字节删除标记，maxM0_* sizeof(tableint) 字节邻居列表
        size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{0}; // 分别是 base 数据、邻居信息、label分别在元素中的偏移量

        /**
         * data_level0_memory_：一个元素的信息依次为： (2 字节保存邻居数，2 字节保存删除标记，maxM0_* sizeof(tableint) 字节保存邻居列表)， (base数据)，(label)
         * linkLists_[Internal_Id]：2 字节保存邻居数，2 字节没有用，maxM_* sizeof(tableint) 字节保存邻居列表
         */
        char *data_level0_memory_{nullptr}; // 保存每个元素的信息，访问方法见 get_linklist0 函数，元素所在第几个位置就是元素的 internal_id
        char **linkLists_{nullptr}; // 保存0层以上的邻居关系，访问方法见 get_linklist 函数
        std::vector<int> element_levels_;  // keeps level of each element，element_levels_[i] 表示第i个元素的层高

        size_t data_size_{0}; // 一条 base 数据的长度(字节数)

        DISTFUNC<dist_t> fstdistfunc_; // 距离函数指针
        void *dist_func_param_{nullptr}; // 距离函数参数，实际上是参与距离计算维度

        mutable std::mutex label_lookup_lock; // lock for label_lookup_，用于对 label_lookup_ 上锁
        std::unordered_map<labeltype, tableint> label_lookup_; // 维护外部 label 到 internal_id 的映射关系

        std::default_random_engine level_generator_; // 随机数生成器，用于确定点的层高
        std::default_random_engine update_probability_generator_; // 随机数生成器，updatePoint 时使用

        /**
         * 这里 metric_distance_computations 和 metric_hops 的统计了所有进行过的 query，而不是一个 query
         */
        mutable std::atomic<long> metric_distance_computations{0}; // 用于统计搜索过程的距离计算次数
        mutable std::atomic<long> metric_hops{0}; // 用于统计搜索过程的跳数（check 点的数量）

        bool allow_replace_deleted_ = false;  // flag to replace deleted elements (marked as deleted) during insertions

        std::mutex deleted_elements_lock;  // lock for deleted_elements，用于对 deleted_elements 上锁
        std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements，记录被标记删除的元素内部id


        HierarchicalNSW(SpaceInterface<dist_t> *s) {
        }

        /**
         * allow_replace_deleted: enables replacing of deleted elements with new added ones.
         */
        HierarchicalNSW(
                SpaceInterface<dist_t> *s,
                const std::string &location,
                bool nmslib = false,
                size_t max_elements = 0,
                bool allow_replace_deleted = false)
                : allow_replace_deleted_(allow_replace_deleted) {
            loadIndex(location, s, max_elements);
        }

        /**
         * 构造函数
         * @param s 度量空间
         * @param max_elements 最大元素数
         * @param M 添加正向边时的最大邻居数
         * @param ef_construction 构造时的搜索结果队列大小
         * @param random_seed 随机种子，用于确定插入点的层高时产生随机数
         * @param allow_replace_deleted 是否允许把标记删除的元素替换为新元素
         */
        HierarchicalNSW(
                SpaceInterface<dist_t> *s,
                size_t max_elements,
                size_t M = 16,
                size_t ef_construction = 200,
                size_t random_seed = 100,
                bool allow_replace_deleted = false)
                : link_list_locks_(max_elements),
                  label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
                  element_levels_(max_elements),
                  allow_replace_deleted_(allow_replace_deleted) {
            max_elements_ = max_elements;
            num_deleted_ = 0;
            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            M_ = M;
            maxM_ = M_;
            maxM0_ = M_ * 2;
            ef_construction_ = std::max(ef_construction, M_);
            ef_ = 10;

            level_generator_.seed(random_seed); // 设置随机数生成器的种子
            update_probability_generator_.seed(random_seed + 1);///

            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
            offsetData_ = size_links_level0_;
            label_offset_ = size_links_level0_ + data_size_;
            offsetLevel0_ = 0;

            data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");

            cur_element_count = 0;

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            // initializations for special treatment of the first node
            enterpoint_node_ = -1;
            maxlevel_ = -1;

            linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
            mult_ = 1 / log(1.0 * M_); // mult_ 与 M_ 相关
            revSize_ = 1.0 / mult_;
        }


        ~HierarchicalNSW() {
            free(data_level0_memory_);
            for (tableint i = 0; i < cur_element_count; i++) {
                if (element_levels_[i] > 0)
                    free(linkLists_[i]);
            }
            free(linkLists_);
            delete visited_list_pool_;
        }

        /**
         * 用于 priority_queue 构建大顶堆进行比较的 struct
         */
        struct CompareByFirst {
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept {
                return a.first < b.first;
            }
        };

        /**
         * 设置搜索队列大小
         */
        void setEf(size_t ef) {
            ef_ = ef;
        }

        /**
         * 获取 label 对于的锁，设置了最多 MAX_LABEL_OPERATION_LOCKS 个锁
         */
        inline std::mutex &getLabelOpMutex(labeltype label) const {
            // calculate hash
            size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
            return label_op_locks_[lock_id];
        }

        /**
         * 根据 internal_id 获取外部 label
         */
        inline labeltype getExternalLabel(tableint internal_id) const {
            labeltype return_label;
            memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_),
                   sizeof(labeltype));
            return return_label;
        }

        /**
         * 更新 data_level0_memory_ 中第 internal_id 个位置元素的 label
         */
        inline void setExternalLabel(tableint internal_id, labeltype label) const {
            memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label,
                   sizeof(labeltype));
        }

        /**
         * 获取 data_level0_memory_ 中第 internal_id 个位置元素的 label 指针
         */
        inline labeltype *getExternalLabeLp(tableint internal_id) const {
            return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
        }

        /**
         * 获取 data_level0_memory_ 中第 internal_id 个位置元素的 base 数据指针
         */
        inline char *getDataByInternalId(tableint internal_id) const {
            return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }

        /**
         * 获取新插入元素层高
         */
        int getRandomLevel(double reverse_size) {
            std::uniform_real_distribution<double> distribution(0.0, 1.0); // 定义一个数值范围0到1的均匀分布
            double r = -log(distribution(level_generator_)) * reverse_size; // 生成均匀分布的随机浮点数
            return (int) r; // 向下取整
        }

        size_t getMaxElements() {
            return max_elements_;
        }

        size_t getCurrentElementCount() {
            return cur_element_count;
        }

        size_t getDeletedCount() {
            return num_deleted_;
        }


        /**
         *  用于构图时的搜索
         *  ep_id 作为入口点，在第 layer 层进行搜索（layer 可以大于 0 也可以等于 0）
         */
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass; // 记录访问状态的数组
            vl_type visited_array_tag = vl->curV; // 如果某个点被访问，则将 visited_array 对应位置设为 visited_array_tag

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates; // 结果队列(大顶堆)
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet; // 搜索队列(大顶堆)

            dist_t lowerBound;
            if (!isMarkedDeleted(ep_id)) { // 如果 ep_id 没有被标记删除，则同时加入搜索队列和结果队列
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                top_candidates.emplace(dist, ep_id);
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
            } else { // 如果 ep_id 已经被标记删除，则只加入搜索队列，不加入结果队列
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag; // 标记 ep_id 已访问(加入过搜索队列的点都应标记已经访问，避免重复加入)

            while (!candidateSet.empty()) {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top(); // 取出搜索队列中距离最小的点进行 check

                if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                    // 停止条件：当前 check 点的距离大于 lowerBound 且结果数量足够
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;

                // 对当前 check 的 curNodeNum 点上锁
                std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

                // 读取 curNodeNum 的邻居节点
                int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                if (layer == 0) {
                    data = (int *) get_linklist0(curNodeNum);
                } else {
                    data = (int *) get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                }
                size_t size = getListCount((linklistsizeint *) data);
                tableint *datal = (tableint *) (data + 1); // 第一个数据是邻居数量，因此这里要加1
#ifdef USE_SSE
                // 预取第一次 for 循环将访问的内存地址，包括 visited_array 和 base 数据
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++) { // 依次扩展每个邻居，即 candidate_id
                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    // 预取下次 for 循环将访问的内存地址，包括 visited_array 和 base 数据
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag) continue; // 如果已经被 visit 过，则跳过
                    visited_array[candidate_id] = visited_array_tag; // 否则进行 visit

                    // 计算距离
                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                        // 这个条件可以减少扩展点的数量
                        // 但如果距离小于 lowerBound 或结果数不足，必须进行扩展

                        candidateSet.emplace(-dist1, candidate_id); // 将邻居加入搜索队列，不管该邻居有没有被标记删除
#ifdef USE_SSE
                        // 预取下次 while 循环将访问的内存地址，即下个 check 点的 base 数据，用于距离计算
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id)) // 如果该邻居没有被标记删除，则将邻居加入结果队列，否则不加入结果队列
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();

                        if (!top_candidates.empty()) // 更新 lowerBound(当前结果队列中的距离最大值)
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl); // 搜索完成，释放 vl

            return top_candidates;
        }


        /**
         * 用于查询时的搜索
         * ep_id 作为入口点，在第 0 层进行搜索
         * 注释可以参考 searchBaseLayer，两者差不多
         * @tparam has_deletions 是否有元素被标记删除
         * @tparam collect_metrics 是否收集统计信息，包括 metric_hops 和 metric_distance_computations
         * @param isIdAllowed 用于进行 Filter，在加入结果队列时进行条件判断来实现Filter
         * @return
         */
        template<bool has_deletions, bool collect_metrics = false>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef,
                          BaseFilterFunctor *isIdAllowed = nullptr) const {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            dist_t lowerBound;
            if ((!has_deletions || !isMarkedDeleted(ep_id)) &&
                ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id)))) {
                // 如果 ep_id 没有被标记删除，且当有 Filter 时满足条件，则同时加入搜索队列和结果队列
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            } else { // 如果 ep_id 已经被标记删除或有 Filter 时不满足条件，则只加入搜索队列，不加入结果队列
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;

            while (!candidate_set.empty()) {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                if ((-current_node_pair.first) > lowerBound &&
                    (top_candidates.size() == ef || (!isIdAllowed && !has_deletions))) {
                    break;
                }
                candidate_set.pop();

                // 读取 current_node_id 的邻居节点
                tableint current_node_id = current_node_pair.second;
                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint *) data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
                if (collect_metrics) { // 收集统计信息
                    metric_hops++;
                    metric_distance_computations += size;
                }

#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0);  ////////////
#endif
                    if (!(visited_array[candidate_id] == visited_array_tag)) {
                        visited_array[candidate_id] = visited_array_tag;

                        char *currObj1 = (getDataByInternalId(candidate_id));
                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                        if (top_candidates.size() < ef || lowerBound > dist) {
                            candidate_set.emplace(-dist, candidate_id); // 将邻居加入搜索队列，不管该邻居有没有被标记删除或是否满足Filter条件
#ifdef USE_SSE
                            _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                         offsetLevel0_,  ///////////
                                         _MM_HINT_T0);  ////////////////////////
#endif

                            // 如果该邻居没有被标记删除，且当有 Filter 时满足条件，则将邻居加入结果队列，否则不加入结果队列
                            if ((!has_deletions || !isMarkedDeleted(candidate_id)) &&
                                ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))
                                top_candidates.emplace(dist, candidate_id);

                            if (top_candidates.size() > ef)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }


        /**
         * 启发式地对邻居进行剪枝（剪枝算法和 NSG/MRNG 相同）
         * @param top_candidates 候选邻居集
         * @param M 最大邻居数
         */
        void getNeighborsByHeuristic2(
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
                const size_t M) {
            if (top_candidates.size() < M) { // 不需要进行剪枝
                return;
            }

            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            std::vector<std::pair<dist_t, tableint>> return_list; // 结果邻居集

            // 把 top_candidates 中的数据转移到 queue_closest 中
            while (top_candidates.size() > 0) {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }
            // 此时 top_candidates 为空

            while (queue_closest.size()) { // 按距离从小到大弹出
                if (return_list.size() >= M)
                    break;
                std::pair<dist_t, tableint> curent_pair = queue_closest.top(); // 当前候选
                dist_t dist_to_query = -curent_pair.first; // 当前距离
                queue_closest.pop();
                bool good = true;

                // 判断是否保留该邻居
                for (std::pair<dist_t, tableint> second_pair: return_list) {
                    dist_t curdist =
                            fstdistfunc_(getDataByInternalId(second_pair.second),
                                         getDataByInternalId(curent_pair.second),
                                         dist_func_param_);
                    if (curdist < dist_to_query) {
                        good = false; // 进行剪枝，不保留该邻居
                        break;
                    }
                }
                if (good) { // 根据判断结果更新结果邻居集
                    return_list.push_back(curent_pair);
                }
            }

            // 更新 top_candidates 为结果邻居集
            for (std::pair<dist_t, tableint> curent_pair: return_list) {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }


        /**
         * 获取 internal_id 对应元素在第0层的邻居
         * @return 指向一片连续地址的指针，指向地址的内容构成如下：
         * 第一个 4 字节中，头 2 字节表示邻居总数，后 2 字节表示删除标记
         * 之后每 4 字节表示一个邻居编号
         */
        linklistsizeint *get_linklist0(tableint internal_id) const {
            return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        }


        linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
            return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        }


        /**
         * 获取 internal_id 对应元素在 level 层的邻居（level > 0）
         * @return 指向一片连续地址的指针，指向地址的内容构成如下：
         * 第一个 4 字节中，头 2 字节表示邻居总数，后 2 字节没有用
         * 之后每 4 字节表示一个邻居编号
         */
        linklistsizeint *get_linklist(tableint internal_id, int level) const {
            return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
        }

        /**
         * 获取 internal_id 对应元素在 level 层的邻居
         */
        linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
            return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
        }

        /**
         * 给插入点/更新点选边，先通过剪枝选择正向边，再通过剪枝选择反向边
         * @param data_point 点的 base 数据
         * @param cur_c 点的 Internal_Id
         * @param top_candidates 点的候选邻居（当前层的搜索结果）
         * @param level 选边层次
         * @param isUpdate true:更新旧点 cur_c 的边；false:对新插入点 cur_c 选边
         * @return 距离 cur_c 最近的邻居
         */
        tableint mutuallyConnectNewElement(
                const void *data_point,
                tableint cur_c,
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
                int level,
                bool isUpdate) {
            size_t Mcurmax = level ? maxM_ : maxM0_; // Mcurmax 是添加反向边时 cur_c 的邻居的最大出度

            // 对候选邻居集进行剪枝，剪枝后的就是选择的邻居
            getNeighborsByHeuristic2(top_candidates, M_); // M_ 是 cur_c 的最大出度
            if (top_candidates.size() > M_)
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

            // 将选择的邻居保存到 selectedNeighbors 中
            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(M_);
            while (top_candidates.size() > 0) {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }

            // 由于 top_candidates 是大顶堆，因此排在 selectedNeighbors 末尾的距离最小
            tableint next_closest_entry_point = selectedNeighbors.back();  // 设置返回值

            {
                // 这个作用域内 cur_c 点将被锁住
                // lock only during the update
                // because during the addition the lock for cur_c is already acquired
                std::unique_lock<std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
                if (isUpdate) { // 如果是旧点，则此时还没获取该点的锁，需要进行获取，如果是新点，则已经获取，不需要重复获取
                    lock.lock();
                }
                // 获取指向 cur_c 的邻居数组的指针
                linklistsizeint *ll_cur;
                if (level == 0)
                    ll_cur = get_linklist0(cur_c);
                else
                    ll_cur = get_linklist(cur_c, level);

                if (*ll_cur && !isUpdate) { // 现在还没有任何邻居，因此邻居数不应大于0
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                // 先设置 cur_c 的邻居数
                setListCount(ll_cur, selectedNeighbors.size());
                // 再依次设置 cur_c 的邻居（添加 cur_c 到 cur_c 的每个邻居的边）
                tableint *data = (tableint *) (ll_cur + 1);
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                    if (data[idx] && !isUpdate) // 如果是新点，则 data[idx] 应该被初始化为 0
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    data[idx] = selectedNeighbors[idx];
                }
            }

            // 依次尝试添加反向边（添加 cur_c 的每个邻居到 cur_c 的边）
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                // 这个作用域内当前访问的 cur_c 点的邻居 selectedNeighbors[idx] 将被锁住
                // 对第 idx 个邻居上锁，即 selectedNeighbors[idx]，这里简称为 idx
                std::unique_lock<std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

                // 获取 idx 的邻居数组的指针
                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(selectedNeighbors[idx]);
                else
                    ll_other = get_linklist(selectedNeighbors[idx], level);

                // 获取 idx 的邻居数
                size_t sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > Mcurmax) // idx 的现有邻居应该是没有超过最大出度限制的
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                // 获取 idx 的邻居列表
                tableint *data = (tableint *) (ll_other + 1);

                // 检查反向边是否已经存在（idx 到 cur_c 的边）
                bool is_cur_c_present = false;
                if (isUpdate) {
                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        if (data[j] == cur_c) {  // 反向边已经存在，则 is_cur_c_present 设为 true
                            is_cur_c_present = true;
                            break;
                        }
                    }
                }

                // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
                if (!is_cur_c_present) { // 如果反向边不存在，则尝试添加
                    if (sz_link_list_other < Mcurmax) { // idx 邻居数没有达到最大出度，则直接添加到 cur_c 的边
                        data[sz_link_list_other] = cur_c;
                        setListCount(ll_other, sz_link_list_other + 1);
                    } else {
                        // idx 邻居数已经达到最大出度，选择一个邻居替换为 cur_c
                        // finding the "weakest" element to replace it with the new one
                        dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c),
                                                    getDataByInternalId(selectedNeighbors[idx]),
                                                    dist_func_param_);

                        // 将 cur_c 和 idx 现有邻居都加入候选邻居集 candidates
                        // Heuristic:
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                        candidates.emplace(d_max, cur_c);

                        for (size_t j = 0; j < sz_link_list_other; j++) {
                            candidates.emplace(
                                    fstdistfunc_(getDataByInternalId(data[j]),
                                                 getDataByInternalId(selectedNeighbors[idx]),
                                                 dist_func_param_), data[j]);
                        }

                        // 启发式地对候选邻居集进行剪枝
                        getNeighborsByHeuristic2(candidates, Mcurmax);

                        // 更新 idx 的邻居
                        int indx = 0;
                        while (candidates.size() > 0) {
                            data[indx] = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }

                        setListCount(ll_other, indx);
                        // Nearest K:
                        /*int indx = -1;
                        for (int j = 0; j < sz_link_list_other; j++) {
                            dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                            if (d > d_max) {
                                indx = j;
                                d_max = d;
                            }
                        }
                        if (indx >= 0) {
                            data[indx] = cur_c;
                        } */
                    }
                }
            }

            return next_closest_entry_point;
        }


        void resizeIndex(size_t new_max_elements) {
            if (new_max_elements < cur_element_count)
                throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

            delete visited_list_pool_;
            visited_list_pool_ = new VisitedListPool(1, new_max_elements);

            element_levels_.resize(new_max_elements);

            std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

            // Reallocate base layer
            char *data_level0_memory_new = (char *) realloc(data_level0_memory_,
                                                            new_max_elements * size_data_per_element_);
            if (data_level0_memory_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
            data_level0_memory_ = data_level0_memory_new;

            // Reallocate all other layers
            char **linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
            if (linkLists_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
            linkLists_ = linkLists_new;

            max_elements_ = new_max_elements;
        }


        void saveIndex(const std::string &location) {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, offsetLevel0_);
            writeBinaryPOD(output, max_elements_);
            writeBinaryPOD(output, cur_element_count);
            writeBinaryPOD(output, size_data_per_element_);
            writeBinaryPOD(output, label_offset_);
            writeBinaryPOD(output, offsetData_);
            writeBinaryPOD(output, maxlevel_);
            writeBinaryPOD(output, enterpoint_node_);
            writeBinaryPOD(output, maxM_);

            writeBinaryPOD(output, maxM0_);
            writeBinaryPOD(output, M_);
            writeBinaryPOD(output, mult_);
            writeBinaryPOD(output, ef_construction_);

            output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

            for (size_t i = 0; i < cur_element_count; i++) {
                unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
                writeBinaryPOD(output, linkListSize);
                if (linkListSize)
                    output.write(linkLists_[i], linkListSize);
            }
            output.close();
        }


        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
            std::ifstream input(location, std::ios::binary);

            if (!input.is_open())
                throw std::runtime_error("Cannot open file");

            // get file size:
            input.seekg(0, input.end);
            std::streampos total_filesize = input.tellg();
            input.seekg(0, input.beg);

            readBinaryPOD(input, offsetLevel0_);
            readBinaryPOD(input, max_elements_);
            readBinaryPOD(input, cur_element_count);

            size_t max_elements = max_elements_i;
            if (max_elements < cur_element_count)
                max_elements = max_elements_;
            max_elements_ = max_elements;
            readBinaryPOD(input, size_data_per_element_);
            readBinaryPOD(input, label_offset_);
            readBinaryPOD(input, offsetData_);
            readBinaryPOD(input, maxlevel_);
            readBinaryPOD(input, enterpoint_node_);

            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, M_);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, ef_construction_);

            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();

            auto pos = input.tellg();

            /// Optional - check if index is ok:
            input.seekg(cur_element_count * size_data_per_element_, input.cur);
            for (size_t i = 0; i < cur_element_count; i++) {
                if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                    throw std::runtime_error("Index seems to be corrupted or unsupported");
                }

                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize != 0) {
                    input.seekg(linkListSize, input.cur);
                }
            }

            // throw exception if it either corrupted or old index
            if (input.tellg() != total_filesize)
                throw std::runtime_error("Index seems to be corrupted or unsupported");

            input.clear();
            /// Optional check end

            input.seekg(pos, input.beg);

            data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
            input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            std::vector<std::mutex>(max_elements).swap(link_list_locks_);
            std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
            element_levels_ = std::vector<int>(max_elements);
            revSize_ = 1.0 / mult_;
            ef_ = 10;
            for (size_t i = 0; i < cur_element_count; i++) {
                label_lookup_[getExternalLabel(i)] = i;
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize == 0) {
                    element_levels_[i] = 0;
                    linkLists_[i] = nullptr;
                } else {
                    element_levels_[i] = linkListSize / size_links_per_element_;
                    linkLists_[i] = (char *) malloc(linkListSize);
                    if (linkLists_[i] == nullptr)
                        throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                    input.read(linkLists_[i], linkListSize);
                }
            }

            for (size_t i = 0; i < cur_element_count; i++) {
                if (isMarkedDeleted(i)) {
                    num_deleted_ += 1;
                    if (allow_replace_deleted_) deleted_elements.insert(i);
                }
            }

            input.close();

            return;
        }

        /**
         * 获取 label 对应的 base 数据
         * @tparam data_t 数据类型，如 float
         * @return
         */
        template<typename data_t>
        std::vector<data_t> getDataByLabel(labeltype label) const {
            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            lock_table.unlock();

            char *data_ptrv = getDataByInternalId(internalId);
            size_t dim = *((size_t *) dist_func_param_);
            std::vector<data_t> data;
            data_t *data_ptr = (data_t *) data_ptrv;
            for (int i = 0; i < dim; i++) {
                data.push_back(*data_ptr);
                data_ptr += 1;
            }
            return data;
        }


        /*
        * Marks an element with the given label deleted, does NOT really change the current graph.
        */
        // 外部接口
        void markDelete(labeltype label) { // 仅标记该点删除，但不会改变图结构，只是搜索时不会将该点作为返回结果
            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label)); // 对这个 label 上锁

            // 通过 label_lookup_ 获取 label 对应的 internalId
            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            lock_table.unlock();

            markDeletedInternal(internalId); // 标记删除 label(此时拥有对这个 label 的锁）
        }


        /*
        * Uses the last 16 bits of the memory for the linked list size to store the mark,
        * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
        */
        // 内部实现，被 markDelete 调用
        void markDeletedInternal(tableint internalId) {
            assert(internalId < cur_element_count);
            if (!isMarkedDeleted(internalId)) { // 标记删除
                unsigned char *ll_cur = ((unsigned char *) get_linklist0(internalId)) + 2;
                *ll_cur |= DELETE_MARK; // 用 1 表示删除标记
                num_deleted_ += 1;
                if (allow_replace_deleted_) {
                    std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
                    deleted_elements.insert(internalId);
                }
            } else {
                throw std::runtime_error("The requested to delete element is already deleted");
            }
        }


        /*
        * Removes the deleted mark of the node, does NOT really change the current graph.
        *
        * Note: the method is not safe to use when replacement of deleted elements is enabled,
        *  because elements marked as deleted can be completely removed by addPoint
        */
        // 外部接口
        void unmarkDelete(labeltype label) { // 仅消除删除标记，但不会改变图结构
            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label)); // 对这个 label 上锁

            // 通过 label_lookup_ 获取 label 对应的 internalId
            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            lock_table.unlock();

            unmarkDeletedInternal(internalId);
        }


        /*
        * Remove the deleted mark of the node. 消除第 internalId 个元素的删除标记
        */
        // 内部实现，被 unmarkDelete 调用
        void unmarkDeletedInternal(tableint internalId) {
            assert(internalId < cur_element_count);
            if (isMarkedDeleted(internalId)) { // 消除删除标记
                unsigned char *ll_cur = ((unsigned char *) get_linklist0(internalId)) + 2;
                *ll_cur &= ~DELETE_MARK;
                num_deleted_ -= 1;
                if (allow_replace_deleted_) {
                    // 互斥更新 deleted_elements
                    std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
                    deleted_elements.erase(internalId);
                }
            } else {
                throw std::runtime_error("The requested to undelete element is not deleted");
            }
        }


        /*
        * Checks the first 16 bits of the memory to see if the element is marked deleted.
        */
        bool isMarkedDeleted(tableint internalId) const { // 检查第 internalId 个元素是否被标记删除
            unsigned char *ll_cur = ((unsigned char *) get_linklist0(internalId)) + 2;
            return *ll_cur & DELETE_MARK;
        }


        /**
         * 获取邻居数
         * @param ptr 指向一片连续地址
         * 第一个 4 字节中，头 2 字节表示邻居总数，后 2 字节表示删除标记
         * 之后每 4 字节表示一个邻居编号
         */
        unsigned short int getListCount(linklistsizeint *ptr) const {
            return *((unsigned short int *) ptr); // short int 类型占 2 字节
        }

        /**
         * 设置邻居数
         */
        void setListCount(linklistsizeint *ptr, unsigned short int size) const {
            *((unsigned short int *) (ptr)) = *((unsigned short int *) &size);
        }


        /*
        * Adds point. Updates the point if it is already in the index.
        * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
        */
        /// 外层 addPoint
        void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) {
            if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
                throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
            }

            // 获取对相应label进行操作的锁，不能同时操作同一个 label（细粒度锁）
            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));
            if (!replace_deleted) {
                /// replace_deleted 等于 false 的情况
                addPoint(data_point, label, -1);
                return;
            }
            /// replace_deleted 等于 true 的情况
            // check if there is vacant place
            tableint internal_id_replaced;
            std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock); // 获取删除操作锁（粗粒度锁）
            bool is_vacant_place = !deleted_elements.empty();
            if (is_vacant_place) { // 如果有被标记删除的空闲空间，则直接使用一个空闲空间进行更新
                internal_id_replaced = *deleted_elements.begin();
                deleted_elements.erase(internal_id_replaced);
            }
            lock_deleted_elements.unlock(); // 释放删除操作锁

            // if there is no vacant place then add or update point
            // else add point to vacant place
            if (!is_vacant_place) { // 如果没有被标记删除的空闲空间
                addPoint(data_point, label, -1);
            } else { // 如果有被标记删除的空闲空间，则直接使用一个空闲空间进行更新
                // we assume that there are no concurrent operations on deleted element
                // 更新空闲空间的 label
                labeltype label_replaced = getExternalLabel(internal_id_replaced);
                setExternalLabel(internal_id_replaced, label);

                // 更新外部 label 到 internal_id 的映射关系
                std::unique_lock<std::mutex> lock_table(label_lookup_lock);
                label_lookup_.erase(label_replaced);
                label_lookup_[label] = internal_id_replaced;
                lock_table.unlock();

                unmarkDeletedInternal(internal_id_replaced);
                updatePoint(data_point, internal_id_replaced, 1.0);
            }
        }

        /**
         * 对旧的数据点进行更新
         * 这个函数会先尝试删除反向边（旧点邻居到旧点的边，即旧点的入边）
         * 再通过 repairConnectionsForUpdate 函数重新更新正向边和反向边（相当于重新插入）
         * @param dataPoint 新 base 数据
         * @param internalId 更新位置
         * @param updateNeighborProbability 更新概率
         */
        void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
            // 更新 base 数据
            // update the feature vector associated with existing point with new vector
            memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

            int maxLevelCopy = maxlevel_;
            tableint entryPointCopy = enterpoint_node_;
            // If point to be updated is entry point and graph just contains single element then just return.
            if (entryPointCopy == internalId && cur_element_count == 1)
                return;

            // 依次在每层进行更新（层高是旧点插入时确定的）
            int elemLevel = element_levels_[internalId];
            std::uniform_real_distribution<float> distribution(0.0, 1.0);
            for (int layer = 0; layer <= elemLevel; layer++) {
                std::unordered_set<tableint> sCand; // 候选邻居集
                std::unordered_set<tableint> sNeigh; // 保存 internalId 的需要尝试删除反向边的邻居
                // 读取 internalId 的现有邻居
                std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
                if (listOneHop.size() == 0)
                    continue;

                sCand.insert(internalId);

                for (auto &&elOneHop: listOneHop) { // 对于每个现有邻居
                    // 把现有邻居 elOneHop 加入候选邻居集
                    sCand.insert(elOneHop);

                    // 以一定概率把现有邻居 elOneHop 加入 sNeigh，sNeigh 中是需要尝试添加反向边的 internalId 的邻居
                    if (distribution(update_probability_generator_) > updateNeighborProbability)
                        continue;
                    sNeigh.insert(elOneHop);

                    // 如果要尝试删除反向边（elOneHop 到 internalId 的边），先把 elOneHop 的现有邻居加入候选邻居集
                    std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                    for (auto &&elTwoHop: listTwoHop) {
                        sCand.insert(elTwoHop);
                    }
                }

                // sCand 中包括 internalId、所有一跳邻居 elOneHop、部分二跳邻居（选择尝试删除反向边的邻居的邻居）
                for (auto &&neigh: sNeigh) { // 尝试删除 neigh 到 internalId 的反向边
                    // if (neigh == internalId)
                    //     continue;

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() -
                                                                                    1;  // sCand guaranteed to have size >= 1
                    size_t elementsToKeep = std::min(ef_construction_, size);

                    // 取候选邻居集 sCand 中最近的 elementsToKeep 个元素到 candidates 作为本次循环的新候选集
                    for (auto &&cand: sCand) {
                        if (cand == neigh) // 不要把 neigh 自己加进去了
                            continue;

                        // 计算和 neigh 的距离
                        dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand),
                                                       dist_func_param_);
                        // 保留 candidates 中最近的 elementsToKeep 个
                        if (candidates.size() < elementsToKeep) {
                            candidates.emplace(distance, cand);
                        } else {
                            if (distance < candidates.top().first) {
                                candidates.pop();
                                candidates.emplace(distance, cand);
                            }
                        }
                    }

                    // 对 neigh 的候选邻居进行剪枝，neigh 到 internalId 的反向边可能会被删除
                    // Retrieve neighbours using heuristic and set connections.
                    getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                    // 更新 neigh 的邻居表
                    {
                        std::unique_lock<std::mutex> lock(link_list_locks_[neigh]);
                        linklistsizeint *ll_cur;
                        ll_cur = get_linklist_at_level(neigh, layer);
                        size_t candSize = candidates.size();
                        setListCount(ll_cur, candSize);
                        tableint *data = (tableint *) (ll_cur + 1);
                        for (size_t idx = 0; idx < candSize; idx++) {
                            data[idx] = candidates.top().second;
                            candidates.pop();
                        }
                    }
                }
            }

            // 对旧点进行更新，相当于重新插入
            repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
        }

        /**
         * 这个函数类似于内层 addPoint 函数，只是用于进行对旧点进行更新，相当于重新插入
         * @param dataPoint
         * @param entryPointInternalId
         * @param dataPointInternalId
         * @param dataPointLevel
         * @param maxLevel
         */
        void repairConnectionsForUpdate(
                const void *dataPoint,
                tableint entryPointInternalId,
                tableint dataPointInternalId,
                int dataPointLevel,
                int maxLevel) {
            tableint currObj = entryPointInternalId;

            // 从高到低依次在每层进行搜索，直到更新点所在层高
            if (dataPointLevel < maxLevel) {
                dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxLevel; level > dataPointLevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist_at_level(currObj, level);
                        int size = getListCount(data);
                        tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                        for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                            _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                            tableint cand = datal[i];
                            dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            if (dataPointLevel > maxLevel)
                throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

            // 再从高到低依次在每层进行搜索，然后根据搜索得到的候选邻居集，通过 mutuallyConnectNewElement 进行更新
            for (int level = dataPointLevel; level >= 0; level--) {
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                        currObj, dataPoint, level);

                // 由于旧点已经位于图中，因此 searchBaseLayer 会搜到要更新的旧点自身，对 topCandidates 过滤得到候选邻居集
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
                while (topCandidates.size() > 0) {
                    if (topCandidates.top().second != dataPointInternalId)
                        filteredTopCandidates.push(topCandidates.top());
                    topCandidates.pop();
                }

                // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
                // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
                if (filteredTopCandidates.size() > 0) {
                    bool epDeleted = isMarkedDeleted(entryPointInternalId);
                    if (epDeleted) {
                        filteredTopCandidates.emplace(
                                fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_),
                                entryPointInternalId);
                        if (filteredTopCandidates.size() > ef_construction_)
                            filteredTopCandidates.pop();
                    }

                    // 这里进行正向边和反向边的更新
                    currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level,
                                                        true);
                }
            }
        }

        /**
         * 获取 internalId 在第 level 层的邻居（只返回邻居编号）
         * @param internalId
         * @param level
         * @return
         */
        std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
            std::unique_lock<std::mutex> lock(link_list_locks_[internalId]);
            unsigned int *data = get_linklist_at_level(internalId, level);
            int size = getListCount(data);
            std::vector<tableint> result(size);
            tableint *ll = (tableint *) (data + 1);
            memcpy(result.data(), ll, size * sizeof(tableint));
            return result;
        }


        /**
         * 插入数据
         * @param data_point 一条 base 数据
         * @param label 数据的 label
         * @param level 数据的层高，如果等于-1，则由 getRandomLevel 确定层高
         * @return 所插入数据的 Internal_Id
         */
        /// 内层 addPoint
        tableint addPoint(const void *data_point, labeltype label, int level) {
            tableint cur_c = 0; // cur_c 是新插入元素的内部id，先初始化为 0
            {
                // 注意! lock_table 的作用域只在这个大括号内，因此超过这个范围 label_lookup_lock 将被释放
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.
                // 获取 label 对应的 Internal_Id
                std::unique_lock<std::mutex> lock_table(label_lookup_lock);
                auto search = label_lookup_.find(label);
                if (search != label_lookup_.end()) { // 如果 label 已经存在
                    tableint existingInternalId = search->second;
                    if (allow_replace_deleted_) {
                        if (isMarkedDeleted(existingInternalId)) {
                            throw std::runtime_error(
                                    "Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                        }
                    }
                    lock_table.unlock();

                    if (isMarkedDeleted(existingInternalId)) {
                        unmarkDeletedInternal(existingInternalId);
                    }

                    // 覆盖更新此 label 的元素，而不创建新元素
                    updatePoint(data_point, existingInternalId, 1.0);

                    return existingInternalId;
                }

                if (cur_element_count >= max_elements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                }
                /// label_lookup_ 的锁还没有被释放
                // 如果 label 不存在，则在末尾创建新元素
                cur_c = cur_element_count; // cur_c 设为 cur_element_count，即 data_level0_memory_ 的末尾
                cur_element_count++;
                label_lookup_[label] = cur_c;
            }
            // 超过 lock_table 作用域，此时 label_lookup_lock 将被释放

            std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]); // 锁 cur_c 的

            // 确定新元素的层高
            int curlevel = getRandomLevel(mult_);
            if (level > 0)
                curlevel = level;
            element_levels_[cur_c] = curlevel;

            std::unique_lock<std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();// 新元素的层高小于当前最大层高，说明入口点将保持不变，释放 global 锁

            // 如果插入点的层高高于当前最大层高，说明这个插入点将被更新为新入口点，那么这个插入操作完成前其他插入操作将不被允许，不能释放 global 锁
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;

            // 把 data_level0_memory_ 中 cur_c 对应位置初始化为0
            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

            // 在对应位置插入base数据和label
            // Initialisation of the data and label
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);

            if (curlevel) { // 如果层高大于 0，则要在 linkLists_[cur_c] 分配空间保存0层以上的邻居关系
                linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
                if (linkLists_[cur_c] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }

            if ((signed) currObj != -1) { // 进行插入操作， currObj(enterpoint_node_) 是最高层的入口点
                if (curlevel < maxlevelcopy) { // 如果插入点层高小于最大层高
                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                    for (int level = maxlevelcopy; level > curlevel; level--) { // 则从高到低依次在每层进行搜索，直到插入点所在层高
                        bool changed = true;
                        while (changed) { // 把插入点当作 query，在 level 层进行最近邻搜索，直到最近邻不再改变
                            changed = false;
                            unsigned int *data;
                            std::unique_lock<std::mutex> lock(link_list_locks_[currObj]); // 锁 linkLists_[currObj]
                            data = get_linklist(currObj, level); // 读 linkLists_[currObj] 在对应 level 的邻居
                            int size = getListCount(data); // 获取邻居数

                            tableint *datal = (tableint *) (data + 1);
                            for (int i = 0; i < size; i++) { // 分别访问每个邻居
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
                                dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                if (d < curdist) { // 如果邻居的距离更小，当前最近邻转移到距离 query 最近的邻居点
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                        // 搜完一层后，把当前层的搜索结果当作下一层的入口点，即 currObj
                    }
                }

                // 继续从高到低依次在每层进行搜索，直到搜完所有层
                bool epDeleted = isMarkedDeleted(enterpoint_copy); ///
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                    if (level > maxlevelcopy || level < 0)  // possible?
                        throw std::runtime_error("Level error");

                    // 在 level 层进行搜索
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                            currObj, data_point, level);
                    if (epDeleted) { // 即使 enterpoint 被删除也在构图上也考虑，如果 enterpoint 足够进，这样可以加快搜索速度，因为搜索是从 enterpoint 开始的
                        top_candidates.emplace(
                                fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_),
                                enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }
                    // 根据当前层的搜索结果对当前层进行连边
                    currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
                    // 处理完一层后，把当前层的搜索结果当作下一层的入口点，即 currObj
                }
            } else {
                // 如果 currObj(enterpoint_node_) 等于 -1，说明这是第一个被插入的点，只需要将此点设为新入口点并更新最大层高
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;
            }

            // Releasing lock for the maximum level
            if (curlevel > maxlevelcopy) {
                // 如果插入点的层高大于当前最大层高，将此插入点设为入口点并更新最大层高，始终保存入口点在最高层
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            return cur_c;
        }

        /**
         * Top-K 查询
         * @param isIdAllowed 是否进行 Filter
         * 注意：查询时只有读操作，因此没有使用锁
         */
        std::priority_queue<std::pair<dist_t, labeltype >>
        searchKnn(const void *query_data, size_t k, BaseFilterFunctor *isIdAllowed = nullptr) const {
            std::priority_queue<std::pair<dist_t, labeltype >> result;
            if (cur_element_count == 0) return result;

            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            // 从高到低依次在每层进行搜索，每层的搜索结果作为下一层的入口点，直到第0层
            for (int level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations += size;

                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            // 在第0层进行搜索
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            if (num_deleted_) {
                top_candidates = searchBaseLayerST<true, true>(
                        currObj, query_data, std::max(ef_, k), isIdAllowed);
            } else {
                top_candidates = searchBaseLayerST<false, true>(
                        currObj, query_data, std::max(ef_, k), isIdAllowed);
            }

            // 保留 k 个结果
            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            // 返回结果
            while (top_candidates.size() > 0) {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }
            return result;
        }

        /**
         * 对图结构进行一些检查和统计
         */
        void checkIntegrity() {
            int connections_checked = 0;
            std::vector<int> inbound_connections_num(cur_element_count, 0);
            for (int i = 0; i < cur_element_count; i++) {
                for (int l = 0; l <= element_levels_[i]; l++) {
                    linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                    int size = getListCount(ll_cur);
                    tableint *data = (tableint *) (ll_cur + 1);
                    std::unordered_set<tableint> s;
                    for (int j = 0; j < size; j++) {
                        assert(data[j] > 0); // 邻居内部id应大于 0
                        assert(data[j] < cur_element_count); // 邻居内部id应小于 cur_element_count
                        assert(data[j] != i); // 自己不能是自己的邻居
                        inbound_connections_num[data[j]]++;
                        s.insert(data[j]);
                        connections_checked++;
                    }
                    assert(s.size() == size);
                }
            }
            // 统计最大和最小入度
            if (cur_element_count > 1) {
                int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
                for (int i = 0; i < cur_element_count; i++) {
                    assert(inbound_connections_num[i] > 0);
                    min1 = std::min(inbound_connections_num[i], min1);
                    max1 = std::max(inbound_connections_num[i], max1);
                }
                std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
            }
            std::cout << "integrity ok, checked " << connections_checked << " connections\n";
        }
    };
}  // namespace hnswlib
