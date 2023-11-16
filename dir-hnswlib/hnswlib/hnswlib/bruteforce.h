#pragma once

#include <unordered_map>
#include <fstream>
#include <mutex>
#include <algorithm>
#include <assert.h>

namespace hnswlib {
    /**
     * 实现 AlgorithmInterface 接口
     */
    template<typename dist_t>
    class BruteforceSearch : public AlgorithmInterface<dist_t> {
    public:
        char *data_;
        size_t maxelements_; // 最大元素数
        size_t cur_element_count; // 当前元素数
        size_t size_per_element_; // 每个元素的长度（数据长度加标签长度）

        size_t data_size_; // 一条 base 数据的长度（字节数），等于 dim * sizeof(float);
        DISTFUNC<dist_t> fstdistfunc_; // 距离函数指针
        void *dist_func_param_; // 距离函数参数，其实是数据维度的指针
        std::mutex index_lock;

        std::unordered_map<labeltype, size_t> dict_external_to_internal;

        /**
         * 构造函数
         */
        BruteforceSearch(SpaceInterface<dist_t> *s)
                : data_(nullptr),
                  maxelements_(0),
                  cur_element_count(0),
                  size_per_element_(0),
                  data_size_(0),
                  dist_func_param_(nullptr) {
        }

        /**
         * 构造函数
         */
        BruteforceSearch(SpaceInterface<dist_t> *s, const std::string &location)
                : data_(nullptr),
                  maxelements_(0),
                  cur_element_count(0),
                  size_per_element_(0),
                  data_size_(0),
                  dist_func_param_(nullptr) {
            loadIndex(location, s);
        }

        /**
         * 构造函数
         */
        BruteforceSearch(SpaceInterface<dist_t> *s, size_t maxElements) {
            maxelements_ = maxElements; // 最大元素数
            data_size_ = s->get_data_size(); // 获取一条 base 数据的长度（字节数），等于 dim * sizeof(float);
            fstdistfunc_ = s->get_dist_func(); // 获取距离函数指针
            dist_func_param_ = s->get_dist_func_param(); // 获取
            size_per_element_ = data_size_ + sizeof(labeltype); // 一个元素的长度等于数据长度加标签长度
            data_ = (char *) malloc(maxElements * size_per_element_); // 按最大元素数分配空间
            if (data_ == nullptr)
                throw std::runtime_error("Not enough memory: BruteforceSearch failed to allocate data");
            cur_element_count = 0;
        }


        ~BruteforceSearch() {
            free(data_);
        }

        /**
         *  插入数据
         * @param datapoint 一条base数据
         * @param label 数据的标签（通常应该为数据在数据集中的编号）
         * @param replace_deleted
         * dict_external_to_internal 用于保存外部label的信息和内部data_中的位置信息的映射
         */
        void addPoint(const void *datapoint, labeltype label, bool replace_deleted = false) {
            int idx;
            {
                // 互斥访问临界资源，不能同时插入数据
                std::unique_lock<std::mutex> lock(index_lock);

                // 看起来 label 不能重复，对于相同label的数据，后插入的数据会覆盖先插入的数据
                auto search = dict_external_to_internal.find(label);
                if (search != dict_external_to_internal.end()) {
                    idx = search->second;
                } else {
                    if (cur_element_count >= maxelements_) {
                        throw std::runtime_error("The number of elements exceeds the specified limit\n");
                    }
                    idx = cur_element_count;
                    dict_external_to_internal[label] = idx;
                    cur_element_count++;
                }
            }
            // 将元素添加到 data_ 中第 idx 个位置
            memcpy(data_ + size_per_element_ * idx + data_size_, &label, sizeof(labeltype));
            memcpy(data_ + size_per_element_ * idx, datapoint, data_size_);
        }

        /**
         * 删除label为cur_external的元素
         * @param cur_external
         */
        void removePoint(labeltype cur_external) {
            // 获取删除元素的存储位置
            size_t cur_c = dict_external_to_internal[cur_external];

            dict_external_to_internal.erase(cur_external);
            // 把 data_ 末尾元素移动到元素被删除后的空白位置，并修改映射关系
            labeltype label = *((labeltype *) (data_ + size_per_element_ * (cur_element_count - 1) + data_size_));
            dict_external_to_internal[label] = cur_c;
            memcpy(data_ + size_per_element_ * cur_c,
                   data_ + size_per_element_ * (cur_element_count - 1),
                   data_size_ + sizeof(labeltype));
            cur_element_count--;
        }

        /**
         * 暴力遍历，返回相应向量的 label 和距离信息
         */
        std::priority_queue<std::pair<dist_t, labeltype >>
        searchKnn(const void *query_data, size_t k, BaseFilterFunctor *isIdAllowed = nullptr) const {
            assert(k <= cur_element_count);
            std::priority_queue<std::pair<dist_t, labeltype >> topResults; // 大顶堆
            if (cur_element_count == 0) return topResults;
            for (int i = 0; i < k; i++) {
                // 获取距离和label
                dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
                labeltype label = *((labeltype *) (data_ + size_per_element_ * i + data_size_));
                if ((!isIdAllowed) || (*isIdAllowed)(label)) { // 有 filter 时 label 需要满足条件才加入队列
                    topResults.push(std::pair<dist_t, labeltype>(dist, label));
                }
            }
            dist_t lastdist = topResults.empty() ? std::numeric_limits<dist_t>::max() : topResults.top().first;
            for (int i = k; i < cur_element_count; i++) {
                dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
                if (dist <= lastdist) { // lastdist 是堆顶的当前最大距离，小于 lastdist 才进行下一步处理
                    labeltype label = *((labeltype *) (data_ + size_per_element_ * i + data_size_));
                    // 更新 topResults 和 lastdist
                    if ((!isIdAllowed) || (*isIdAllowed)(label)) {
                        topResults.push(std::pair<dist_t, labeltype>(dist, label));
                    }
                    if (topResults.size() > k)
                        topResults.pop();

                    if (!topResults.empty()) {
                        lastdist = topResults.top().first;
                    }
                }
            }
            return topResults;
        }

        /**
         * 按顺序写入文件
         */
        void saveIndex(const std::string &location) {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, maxelements_);
            writeBinaryPOD(output, size_per_element_);
            writeBinaryPOD(output, cur_element_count);

            output.write(data_, maxelements_ * size_per_element_);

            output.close();
        }

        /**
         * 按顺序读出文件，并进行初始化
         */
        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s) {
            std::ifstream input(location, std::ios::binary);
            std::streampos position;

            readBinaryPOD(input, maxelements_);
            readBinaryPOD(input, size_per_element_);
            readBinaryPOD(input, cur_element_count);

            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            size_per_element_ = data_size_ + sizeof(labeltype);
            data_ = (char *) malloc(maxelements_ * size_per_element_);
            if (data_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate data");

            input.read(data_, maxelements_ * size_per_element_);

            input.close();
        }
    };
}  // namespace hnswlib
