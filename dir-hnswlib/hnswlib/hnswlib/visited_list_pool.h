#pragma once

#include <mutex>
#include <string.h>
#include <deque>

namespace hnswlib {
    typedef unsigned short int vl_type;

    class VisitedList {
    public:
        vl_type curV; // 访问状态标志，如果某个点被访问，则将 mass 对应位置设为 curV
        vl_type *mass; // 记录访问状态的数组
        unsigned int numelements; // 能容纳的元素数量

        VisitedList(int numelements1) {
            curV = -1;
            numelements = numelements1;
            mass = new vl_type[numelements];
        }

        /**
         * 在 VisitedListPool 中，VisitedList 是可以复用的
         * 因此 mass 数组中可能是有数据的，curV++ 相当于产生新的访问标志
         * 那么即使 mass 中存在数据，但是不等于新的 curV，相当于都还没有被访问，达到初始化 VisitedList 一样的效果
         */
        void reset() {
            curV++;
            if (curV == 0) {
                memset(mass, 0, sizeof(vl_type) * numelements);
                curV++;
            }
        }

        ~VisitedList() { delete[] mass; }
    };
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

    class VisitedListPool {
        std::deque<VisitedList *> pool; // 空闲 VisitedList 池
        std::mutex poolguard; // 用于互斥访问 pool，保证线程安全
        int numelements; // 一个 VisitedList 能容纳的元素数量

    public:
        VisitedListPool(int initmaxpools, int numelements1) {
            numelements = numelements1;
            for (int i = 0; i < initmaxpools; i++)
                pool.push_front(new VisitedList(numelements));
        }

        /**
         * 获取 VisitedList
         */
        VisitedList *getFreeVisitedList() {
            VisitedList *rez;
            {
                std::unique_lock<std::mutex> lock(poolguard);
                if (pool.size() > 0) { // 如果池中还有空闲 VisitedList，则从池中获取
                    rez = pool.front();
                    pool.pop_front();
                } else { // 如果池中没有空闲 VisitedList，则直接 new 一个
                    rez = new VisitedList(numelements);
                }
            }
            rez->reset();
            return rez;
        }

        /**
         * 释放 VisitedList，放回空闲 VisitedList 池中
         */
        void releaseVisitedList(VisitedList *vl) {
            std::unique_lock<std::mutex> lock(poolguard);
            pool.push_front(vl);
        }

        ~VisitedListPool() {
            while (pool.size()) {
                VisitedList *rez = pool.front();
                pool.pop_front();
                delete rez;
            }
        }
    };
}  // namespace hnswlib
