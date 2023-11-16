#pragma once
#ifndef NO_MANUAL_VECTORIZATION
#if (defined(__SSE__) || _M_IX86_FP > 0 || defined(_M_AMD64) || defined(_M_X64))
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#ifdef __AVX512F__
#define USE_AVX512
#endif
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
void cpuid(int32_t out[4], int32_t eax, int32_t ecx) {
    __cpuidex(out, eax, ecx);
}
static __int64 xgetbv(unsigned int x) {
    return _xgetbv(x);
}
#else

#include <x86intrin.h>
#include <cpuid.h>
#include <stdint.h>

/**
 * __cpuid_count 是一个内建函数，用于获取CPU的特定信息，eax和ecx用于指定要查询的功能
 */
static void cpuid(int32_t cpuInfo[4], int32_t eax, int32_t ecx) {
    __cpuid_count(eax, ecx, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
}

static uint64_t xgetbv(unsigned int index) {
    uint32_t eax, edx;
    // __asm__ 是GCC中的关键字，用于在c/c++代码中插入汇编代码
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return ((uint64_t) edx << 32) | eax;
}

#endif

#if defined(USE_AVX512)
#include <immintrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif

// Adapted from https://github.com/Mysticial/FeatureDetector
#define _XCR_XFEATURE_ENABLED_MASK  0

/**
 * 查询 CPU 信息，判断 AVX 是否可用
 */
static bool AVXCapable() {
    int cpuInfo[4];

    // CPU support
    cpuid(cpuInfo, 0, 0);
    int nIds = cpuInfo[0];

    bool HW_AVX = false;
    if (nIds >= 0x00000001) {
        cpuid(cpuInfo, 0x00000001, 0);
        HW_AVX = (cpuInfo[2] & ((int) 1 << 28)) != 0;
    }

    // OS support
    cpuid(cpuInfo, 1, 0);

    bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
    bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

    bool avxSupported = false;
    if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        avxSupported = (xcrFeatureMask & 0x6) == 0x6;
    }
    return HW_AVX && avxSupported;
}

static bool AVX512Capable() {
    if (!AVXCapable()) return false;

    int cpuInfo[4];

    // CPU support
    cpuid(cpuInfo, 0, 0);
    int nIds = cpuInfo[0];

    bool HW_AVX512F = false;
    if (nIds >= 0x00000007) {  //  AVX512 Foundation
        cpuid(cpuInfo, 0x00000007, 0);
        HW_AVX512F = (cpuInfo[1] & ((int) 1 << 16)) != 0;
    }

    // OS support
    cpuid(cpuInfo, 1, 0);

    bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
    bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

    bool avx512Supported = false;
    if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        avx512Supported = (xcrFeatureMask & 0xe6) == 0xe6;
    }
    return HW_AVX512F && avx512Supported;
}

#endif

#include <queue>
#include <vector>
#include <iostream>
#include <string.h>

namespace hnswlib {
    typedef size_t labeltype;

    // This can be extended to store state for filtering (e.g. from a std::set)
    /**
     * 用于实现 filter 的基类(labeltype 是在 addPoint 时输入的)，重载()操作符，使其可以像函数一样使用，用于判断 label 是否满足条件
     */
    class BaseFilterFunctor {
    public:
        virtual bool operator()(hnswlib::labeltype id) { return true; }
    };

    template<typename T>
    class pairGreater {
    public:
        bool operator()(const T &p1, const T &p2) {
            return p1.first > p2.first;
        }
    };

    /**
     * 写入二进制数据 podRef
     */
    template<typename T>
    static void writeBinaryPOD(std::ostream &out, const T &podRef) {
        out.write((char *) &podRef, sizeof(T));
    }

    /**
     * 读出二进制数据 podRef
     */
    template<typename T>
    static void readBinaryPOD(std::istream &in, T &podRef) {
        in.read((char *) &podRef, sizeof(T));
    }

    /**
     * 定义别名为 DISTFUNC 的函数指针类型
     */
    template<typename MTYPE>
    using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);

    /**
     * 空间度量实现接口
     * @tparam MTYPE 通常为 float，如 L2Space
     */
    template<typename MTYPE>
    class SpaceInterface {
    public:
        // virtual void search(void *);
        virtual size_t get_data_size() = 0;

        virtual DISTFUNC<MTYPE> get_dist_func() = 0;

        virtual void *get_dist_func_param() = 0;

        virtual ~SpaceInterface() {}
    };

    /**
     * 算法实现接口，包括插入和搜索
     */
    template<typename dist_t>
    class AlgorithmInterface {
    public:
        virtual void addPoint(const void *datapoint, labeltype label, bool replace_deleted = false) = 0;

        virtual std::priority_queue<std::pair<dist_t, labeltype>>
        searchKnn(const void *, size_t, BaseFilterFunctor *isIdAllowed = nullptr) const = 0;

        // Return k nearest neighbor in the order of closer first
        virtual std::vector<std::pair<dist_t, labeltype>>
        searchKnnCloserFirst(const void *query_data, size_t k, BaseFilterFunctor *isIdAllowed = nullptr) const;

        virtual void saveIndex(const std::string &location) = 0;

        virtual ~AlgorithmInterface() {
        }
    };

    /**
     * searchKnnCloserFirst 通过调用 searchKnn 来实现，返回 vector
     * searchKnn 返回 priority_queue
     */
    template<typename dist_t>
    std::vector<std::pair<dist_t, labeltype>>
    AlgorithmInterface<dist_t>::searchKnnCloserFirst(const void *query_data, size_t k,
                                                     BaseFilterFunctor *isIdAllowed) const {
        std::vector<std::pair<dist_t, labeltype>> result;

        // here searchKnn returns the result in the order of further first
        auto ret = searchKnn(query_data, k, isIdAllowed);
        {
            size_t sz = ret.size();
            result.resize(sz);
            while (!ret.empty()) {
                result[--sz] = ret.top();
                ret.pop();
            }
        }

        return result;
    }
}  // namespace hnswlib

#include "space_l2.h"
#include "space_ip.h"
#include "bruteforce.h"
#include "hnswalg.h"
