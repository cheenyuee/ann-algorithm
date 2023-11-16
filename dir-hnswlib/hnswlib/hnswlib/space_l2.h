#pragma once

#include "hnswlib.h"

namespace hnswlib {

    /**
     * 最基本的计算L2平方距离的函数
     * @param pVect1v
     * @param pVect2v
     * @param qty_ptr 数据维度
     * @return
     */
    static float
    L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

        float res = 0;
        for (size_t i = 0; i < qty; i++) {
            float t = *pVect1 - *pVect2;
            pVect1++;
            pVect2++;
            res += t * t;
        }
        return (res);
    }

#if defined(USE_AVX512)
    /**
     * 使用 AVX512 计算距离，要求数据维度qty是16倍数，一次计算16个维度，
     */
    // Favor using AVX512 if available.
    static float
    L2SqrSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);
        float PORTABLE_ALIGN64 TmpRes[16];
        size_t qty16 = qty >> 4;

        const float *pEnd1 = pVect1 + (qty16 << 4);

        __m512 diff, v1, v2;
        __m512 sum = _mm512_set1_ps(0);

        while (pVect1 < pEnd1) { // 一次计算16个维度，是并行进行的
            v1 = _mm512_loadu_ps(pVect1);
            pVect1 += 16;
            v2 = _mm512_loadu_ps(pVect2);
            pVect2 += 16;
            diff = _mm512_sub_ps(v1, v2);
            // sum = _mm512_fmadd_ps(diff, diff, sum);
            sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
        }

        // 把向量计算结果保存到内存中
        _mm512_store_ps(TmpRes, sum);

        // sum 是16维的，还需要汇总加起来才是最终的距离
        float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
                TmpRes[13] + TmpRes[14] + TmpRes[15];

        return (res);
    }
#endif

#if defined(USE_AVX)

    /**
     * 使用 AVX 计算距离，一次计算 8 个维度，要求数据维度qty是16倍数，因为一次循环计算了两次
     */
    // Favor using AVX if available.
    static float
    L2SqrSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);
        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty16 = qty >> 4;

        const float *pEnd1 = pVect1 + (qty16 << 4);

        __m256 diff, v1, v2;
        __m256 sum = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }

        _mm256_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
    }

#endif

#if defined(USE_SSE)

    /**
     * 使用 SSE 计算距离，一次计算 4 个维度，要求数据维度qty是16倍数，因为一次循环计算了四次
     */
    static float
    L2SqrSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);
        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty16 = qty >> 4;

        const float *pEnd1 = pVect1 + (qty16 << 4);

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }

        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }

#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    static DISTFUNC<float> L2SqrSIMD16Ext = L2SqrSIMD16ExtSSE;

    /**
     * 将16倍数的维度用 SSE 计算，剩余维度用普通函数计算
     */
    static float
    L2SqrSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty16 = qty >> 4 << 4;
        float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
        float *pVect1 = (float *) pVect1v + qty16;
        float *pVect2 = (float *) pVect2v + qty16;

        size_t qty_left = qty - qty16;
        float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
        return (res + res_tail);
    }

#endif


#if defined(USE_SSE)

    /**
     * 使用 SSE 计算距离，一次计算 4 个维度，要求数据维度qty是4的倍数
     */
    static float
    L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[8];
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);


        size_t qty4 = qty >> 2;

        const float *pEnd1 = pVect1 + (qty4 << 2);

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }

    /**
     * 将4倍数的维度用 SSE 计算，剩余维度用普通函数计算
     */
    static float
    L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty4 = qty >> 2 << 2;

        float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4);
        size_t qty_left = qty - qty4;

        float *pVect1 = (float *) pVect1v + qty4;
        float *pVect2 = (float *) pVect2v + qty4;
        float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

        return (res + res_tail);
    }

#endif

    /**
     * L2 度量空间
     */
    class L2Space : public SpaceInterface<float> {
        DISTFUNC<float> fstdistfunc_;
        size_t data_size_; // 一条 base 数据的长度（字节数），等于 dim * sizeof(float);
        size_t dim_; // 一条 base 数据的维度

    public:
        L2Space(size_t dim) {
            fstdistfunc_ = L2Sqr; // 默认用普通函数进行距离计算
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
            // 如果支持指令集加速
            // 默认函数指针 L2SqrSIMD16Ext 和 L2SqrSIMD4Ext 都是用的 SSE
            // 但是如果 AVX512 或 AVX 可用，会替换为 AVX512 或 AVX 的实现
#if defined(USE_AVX512)
            if (AVX512Capable())
                L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX512;
            else if (AVXCapable())
                L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
#elif defined(USE_AVX)
            if (AVXCapable())
                L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
#endif
            // 根据 dim 选用不同的函数进行处理
            if (dim % 16 == 0)
                fstdistfunc_ = L2SqrSIMD16Ext;
            else if (dim % 4 == 0)
                fstdistfunc_ = L2SqrSIMD4Ext;
            else if (dim > 16)
                fstdistfunc_ = L2SqrSIMD16ExtResiduals;
            else if (dim > 4)
                fstdistfunc_ = L2SqrSIMD4ExtResiduals;
#endif
            dim_ = dim;
            data_size_ = dim * sizeof(float);
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &dim_;
        }

        ~L2Space() {}
    };

    static int
    L2SqrI4x(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        int res = 0;
        unsigned char *a = (unsigned char *) pVect1;
        unsigned char *b = (unsigned char *) pVect2;

        qty = qty >> 2;
        for (size_t i = 0; i < qty; i++) {
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
        }
        return (res);
    }

    /**
     * __restrict 是 GCC 编译器的关键字，用于告诉编译器该指针是唯一访问其指向内存的路径
     */
    static int L2SqrI(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        int res = 0;
        unsigned char *a = (unsigned char *) pVect1;
        unsigned char *b = (unsigned char *) pVect2;

        for (size_t i = 0; i < qty; i++) {
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
        }
        return (res);
    }

    class L2SpaceI : public SpaceInterface<int> {
        DISTFUNC<int> fstdistfunc_;
        size_t data_size_;
        size_t dim_;

    public:
        L2SpaceI(size_t dim) {
            if (dim % 4 == 0) {
                fstdistfunc_ = L2SqrI4x;
            } else {
                fstdistfunc_ = L2SqrI;
            }
            dim_ = dim;
            data_size_ = dim * sizeof(unsigned char);
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<int> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &dim_;
        }

        ~L2SpaceI() {}
    };
}  // namespace hnswlib
