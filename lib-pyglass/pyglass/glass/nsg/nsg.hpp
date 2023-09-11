#pragma once

#include <atomic>
#include <random>
#include <stack>

#include "glass/builder.hpp"
#include "glass/graph.hpp"
#include "glass/neighbor.hpp"
#include "glass/utils.hpp"
#include "nndescent.hpp"

namespace glass {

struct NSG : public Builder {
  int d; // 维度
  std::string metric;
  int R; // 最大出度
  int L; // 候选队列大小
  int C; // C = R + 100
  int nb; // point 个数
  float *data;
  int ep; // entry point
  Graph<int> final_graph;
  RandomGenerator rng; ///< random generator
  Dist<float, float, float> dist_func;
  int GK;
  int nndescent_S;
  int nndescent_R;
  int nndescent_L;
  int nndescent_iter;

  explicit NSG(int dim, const std::string &metric, int R = 32, int L = 200)
      : d(dim), metric(metric), R(R), L(L), rng(0x0903) {
    this->C = R + 100;
    srand(0x1998);
    if (metric == "L2") {
      dist_func = L2SqrRef;
    } else if (metric == "IP") {
      dist_func = IPRef;
    }
    this->GK = 64;
    this->nndescent_S = 10;
    this->nndescent_R = 100;
    this->nndescent_L = this->GK + 50;
    this->nndescent_iter = 10;
  }

  void Build(float *data, int n) override {
    this->nb = n;
    this->data = data;
    NNDescent nnd(d, metric); // 使用 nn-descent 算法构建 kNN 图
    nnd.S = nndescent_S;
    nnd.R = nndescent_R;
    nnd.L = nndescent_L;
    nnd.iters = nndescent_iter;
    nnd.Build(data, n, GK);
    const auto &knng = nnd.final_graph;
    Init(knng); // 确定 NSG 的导航点

    std::vector<int> degrees(n, 0);
    {
        // 赋边
      Graph<Node> tmp_graph(n, R);
      link(knng, tmp_graph);
      // 将 tmp_graph 整理填入 final_graph
      final_graph.init(n, R);
      std::fill_n(final_graph.data, n * R, EMPTY_ID);
      final_graph.eps = {ep};
#pragma omp parallel for
      for (int i = 0; i < n; i++) { // 计算每个点的出度，整理 final_graph
        int cnt = 0;
        for (int j = 0; j < R; j++) {
          int id = tmp_graph.at(i, j).id;
          if (id != EMPTY_ID) {
            final_graph.at(i, cnt) = id;
            cnt += 1;
          }
          degrees[i] = cnt;
        }
      }
    }
    // 形成DFS树，进行连通
    [[maybe_unused]] int num_attached = tree_grow(degrees);
    // 统计 point 的 最大、最小、平均 出度
    int max = 0, min = 1e6;
    double avg = 0;
    for (int i = 0; i < n; i++) {
      int size = 0;
      while (size < R && final_graph.at(i, size) != EMPTY_ID) {
        size += 1;
      }
      max = std::max(size, max);
      min = std::min(size, min);
      avg += size;
    }
    avg = avg / n;
    printf("Degree Statistics: Max = %d, Min = %d, Avg = %lf\n", max, min, avg);
  }

  Graph<int> GetGraph() override { return final_graph; }

  /**
   *  Init：确定 NSG 的 enterpoint（导航点）
   */
  void Init(const Graph<int> &knng) {
      // 计算 centroid
    std::vector<float> center(d);
    for (int i = 0; i < d; ++i) {
      center[i] = 0.0;
    }
    for (int i = 0; i < nb; i++) {
      for (int j = 0; j < d; j++) {
        center[j] += data[i * d + j];
      }
    }
    for (int i = 0; i < d; i++) {
      center[i] /= nb;
    }
    // 确定 medoid
    int ep_init = rng.rand_int(nb);
    std::vector<Neighbor> retset;
    std::vector<Node> tmpset;
    std::vector<bool> vis(nb);
    search_on_graph<false>(center.data(), knng, vis, ep_init, L, retset,
                           tmpset);
    // set enterpoint
    this->ep = retset[0].id;
  }

  /**
   * @param retset 搜索结果
   * @param fullset 搜索过程中的所有访问点
   */
  template <bool collect_fullset>
  void search_on_graph(const float *q, const Graph<int> &graph,
                       std::vector<bool> &vis, int ep, int pool_size,
                       std::vector<Neighbor> &retset,
                       std::vector<Node> &fullset) const {
    RandomGenerator gen(0x1234);
    retset.resize(pool_size + 1);
    // 初始化候选队列
    // 将 entry point 的邻居加入候选队列
    std::vector<int> init_ids(pool_size);
    int num_ids = 0;
    for (int i = 0; i < (int)init_ids.size() && i < graph.K; i++) {
      int id = (int)graph.at(ep, i); // entry point 的邻居 id
      if (id < 0 || id >= nb) {
        continue;
      }
      init_ids[i] = id;
      vis[id] = true;
      num_ids += 1;
    }
    while (num_ids < pool_size) { // 使随机补充候选队列达到 pool_size
      int id = gen.rand_int(nb);
      if (vis[id]) {
        continue;
      }
      init_ids[num_ids] = id;
      num_ids++;
      vis[id] = true;
    }
    for (int i = 0; i < (int)init_ids.size(); i++) { // 计算距离并排序
      int id = init_ids[i];
      float dist = dist_func(q, data + id * d, d);
      retset[i] = Neighbor(id, dist, true); // true 表示
      if (collect_fullset) {
        fullset.emplace_back(retset[i].id, retset[i].distance);
      }
    }
    std::sort(retset.begin(), retset.begin() + pool_size);
    // 开始搜索
    int k = 0; // the index of the first unchecked node
    while (k < pool_size) {
      int updated_pos = pool_size;
      if (retset[k].flag) { // 可以check，访问 retset[k] 的邻居
        retset[k].flag = false;
        int n = retset[k].id;
        for (int m = 0; m < graph.K; m++) { // 对于 n 的每个邻居
          int id = (int)graph.at(n, m);
          if (id < 0 || id > nb || vis[id]) {
            continue;
          }
          vis[id] = true; // 访问 id
          float dist = dist_func(q, data + id * d, d);
          Neighbor nn(id, dist, true);
          if (collect_fullset) {
            fullset.emplace_back(id, dist);
          }
          if (dist >= retset[pool_size - 1].distance) {
            continue;
          }
          int r = insert_into_pool(retset.data(), pool_size, nn);
          updated_pos = std::min(updated_pos, r);
        }
      }
      k = (updated_pos <= k) ? updated_pos : (k + 1); // k之前的都是已经 check 过的
    }
  }

  /**
   *  根据 knng 图生成 NSG 图
   */
  void link(const Graph<int> &knng, Graph<Node> &graph) {
    auto st = std::chrono::high_resolution_clock::now();
    std::atomic<int> cnt{0}; // 原子访问 cnt
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < nb; i++) {
      std::vector<Node> pool;
      std::vector<Neighbor> tmp;
      std::vector<bool> vis(nb);
      // 搜索过程中的途经点设为备选集 pool
      search_on_graph<true>(data + i * d, knng, vis, ep, L, tmp, pool);
      sync_prune(i, pool, vis, knng, graph); // 备选集剪枝，确定外邻居
      pool.clear();
      tmp.clear();
      int cur = cnt += 1;
      if (cur % 10000 == 0) {
        printf("NSG building progress: [%d/%d]\n", cur, nb);
      }
    }
    auto ed = std::chrono::high_resolution_clock::now();
    auto ela = std::chrono::duration<double>(ed - st).count();
    printf("NSG building cost: %.2lfs\n", ela);

    std::vector<std::mutex> locks(nb);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < nb; ++i) { // 添加反向边
      add_reverse_links(i, locks, graph);
    }
  }

  void sync_prune(int q, std::vector<Node> &pool, std::vector<bool> &vis,
                  const Graph<int> &knng, Graph<Node> &graph) {
      // 将 knng 上没有被访问的 q 的邻居也加入备选集 pool
    for (int i = 0; i < knng.K; i++) {
      int id = knng.at(q, i);
      if (id < 0 || id >= nb || vis[id]) {
        continue;
      }

      float dist = dist_func(data + q * d, data + id * d, d);
      pool.emplace_back(id, dist);
    }

    std::sort(pool.begin(), pool.end());

    std::vector<Node> result;

    int start = 0;
    if (pool[start].id == q) {
      start++;
    }
    result.push_back(pool[start]);

    while ((int)result.size() < R && (++start) < (int)pool.size() &&
           start < C) { // MRNG 的剪枝方法
      auto &p = pool[start];
      bool occlude = false;
      for (int t = 0; t < (int)result.size(); t++) { // 判断是否排除这个点
        if (p.id == result[t].id) {
          occlude = true;
          break;
        }

        float djk = dist_func(data + result[t].id * d, data + p.id * d, d);
        if (djk < p.distance /* dik */) {
          occlude = true;
          break;
        }
      }
      if (!occlude) {
        result.push_back(p);
      }
    }
    // 将剪枝结果写入 graph，设为 q 点的外邻居
    for (int i = 0; i < R; i++) {
      if (i < (int)result.size()) {
        graph.at(q, i).id = result[i].id;
        graph.at(q, i).distance = result[i].distance;
      } else {
        graph.at(q, i).id = EMPTY_ID;
      }
    }
  }

  void add_reverse_links(int q, std::vector<std::mutex> &locks,
                         Graph<Node> &graph) {
    for (int i = 0; i < R; i++) {
      if (graph.at(q, i).id == EMPTY_ID) {
        break;
      }

      Node sn(q, graph.at(q, i).distance);
      int des = graph.at(q, i).id;

      // 将 q 点和 des 点当前的外邻居加入 des 点的外邻居备选集 tmp_pool 进行剪枝，试图将 q 谁为 des 的外邻居（加反向边）
      std::vector<Node> tmp_pool;
      int dup = 0;
      {
        LockGuard guard(locks[des]);
        for (int j = 0; j < R; j++) {
          if (graph.at(des, j).id == EMPTY_ID) {
            break;
          }
          if (q == graph.at(des, j).id) {
            dup = 1;
            break;
          }
          tmp_pool.push_back(graph.at(des, j));
        }
      }

      if (dup) {
        continue; // 已经具有反向边
      }

      tmp_pool.push_back(sn);
      if ((int)tmp_pool.size() > R) { // 如果小于 R 则不需要剪枝可以直接添加反向边
        std::vector<Node> result;
        int start = 0;
        std::sort(tmp_pool.begin(), tmp_pool.end());
        result.push_back(tmp_pool[start]);

        // 和 sync_prune 一样的剪枝方法
        while ((int)result.size() < R && (++start) < (int)tmp_pool.size()) {
          auto &p = tmp_pool[start];
          bool occlude = false;
          for (int t = 0; t < (int)result.size(); t++) {
            if (p.id == result[t].id) {
              occlude = true;
              break;
            }
            float djk = dist_func(data + result[t].id * d, data + p.id * d, d);
            if (djk < p.distance /* dik */) {
              occlude = true;
              break;
            }
          }
          if (!occlude) {
            result.push_back(p);
          }
        }

        {
          LockGuard guard(locks[des]);
          for (int t = 0; t < (int)result.size(); t++) {
            graph.at(des, t) = result[t];
          }
        }

      } else { // 直接添加反向边
        LockGuard guard(locks[des]);
        for (int t = 0; t < R; t++) {
          if (graph.at(des, t).id == EMPTY_ID) {
            graph.at(des, t) = sn;
            break;
          }
        }
      }
    }
  }

  /**
   * 通过 dfs 树，将图连通
   */
  int tree_grow(std::vector<int> &degrees) {
      // 先以导航点为根，在对每个点赋边后产生的图上进行DFS，形成DFS树
    int root = ep;
    std::vector<bool> vis(nb);
    int num_attached = 0;
    int cnt = 0; // 已经连通的点数
    while (true) {
      cnt = dfs(vis, root, cnt);
      if (cnt >= nb) {
        break;
      }
      std::vector<bool> vis2(nb);
      root = attach_unlinked(vis, vis2, degrees);
      num_attached += 1;
    }
    return num_attached;
  }

  int dfs(std::vector<bool> &vis, int root, int cnt) const {
    int node = root;
    std::stack<int> stack;
    stack.push(root);
    if (vis[root]) {
      cnt++;
    }
    vis[root] = true;
    while (!stack.empty()) {
      int next = EMPTY_ID;
      for (int i = 0; i < R; i++) {
        int id = final_graph.at(node, i);
        if (id != EMPTY_ID && !vis[id]) {
          next = id;
          break;
        }
      }
      if (next == EMPTY_ID) {
        stack.pop();
        if (stack.empty()) {
          break;
        }
        node = stack.top();
        continue;
      }
      node = next;
      vis[node] = true;
      stack.push(node);
      cnt++;
    }
    return cnt;
  }

  int attach_unlinked(std::vector<bool> &vis, std::vector<bool> &vis2,
                      std::vector<int> &degrees) {
    int id = EMPTY_ID;
    for (int i = 0; i < nb; i++) {
      if (vis[i]) { /// ？？？
        id = i;
        break;
      }
    }
    if (id == EMPTY_ID) {
      return EMPTY_ID;
    }
    // 将该点作为query点，从导航点开始贪心搜索，将该点连接到近似最近邻上
    std::vector<Neighbor> tmp;
    std::vector<Node> pool;
    search_on_graph<true>(data + id * d, final_graph, vis2, ep, L, tmp, pool);
    std::sort(pool.begin(), pool.end());
    int node;
    bool found = false;
    // 用距离 id 最近的点进行连通
    for (int i = 0; i < (int)pool.size(); i++) {
      node = pool[i].id;
      if (degrees[node] < R && node != id) {
        found = true;
        break;
      }
    }
    if (!found) {
      do {
        node = rng.rand_int(nb);
        if (vis[node] && degrees[node] < R && node != id) {
          found = true;
        }
      } while (!found);
    }
    int pos = degrees[node];
    final_graph.at(node, pos) = id;
    degrees[node] += 1;
    return node;
  }
};

} // namespace glass