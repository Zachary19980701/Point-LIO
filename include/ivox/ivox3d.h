/**
 * @file ivox3d.h
 * @brief 增量式体素地图 (iVox) - 对应论文Section IV
 *
 * iVox是一种高效的3D点云地图数据结构,主要特点:
 * 1. 使用哈希表存储体素网格,支持O(1)平均时间复杂度的插入和查找
 * 2. 使用LRU缓存策略管理内存,超过容量时自动删除最久未访问的体素
 * 3. 支持多种邻域搜索模式 (6邻域, 18邻域, 26邻域)
 * 4. 每个体素内部使用线性存储或PHC (Pseudo-Hilbert Curve) 存储
 *
 * 相比传统kd-tree的优势:
 * - 插入效率更高 (O(1) vs O(log n))
 * - 支持增量更新
 * - 内存占用可控 (通过capacity参数)
 */

//
// Created by xiang on 2021/9/16.
//

#ifndef FASTER_LIO_IVOX3D_H
#define FASTER_LIO_IVOX3D_H

#include <glog/logging.h>
#include <list>
#include <thread>
#include <unordered_map>

#include "eigen_types.h"
#include "ivox3d_node.hpp"

namespace faster_lio {

// ==================== 体素节点类型枚举 ====================
enum class IVoxNodeType {
    DEFAULT,  // 线性存储: 适合稀疏点云
    PHC,      // PHC存储: 使用伪希尔伯特曲线,适合密集点云
};

// ==================== 体素节点类型特征模板 ====================
template <IVoxNodeType node_type, typename PointT, int dim>
struct IVoxNodeTypeTraits {};

template <typename PointT, int dim>
struct IVoxNodeTypeTraits<IVoxNodeType::DEFAULT, PointT, dim> {
    using NodeType = IVoxNode<PointT, dim>;
};

template <typename PointT, int dim>
struct IVoxNodeTypeTraits<IVoxNodeType::PHC, PointT, dim> {
    using NodeType = IVoxNodePhc<PointT, dim>;
};

/**
 * @brief iVox增量体素地图类
 *
 * @tparam dim 维度 (默认3)
 * @tparam node_type 体素节点类型
 * @tparam PointType 点类型
 */
template <int dim = 3, IVoxNodeType node_type = IVoxNodeType::DEFAULT, typename PointType = pcl::PointXYZ>
class IVox {
   public:
    using KeyType = Eigen::Matrix<int, dim, 1>;  // 体素网格索引
    using PtType = Eigen::Matrix<float, dim, 1>; // 点坐标
    using NodeType = typename IVoxNodeTypeTraits<node_type, PointType, dim>::NodeType;
    using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
    using DistPoint = typename NodeType::DistPoint;

    // ==================== 邻域搜索类型 ====================
    enum class NearbyType {
        CENTER,    // 仅中心网格
        NEARBY6,   // 6邻域 (上下左右前后)
        NEARBY18,  // 18邻域
        NEARBY26,  // 26邻域 (所有相邻网格)
    };

    // ==================== iVox配置选项 ====================
    struct Options {
        float resolution_ = 0.2;                        // 体素分辨率 (米)
        float inv_resolution_ = 10.0;                   // 分辨率倒数 (用于快速计算)
        NearbyType nearby_type_ = NearbyType::NEARBY6;  // 最近邻搜索的邻域范围
        std::size_t capacity_ = 1000000;                // 最大体素数量 (内存控制)
    };

    /**
     * @brief 构造函数
     * @param options iVox配置选项
     */
    explicit IVox(Options options) : options_(options) {
        options_.inv_resolution_ = 1.0 / options_.resolution_;
        GenerateNearbyGrids();  // 预计算邻域网格偏移量
    }

    /**
     * @brief 添加点到地图
     * @param points_to_add 待添加的点集
     */
    void AddPoints(const PointVector& points_to_add);

    /**
     * @brief 获取最近邻点 (单个最近邻)
     * @param pt 查询点
     * @param closest_pt 输出的最近邻点
     * @return 是否找到
     */
    bool GetClosestPoint(const PointType& pt, PointType& closest_pt);

    /**
     * @brief 获取K个最近邻点
     * @param pt 查询点
     * @param closest_pt 输出的最近邻点集
     * @param max_num 最大返回点数
     * @param max_range 最大搜索距离
     * @return 是否找到
     */
    bool GetClosestPoint(const PointType& pt, PointVector& closest_pt, int max_num = 5, double max_range = 5.0);

    /// 批量获取最近邻点
    bool GetClosestPoint(const PointVector& cloud, PointVector& closest_cloud);

    /// 获取地图中的总点数
    size_t NumPoints() const;

    /// 导出当前 iVox 内部实际保存的全部地图点
    void GetAllPoints(PointVector& points) const;

    /// 获取有效体素网格数量
    size_t NumValidGrids() const;

    /// 获取点云统计信息
    std::vector<float> StatGridPoints() const;

    // ==================== 公开成员变量 ====================
    // 哈希表: 体素索引 -> 体素节点迭代器
    std::unordered_map<KeyType, typename std::list<std::pair<KeyType, NodeType>>::iterator, hash_vec<dim>>
        grids_map_;

    /**
     * @brief 将点坐标转换为体素网格索引
     * @param pt 点坐标
     * @return 体素网格索引
     */
    KeyType Pos2Grid(const PtType& pt) const;
    KeyType Pos2Grid_(const PtType& pt, const double &defined_res) const;

   private:
    /// generate the nearby grids according to the given options
    void GenerateNearbyGrids();

    /// position to grid
    // KeyType Pos2Grid(const PtType& pt) const;

    Options options_;
    // std::unordered_map<KeyType, typename std::list<std::pair<KeyType, NodeType>>::iterator, hash_vec<dim>>
        // grids_map_;                                        // voxel hash map
    std::list<std::pair<KeyType, NodeType>> grids_cache_;  // voxel cache
    std::vector<KeyType> nearby_grids_;                    // nearbys
};

/**
 * @brief 获取单个最近邻点
 *
 * 算法流程:
 * 1. 计算查询点所属的体素网格索引
 * 2. 在所有邻域网格中搜索候选点
 * 3. 返回距离最近的点
 */
template <int dim, IVoxNodeType node_type, typename PointType>
bool IVox<dim, node_type, PointType>::GetClosestPoint(const PointType& pt, PointType& closest_pt) {
    std::vector<DistPoint> candidates;

    // 计算查询点所属的体素网格索引
    auto key = Pos2Grid(ToEigen<float, dim>(pt));

    // 在所有邻域网格中搜索
    std::for_each(nearby_grids_.begin(), nearby_grids_.end(), [&key, &candidates, &pt, this](const KeyType& delta) {
        auto dkey = key + delta;
        auto iter = grids_map_.find(dkey);
        if (iter != grids_map_.end()) {
            DistPoint dist_point;
            bool found = iter->second->second.NNPoint(pt, dist_point);
            if (found) {
                candidates.emplace_back(dist_point);
            }
        }
    });

    if (candidates.empty()) {
        return false;
    }

    // 返回距离最近的点
    auto iter = std::min_element(candidates.begin(), candidates.end());
    closest_pt = iter->Get();
    return true;
}

template <int dim, IVoxNodeType node_type, typename PointType>
bool IVox<dim, node_type, PointType>::GetClosestPoint(const PointType& pt, PointVector& closest_pt, int max_num,
                                                      double max_range) {
    std::vector<DistPoint> candidates;
    candidates.reserve(max_num * nearby_grids_.size());

    auto key = Pos2Grid(ToEigen<float, dim>(pt));

// #define INNER_TIMER
#ifdef INNER_TIMER
    static std::unordered_map<std::string, std::vector<int64_t>> stats;
    if (stats.empty()) {
        stats["knn"] = std::vector<int64_t>();
        stats["nth"] = std::vector<int64_t>();
    }
#endif

    for (const KeyType& delta : nearby_grids_) {
        auto dkey = key + delta;
        auto iter = grids_map_.find(dkey);
        if (iter != grids_map_.end()) {
#ifdef INNER_TIMER
            auto t1 = std::chrono::high_resolution_clock::now();
#endif
            auto tmp = iter->second->second.KNNPointByCondition(candidates, pt, max_num, max_range);
#ifdef INNER_TIMER
            auto t2 = std::chrono::high_resolution_clock::now();
            auto knn = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
            stats["knn"].emplace_back(knn);
#endif
        }
    }

    if (candidates.empty()) {
        return false;
    }

#ifdef INNER_TIMER
    auto t1 = std::chrono::high_resolution_clock::now();
#endif

    if (candidates.size() <= max_num) {
    } else {
        std::nth_element(candidates.begin(), candidates.begin() + max_num - 1, candidates.end());
        candidates.resize(max_num);
    }
    std::nth_element(candidates.begin(), candidates.begin(), candidates.end());

#ifdef INNER_TIMER
    auto t2 = std::chrono::high_resolution_clock::now();
    auto nth = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    stats["nth"].emplace_back(nth);

    constexpr int STAT_PERIOD = 100000;
    if (!stats["nth"].empty() && stats["nth"].size() % STAT_PERIOD == 0) {
        for (auto& it : stats) {
            const std::string& key = it.first;
            std::vector<int64_t>& stat = it.second;
            int64_t sum_ = std::accumulate(stat.begin(), stat.end(), 0);
            int64_t num_ = stat.size();
            stat.clear();
            std::cout << "inner_" << key << "(ns): sum=" << sum_ << " num=" << num_ << " ave=" << 1.0 * sum_ / num_
                      << " ave*n=" << 1.0 * sum_ / STAT_PERIOD << std::endl;
        }
    }
#endif

    closest_pt.clear();
    for (auto& it : candidates) {
        closest_pt.emplace_back(it.Get());
    }
    return closest_pt.empty() == false;
}

template <int dim, IVoxNodeType node_type, typename PointType>
size_t IVox<dim, node_type, PointType>::NumValidGrids() const {
    return grids_map_.size();
}

template <int dim, IVoxNodeType node_type, typename PointType>
size_t IVox<dim, node_type, PointType>::NumPoints() const {
    size_t num_points = 0;
    for (const auto& grid : grids_cache_) {
        num_points += grid.second.Size();
    }
    return num_points;
}

template <int dim, IVoxNodeType node_type, typename PointType>
void IVox<dim, node_type, PointType>::GetAllPoints(PointVector& points) const {
    points.clear();
    points.reserve(NumPoints());
    for (const auto& grid : grids_cache_) {
        const NodeType& node = grid.second;
        for (size_t i = 0; i < node.Size(); ++i) {
            points.emplace_back(node.GetPoint(i));
        }
    }
}

/**
 * @brief 生成邻域网格偏移量列表
 *
 * 根据配置的邻域类型,预计算所有需要搜索的邻域网格偏移量:
 * - CENTER: 仅搜索当前网格
 * - NEARBY6: 搜索6个相邻网格 (上下左右前后)
 * - NEARBY18: 搜索18个相邻网格
 * - NEARBY26: 搜索所有26个相邻网格 (3×3×3-1)
 *
 * 预计算可以避免在搜索时重复计算偏移量
 */
template <int dim, IVoxNodeType node_type, typename PointType>
void IVox<dim, node_type, PointType>::GenerateNearbyGrids() {
    if (options_.nearby_type_ == NearbyType::CENTER) {
        // 仅中心网格
        nearby_grids_.emplace_back(KeyType::Zero());
    } else if (options_.nearby_type_ == NearbyType::NEARBY6) {
        // 6邻域: 中心 + 6个直接相邻网格
        nearby_grids_ = {KeyType(0, 0, 0),  KeyType(-1, 0, 0), KeyType(1, 0, 0), KeyType(0, 1, 0),
                         KeyType(0, -1, 0), KeyType(0, 0, -1), KeyType(0, 0, 1)};
    } else if (options_.nearby_type_ == NearbyType::NEARBY18) {
        // 18邻域: 中心 + 面相邻 + 边相邻
        nearby_grids_ = {KeyType(0, 0, 0),  KeyType(-1, 0, 0), KeyType(1, 0, 0),   KeyType(0, 1, 0),
                         KeyType(0, -1, 0), KeyType(0, 0, -1), KeyType(0, 0, 1),   KeyType(1, 1, 0),
                         KeyType(-1, 1, 0), KeyType(1, -1, 0), KeyType(-1, -1, 0), KeyType(1, 0, 1),
                         KeyType(-1, 0, 1), KeyType(1, 0, -1), KeyType(-1, 0, -1), KeyType(0, 1, 1),
                         KeyType(0, -1, 1), KeyType(0, 1, -1), KeyType(0, -1, -1)};
    } else if (options_.nearby_type_ == NearbyType::NEARBY26) {
        // 26邻域: 所有相邻网格 (包括角相邻)
        nearby_grids_ = {KeyType(0, 0, 0),   KeyType(-1, 0, 0),  KeyType(1, 0, 0),   KeyType(0, 1, 0),
                         KeyType(0, -1, 0),  KeyType(0, 0, -1),  KeyType(0, 0, 1),   KeyType(1, 1, 0),
                         KeyType(-1, 1, 0),  KeyType(1, -1, 0),  KeyType(-1, -1, 0), KeyType(1, 0, 1),
                         KeyType(-1, 0, 1),  KeyType(1, 0, -1),  KeyType(-1, 0, -1), KeyType(0, 1, 1),
                         KeyType(0, -1, 1),  KeyType(0, 1, -1),  KeyType(0, -1, -1), KeyType(1, 1, 1),
                         KeyType(-1, 1, 1),  KeyType(1, -1, 1),  KeyType(1, 1, -1),  KeyType(-1, -1, 1),
                         KeyType(-1, 1, -1), KeyType(1, -1, -1), KeyType(-1, -1, -1)};
    } else {
        // LOG(ERROR) << "Unknown nearby_type!";
    }
}

template <int dim, IVoxNodeType node_type, typename PointType>
bool IVox<dim, node_type, PointType>::GetClosestPoint(const PointVector& cloud, PointVector& closest_cloud) {
    std::vector<size_t> index(cloud.size());
    
    closest_cloud.resize(cloud.size());

    for (int i = 0; i < cloud.size(); ++i) {
        PointType pt;
        if (GetClosestPoint(cloud[i], pt)) {
            closest_cloud[i] = pt;
        } else {
            closest_cloud[i] = PointType();
        }
    };
    return true;
}

/**
 * @brief 添加点到iVox地图
 *
 * 算法流程:
 * 1. 对于每个点,计算其所属的体素网格索引
 * 2. 如果网格不存在:
 *    a. 创建新网格并插入到缓存列表头部
 *    b. 更新哈希表
 *    c. 检查容量,超过则删除最久未访问的网格 (LRU策略)
 * 3. 如果网格已存在:
 *    a. 将点插入到网格中
 *    b. 更新LRU顺序 (将网格移到缓存列表头部)
 *
 * LRU (Least Recently Used) 缓存策略确保:
 * - 最近访问的网格保持在内存中
 * - 超过容量时自动删除最久未访问的网格
 *
 * @param points_to_add 待添加的点集
 */
template <int dim, IVoxNodeType node_type, typename PointType>
void IVox<dim, node_type, PointType>::AddPoints(const PointVector& points_to_add) {
    for(size_t i = 0; i < points_to_add.size(); i++) {
        // 步骤1: 计算点所属的体素网格索引
        auto key = Pos2Grid(Eigen::Matrix<float, dim, 1>(points_to_add[i].x, points_to_add[i].y, points_to_add[i].z));

        // 步骤2: 查找体素网格
        auto iter = grids_map_.find(key);
        if (iter == grids_map_.end()) {
            // 网格不存在: 创建新网格

            // 计算网格中心点
            PointType center;
            center.getVector3fMap() = key.template cast<float>() * options_.resolution_;

            // 插入到缓存列表头部
            grids_cache_.push_front({key, NodeType(center, options_.resolution_)});
            grids_map_.insert({key, grids_cache_.begin()});

            // 将点插入到新网格中
            grids_cache_.front().second.InsertPoint(points_to_add[i]);

            // 容量检查: 超过容量时删除最久未访问的网格
            if (grids_map_.size() >= options_.capacity_) {
                grids_map_.erase(grids_cache_.back().first);
                grids_cache_.pop_back();
            }
        } else {
            // 网格已存在: 插入点并更新LRU顺序

            iter->second->second.InsertPoint(points_to_add[i]);

            // 将网格移到缓存列表头部 (LRU更新)
            grids_cache_.splice(grids_cache_.begin(), grids_cache_, iter->second);
            grids_map_[key] = grids_cache_.begin();
        }
    }
}

/**
 * @brief 将点坐标转换为体素网格索引
 *
 * 公式: grid_index = floor(point * inv_resolution)
 *
 * @param pt 点坐标
 * @return 体素网格索引 (整数坐标)
 */
template <int dim, IVoxNodeType node_type, typename PointType>
Eigen::Matrix<int, dim, 1> IVox<dim, node_type, PointType>::Pos2Grid(const IVox::PtType& pt) const {
    return (pt * options_.inv_resolution_).array().floor().template cast<int>();
}

template <int dim, IVoxNodeType node_type, typename PointType>
Eigen::Matrix<int, dim, 1> IVox<dim, node_type, PointType>::Pos2Grid_(const IVox::PtType& pt, const double &defined_res) const {
    return (pt / defined_res).array().floor().template cast<int>();
}

/**
 * @brief 获取网格中点数量的统计信息
 *
 * @return 统计向量 [有效网格数, 平均点数, 最大点数, 最小点数, 标准差]
 */
template <int dim, IVoxNodeType node_type, typename PointType>
std::vector<float> IVox<dim, node_type, PointType>::StatGridPoints() const {
    int num = grids_cache_.size(), valid_num = 0, max = 0, min = 100000000;
    int sum = 0, sum_square = 0;

    for (auto& it : grids_cache_) {
        int s = it.second.Size();
        valid_num += s > 0;
        max = s > max ? s : max;
        min = s < min ? s : min;
        sum += s;
        sum_square += s * s;
    }

    float ave = float(sum) / num;
    float stddev = num > 1 ? sqrt((float(sum_square) - num * ave * ave) / (num - 1)) : 0;

    return std::vector<float>{valid_num, ave, max, min, stddev};
}

}  // namespace faster_lio

#endif
