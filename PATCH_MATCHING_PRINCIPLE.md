# Point-LIO 非共面同步 Patch ICP

## 1. 适用数据

该模式面向一次曝光同时输出一组点的面阵雷达。工程使用
`curvature` 保存点相对时间，`time_compressing()` 将时间戳完全相同的点组成
一个 patch。一个 patch 内的点共享同一个雷达位姿，但不要求：

- 位于同一个平面；
- 命中同一个物体；
- 具有相同法向量。

只要地图对应正确，不同表面的点仍然服从同一个刚体变换。

## 2. 新匹配流程

启用 `mapping/patch_matching_en` 后，点数达到 `patch_icp_min_inliers` 的时间组
走以下流程；更小的时间组自动回退到原逐点点面模型：

```text
同步点组
  → 使用当前 ESKF 状态投影到世界系
  → 每个点从 iVox 查询 1 个最近点
  → Huber 加权
  → 对整组对应执行一次共享的 SVD 刚体对齐
  → 更新临时 patch 并重复 1~3 次
  → 重新查询最终对应
  → patch 级内点数、RMSE、位移和旋转门限
  → 构造三维点到点联合测量
  → 信息矩阵特征压缩
  → 一次 ESKF 更新
```

局部 ICP 只改变临时 patch，用于改善对应关系；不会直接修改滤波状态。最终状态
增量仍由 ESKF 根据先验协方差和 LiDAR 测量共同计算。

## 3. 点到点测量

对 patch 中第 (j) 个点：

[
p_j^w = R(R_{LI}p_j^L+t_{LI})+t
]

iVox 最近点为 (q_j^w)，测量残差为：

[
z_j=q_j^w-p_j^w
]

每个对应提供三维约束。固定外参时雅可比为：

[
H_j=[I,;-R[p_j^I]_\times,;0,;0]
]

估计外参时增加：

[
H_{R_{LI}}=-RR_{LI}[p_j^L]_\times,qquad H_{t_{LI}}=R
]

Huber 权重为 (w_j)，联合信息为：

[
A=\sum_j w_jH_j^TH_j,qquad b=\sum_jw_jH_j^Tz_j
]

随后对 (A) 特征分解并输出压缩测量。这一步只删除低信息方向，不再进行
任何平面拟合。

## 4. Patch 级拒绝条件

满足任一条件时整块测量无效：

- 有效对应少于 `patch_icp_min_inliers`；
- 最近点距离超过 `patch_icp_max_corr_dist`；
- 最终 RMSE 超过 `patch_icp_max_rmse`；
- 临时 ICP 平均位移超过局部搜索范围；
- 临时 ICP 旋转超过 0.35 rad；
- patch 空间分布退化成近似一条直线；
- 信息矩阵没有可保留的特征方向。

这避免少量错误点单独把滤波状态拉向错误地图位置。

## 5. 时间组保护

普通 PCL VoxelGrid 可能对 `PointXYZINormal::curvature` 求平均，使不同曝光时刻
产生虚假的平均时间戳。默认：

```yaml
patch_preserve_points: true
```

启用 patch 匹配后会跳过扫描内的全局体素降采样，从而保留原始同步点组。地图仍由
iVox 管理。若上游已经提供显式 patch ID 并实现了 patch-aware 降采样，可以关闭该
选项。

## 6. 参数

```yaml
mapping:
  patch_matching_en: true
  patch_cov_scale: 1.0
  patch_eigenvalue_thr: 0.01
  patch_icp_max_iterations: 3
  patch_icp_min_inliers: 3
  patch_icp_max_corr_dist: 1.0
  patch_icp_huber_delta: 0.20
  patch_icp_max_rmse: 0.30
  patch_preserve_points: true
```

参数含义：

- `patch_cov_scale`：点到点残差协方差缩放；漂移或震荡时优先增大。
- `patch_eigenvalue_thr`：相对最大特征值的信息方向截止比例。
- `patch_icp_max_iterations`：局部对应细化次数；8 点 patch 建议 2~3。
- `patch_icp_min_inliers`：最少有效对应数；8 点 patch 建议 3~5。
- `patch_icp_max_corr_dist`：初始状态允许的局部匹配半径。
- `patch_icp_huber_delta`：超过此残差后逐渐降低对应权重。
- `patch_icp_max_rmse`：最终 patch 一致性门限。
- `patch_preserve_points`：保护原始同步点组。

仓库中的通用雷达配置仍默认关闭 `patch_matching_en`，避免旋转扫描雷达被误当成
同步面阵雷达。面阵雷达使用的 YAML 必须显式设为 `true`。

## 7. 调参建议与限制

建议从以下组合开始：

```yaml
patch_icp_max_iterations: 2
patch_icp_min_inliers: 4
patch_icp_max_corr_dist: 0.5
patch_icp_huber_delta: 0.10
patch_icp_max_rmse: 0.15
patch_cov_scale: 2.0
```

再根据地图分辨率和初始位姿误差放宽距离门限。

点到点 ICP 不依赖平面，但会受到地图采样密度影响：在大面积平滑表面上，离散地图点
可能产生不真实的切向约束。因此建议提高 `patch_cov_scale`，并通过真实数据比较
patch RMSE、轨迹抖动和闭环误差。若后续需要进一步提高平滑表面的精度，可在不要求
patch 共面的前提下，为每个对应加入局部协方差，升级为 Generalized ICP。
