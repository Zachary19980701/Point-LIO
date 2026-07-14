# Point-LIO 面阵雷达 Patch ICP 匹配原理文档

## 1. 问题

面阵雷达的 patch 内 N 个点共享时间戳和传感器位姿。现有逐点匹配将 N 个点作为 N 个独立测量，patch 区域被过度加权 N 倍。

**目标**：把 patch 作为一个子点云，和地图做 ICP 匹配，输出 1 个等效测量。

## 2. 原理：法方程压缩

### 2.1 从 N 个约束到 1 个 ICP 解

每个点贡献一个点到平面约束：

```
点 j:    r_j = n_j^T * p_world_j + d_j ≈ 0
线性化:  r_j + H_j * δx ≈ 0    (H_j: 1×12 行向量)
```

N 个点构成一个局部最小二乘问题：

```
min  Σ (H_j * δx + z_j)²
δx
```

其法方程为：

```
A * δx = -b

其中:  A = Σ H_j^T * H_j    (12×12)
       b = Σ H_j^T * z_j    (12×1)
```

直接解 `δx = -A^(-1) * b` 就是 Gauss-Newton 步长（即 ICP 的一次迭代修正量）。

### 2.2 信息压缩

但 A 通常是秩亏的。对于平面上的点到平面约束：

```
A 的秩 r ≤ 3（1 个法向平移 + 2 个旋转）
```

只有跨越多个不同方向的平面时，r 才能达到 6。

做特征分解只保留有效维度：

```
A = V * D * V^T

D = diag(λ_1, ..., λ_12),  λ_1 ≥ λ_2 ≥ ... ≥ λ_12

保留: λ_i > eigenvalue_thr 的 r 个维度
```

构造压缩测量：

```
D_r = diag(λ_1, ..., λ_r)              (r×r)
V_r = [v_1, ..., v_r]                  (12×r)

H_icp = sqrt(D_r) * V_r^T             (r×12)
z_icp = D_r^(-1/2) * V_r^T * b        (r×1)
R_icp = σ² * I_r                       (r×r)
```

这个 `(H_icp, z_icp, R_icp)` 在信息论意义上与原 N 个约束等价：

```
H_icp^T * R_icp^(-1) * H_icp = A / σ²   ← 同样的信息矩阵
H_icp^T * R_icp^(-1) * z_icp = b / σ²   ← 同样的梯度
```

### 2.3 为什么这等价于局部 ICP

Gauss-Newton ICP 的一次迭代：

```
δx = -(J^T * J)^(-1) * J^T * r
   = -A^(-1) * b
```

其中 J 是 N 个点到平面距离的 Jacobian，r 是残差。在 IESKF 中，这个修正量通过 Kalman 更新融入，而非直接应用——IESKF 会结合先验协方差 P 做加权：

```
dx = P * H_icp^T * (H_icp * P * H_icp^T + R_icp)^(-1) * z_icp
```

这和直接应用 δx 不同——IESKF 在 ICP 修正和先验之间做了最优权衡。

## 3. 方案对比

| 维度 | 逐点 (现有) | 平均合并 | ICP 压缩 (本方案) |
|------|-----------|---------|-------------------|
| 第一遍 (KNN+平面) | N 次 | N 次 | N 次 |
| 第二遍 | N 行 H, N 个 z | avg → 1 行 | A,b → 特征分解 → r 行 |
| 输出维度 | N | 1 | **r (1~6, 自动)** |
| 秩自动检测 | 否 | 否 | **是** |
| 信息丢失 | 无 (但过量) | 有 (假设共线) | **无** |
| 统计正确 | ❌ N倍过量 | ⚠️ | **✅** |
| IESKF 求逆 | (N×N)⁻¹ | 标量 | **(r×r)⁻¹** |
| 单平面退化为 | N 行 | 1 行 | **自动退化到 r≤3** |

## 4. 数据流

```
LiDAR frame
  → curvature = 时间戳
  → sort + time_compressing → time_seq
  → 预计算: pbody_list, crossmat_list

IESKF 主循环 (不变):
  for k in time_seq:
      
      IMU 传播到 time_current
      
      if patch_matching_en && time_seq[k] > 1:
          ┌──────────────────────────────────────────┐
          │ h_model_*_patch():                       │
          │                                          │
          │ ① 逐点匹配 (与现有完全相同):              │
          │    for j in [0, N):                      │
          │        变换 → KNN → esti_plane → 验证     │
          │        → H_j(1×12), z_j(标量)             │
          │                                          │
          │ ② 构建法方程:                             │
          │    A = Σ H_j^T * H_j   (12×12)           │
          │    b = Σ H_j^T * z_j   (12×1)            │
          │                                          │
          │ ③ 特征分解 + 压缩:                        │
          │    A = V*D*V^T                           │
          │    保留 λ_i > thr 的 r 个维度              │
          │    H_icp = sqrt(D_r)*V_r^T (r×12)        │
          │    z_icp = D_r^(-1/2)*V_r^T*b (r×1)      │
          │                                          │
          │ 输出: h_x(r×12), z(r×1), R=σ²*I_r       │
          └──────────────────────────────────────────┘
      else:
          逐点匹配 (现有逻辑)
      
      IESKF ← K = P*H^T*(H*P*H^T+R)^(-1)
      dx = K*z, x = x ⊞ dx
      
      投影点到世界系 → MapIncremental
```

## 5. 关键代码结构

```cpp
void h_model_input_patch(state_input &s, ..., ekfom_data)
{
    int N = time_seq[k];
    int start = idx + 1;

    // ===== ① 逐点 KNN + 平面拟合 (与现有完全相同) =====
    normvec->resize(N);
    for (int j = 0; j < N; j++) {
        pointBodyToWorld(...);
        ivox_->GetClosestPoint(...);
        if (esti_plane(...) && valid) {
            normvec->points[j] = ...;   // 保存地图平面参数
            point_selected_surf[start+j] = true;
        }
    }

    // ===== ② 构建法方程 A(12×12), b(12×1) =====
    Eigen::Matrix<double, 12, 12> A = Eigen::Matrix<double, 12, 12>::Zero();
    Eigen::Matrix<double, 12, 1>  b = Eigen::Matrix<double, 12, 1>::Zero();
    int valid_count = 0;

    for (int j = 0; j < N; j++) {
        if (!point_selected_surf[start+j]) continue;

        // 计算 H_j (1×12) — 与现有代码完全相同
        Eigen::Matrix<double, 1, 12> H_j;
        V3D norm_vec(...从 normvec[j] 取得...);
        V3D p_body = pbody_list[start+j];
        
        // H_j 的前3列: n^T
        // H_j 的第4-6列: n^T * R * [p_imu]×  (旋转部分)
        // ... Jacobian 计算与现有完全一致 ...

        double z_j = -(n_i^T * p_world_i + d_i);

        A += H_j.transpose() * H_j;
        b += H_j.transpose() * z_j;
        valid_count++;
    }

    if (valid_count == 0) { ekfom_data.valid = false; return; }

    // ===== ③ 特征分解 + 压缩 =====
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 12, 12>> eig(A);
    auto &D = eig.eigenvalues();   // 12 个特征值 (升序)
    auto &V = eig.eigenvectors();  // 12×12 特征向量

    // 统计特征值 > threshold 的个数 (从大到小)
    int r = 0;
    for (int i = 11; i >= 0; i--) {  // 从大到小
        if (D(i) > patch_eigenvalue_thr * D(11)) r++;
        else break;
    }

    // 构造 H_icp (r×12), z_icp (r×1)
    ekfom_data.h_x.resize(r, 12);
    ekfom_data.z.resize(r);

    for (int i = 0; i < r; i++) {
        double sqrt_lambda = sqrt(D(11 - i));  // 从大到小取
        ekfom_data.h_x.row(i) = sqrt_lambda * V.col(11 - i).transpose();
        ekfom_data.z(i) = V.col(11 - i).dot(b) / sqrt_lambda;
    }

    ekfom_data.M_Noise = laser_point_cov * patch_cov_scale;
    effct_feat_num += 1;
}
```

## 6. 配置参数

```yaml
patch_matching_en: false         # 启用 patch ICP 匹配
patch_cov_scale: 1.0             # 测量噪声缩放
patch_eigenvalue_thr: 0.01       # 特征值阈值 (相对最大特征值的比例)
                                 # 控制有效秩: 仅 λ_i/λ_max > thr 被保留
```

## 7. 为什么这个方案正确

1. **信息无损**：法方程 A 包含了 N 个约束的全部有效信息，特征分解只丢弃了噪声维度的分量
2. **自动降秩**：对于平面约束（rank ≤ 3），自动只保留 1~3 维；对于丰富几何（跨越多平面），自动保留更多维度
3. **IESKF 友好**：输出为标准 `(H, z, R)` 格式，后端无需任何修改
4. **退化处理**：当 patch 在退化几何上（如长走廊），A 的秩更小，自动调整测量维度
