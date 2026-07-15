/**
 * @file common_lib.h
 * @brief Point-LIO通用库 - 数据结构定义和工具函数
 *
 * 本文件定义了Point-LIO的核心数据结构:
 * 1. 状态向量 (state_input, state_output)
 * 2. 输入向量 (input_ikfom)
 * 3. 测量数据结构 (MeasureGroup)
 * 4. 各种工具函数和宏定义
 */

#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include <so3_math.h>
#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <../include/IKFoM/IKFoM_toolkit/esekfom/esekfom.hpp>
#include <queue>

using namespace std;
using namespace Eigen;

// ==================== 流形类型定义 ====================
typedef MTK::vect<3, double> vect3;  // 3维向量
typedef MTK::SO3<double> SO3;        // SO(3)旋转群
typedef MTK::S2<double, 98090, 10000, 1> S2;  // S2球面 (用于重力向量表示)
typedef MTK::vect<1, double> vect1;  // 1维向量
typedef MTK::vect<2, double> vect2;  // 2维向量

// ==================== 输入模式状态向量定义 (24维) ====================
// 对应论文公式(2): 状态向量
MTK_BUILD_MANIFOLD(state_input,
((vect3, pos))          // [0:3]   位置
((SO3, rot))            // [3:6]   姿态 (SO3流形)
((SO3, offset_R_L_I))   // [6:9]   LiDAR到IMU的旋转外参
((vect3, offset_T_L_I)) // [9:12]  LiDAR到IMU的平移外参
((vect3, vel))          // [12:15] 速度
((vect3, bg))           // [15:18] 陀螺仪零偏
((vect3, ba))           // [18:21] 加速度计零偏
((vect3, gravity))      // [21:24] 重力向量
);

// ==================== 输出模式状态向量定义 (30维) ====================
// 输出模式额外包含角速度和加速度状态
MTK_BUILD_MANIFOLD(state_output,
((vect3, pos))          // [0:3]   位置
((SO3, rot))            // [3:6]   姿态
((SO3, offset_R_L_I))   // [6:9]   LiDAR到IMU的旋转外参
((vect3, offset_T_L_I)) // [9:12]  LiDAR到IMU的平移外参
((vect3, vel))          // [12:15] 速度
((vect3, omg))          // [15:18] 角速度 (输出模式特有)
((vect3, acc))          // [18:21] 加速度 (输出模式特有)
((vect3, gravity))      // [21:24] 重力向量
((vect3, bg))           // [24:27] 陀螺仪零偏
((vect3, ba))           // [27:30] 加速度计零偏
);

// ==================== 输入向量定义 ====================
// IMU测量输入: 加速度和角速度
MTK_BUILD_MANIFOLD(input_ikfom,
((vect3, acc))          // 加速度计测量
((vect3, gyro))         // 陀螺仪测量
);

MTK_BUILD_MANIFOLD(process_noise_input,
((vect3, ng))
((vect3, na))
((vect3, nbg))
((vect3, nba))
);

MTK_BUILD_MANIFOLD(process_noise_output,
((vect3, vel))
((vect3, ng))
((vect3, na))
((vect3, nbg))
((vect3, nba))
);

extern esekfom::esekf<state_input, 24, input_ikfom> kf_input;
extern esekfom::esekf<state_output, 30, input_ikfom> kf_output;

#define PBWIDTH 30
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"

#define PI_M (3.14159265358)
// #define G_m_s2 (9.81)         // Gravaty const in GuangDong/China
#define DIM_STATE (24)      // Dimension of states (Let Dim(SO(3)) = 3)
#define DIM_PROC_N (12)      // Dimension of process noise (Let Dim(SO(3)) = 3)
#define CUBE_LEN  (6.0)
#define LIDAR_SP_LEN    (2)
#define INIT_COV   (0.0001)
#define NUM_MATCH_POINTS    (5)
#define MAX_MEAS_DIM        (10000)

#define VEC_FROM_ARRAY(v)        v[0],v[1],v[2]
#define VEC_FROM_ARRAY_SIX(v)        v[0],v[1],v[2],v[3],v[4],v[5]
#define MAT_FROM_ARRAY(v)        v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]
#define CONSTRAIN(v,min,max)     ((v>min)?((v<max)?v:max):min)
#define ARRAY_FROM_EIGEN(mat)    mat.data(), mat.data() + mat.rows() * mat.cols()
#define STD_VEC_FROM_EIGEN(mat)  vector<decltype(mat)::Scalar> (mat.data(), mat.data() + mat.rows() * mat.cols())
#define DEBUG_FILE_DIR(name)     (string(string(ROOT_DIR) + "Log/"+ name))

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointXYZRGB     PointTypeRGB;
typedef pcl::PointCloud<PointType>    PointCloudXYZI;
typedef pcl::PointCloud<PointTypeRGB> PointCloudXYZRGB;
typedef vector<PointType, Eigen::aligned_allocator<PointType>>  PointVector;
typedef Vector3d V3D;
typedef Matrix3d M3D;
typedef Vector3f V3F;
typedef Matrix3f M3F;

#define MD(a,b)  Matrix<double, (a), (b)>
#define VD(a)    Matrix<double, (a), 1>
#define MF(a,b)  Matrix<float, (a), (b)>
#define VF(a)    Matrix<float, (a), 1>

const M3D Eye3d(M3D::Identity());
const M3F Eye3f(M3F::Identity());
const V3D Zero3d(0, 0, 0);
const V3F Zero3f(0, 0, 0);

/**
 * @brief 测量数据组结构
 *
 * 存储一个LiDAR扫描周期内的所有测量数据:
 * - LiDAR点云数据
 * - 对应时间段的IMU测量数据
 *
 * 用于数据同步和时间对齐
 */
struct MeasureGroup
{
    MeasureGroup()
    {
        lidar_beg_time = 0.0;
        lidar_last_time = 0.0;
        this->lidar.reset(new PointCloudXYZI());
    };
    double lidar_beg_time;   // LiDAR扫描开始时间
    double lidar_last_time;  // LiDAR扫描结束时间
    PointCloudXYZI::Ptr lidar;  // LiDAR点云数据
    deque<sensor_msgs::Imu::ConstPtr> imu;  // 对应时间段的IMU数据队列
};

template <typename T>
T calc_dist(PointType p1, PointType p2){
    T d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    return d;
}

template <typename T>
T calc_dist(Eigen::Vector3d p1, PointType p2){
    T d = (p1(0) - p2.x) * (p1(0) - p2.x) + (p1(1) - p2.y) * (p1(1) - p2.y) + (p1(2) - p2.z) * (p1(2) - p2.z);
    return d;
}

/**
 * @brief 时间序列压缩函数
 *
 * 将降采样后的点云按时间戳分组:
 * - 点云中每个点的curvature字段存储了时间偏移量
 * - 连续相同时间戳的点被分为一组
 * - 返回每组中的点数量
 *
 * 用于逐点卡尔曼滤波更新: 每组点使用相同的状态进行更新
 *
 * @param point_cloud 输入点云
 * @return 每组点数量的向量
 */
template<typename T>
std::vector<int> time_compressing(const PointCloudXYZI::Ptr &point_cloud)
{
  int points_size = point_cloud->points.size();
  int j = 0;
  std::vector<int> time_seq;
  time_seq.reserve(points_size);

  for(int i = 0; i < points_size - 1; i++)
  {
    j++;
    // 如果下一个点的时间戳大于当前点,说明是新的一组
    if (point_cloud->points[i+1].curvature > point_cloud->points[i].curvature)
    {
      time_seq.emplace_back(j);
      j = 0;
    }
  }
  // 处理最后一组点
  {
    time_seq.emplace_back(j+1);
  }
  return time_seq;
}

/* comment
plane equation: Ax + By + Cz + D = 0
convert to: A/D*x + B/D*y + C/D*z = -1
solve: A0*x0 = b0
where A0_i = [x_i, y_i, z_i], x0 = [A/D, B/D, C/D]^T, b0 = [-1, ..., -1]^T
normvec:  normalized x0
*/
template<typename T>
bool esti_normvector(Matrix<T, 3, 1> &normvec, const PointVector &point, const T &threshold, const int &point_num)
{
    MatrixXf A(point_num, 3);
    MatrixXf b(point_num, 1);
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < point_num; j++)
    {
        A(j,0) = point[j].x;
        A(j,1) = point[j].y;
        A(j,2) = point[j].z;
    }
    normvec = A.colPivHouseholderQr().solve(b);
    
    for (int j = 0; j < point_num; j++)
    {
        if (fabs(normvec(0) * point[j].x + normvec(1) * point[j].y + normvec(2) * point[j].z + 1.0f) > threshold)
        {
            return false;
        }
    }

    normvec.normalize();
    return true;
}

/**
 * @brief 平面拟合函数 - 使用最小二乘法拟合局部平面 逐点的拟合平面
 *
 * 对应论文Section III-C: 点到平面距离测量模型
 *
 * 平面方程: Ax + By + Cz + D = 0
 *
 * 算法步骤:
 * 1. 将平面方程转化为: A/D*x + B/D*y + C/D*z = -1
 * 2. 构建线性方程组 A * x = b,其中:
 *    - A = [x_i, y_i, z_i] (点坐标矩阵)
 *    - x = [A/D, B/D, C/D]^T (未知向量)
 *    - b = [-1, ..., -1]^T
 * 3. 使用QR分解求解最小二乘解
 * 4. 归一化得到平面参数 [A, B, C, D]
 *
 * @param pca_result 输出的平面参数 [A, B, C, D]
 * @param point 输入的点集 (NUM_MATCH_POINTS个点)
 * @param threshold 平面拟合误差阈值
 * @return true 拟合成功, false 拟合失败
 */
template<typename T>
bool esti_plane(Matrix<T, 4, 1> &pca_result, const PointVector &point, const T &threshold)
{
    // 构建线性方程组 A * x = b
    Matrix<T, NUM_MATCH_POINTS, 3> A;
    Matrix<T, NUM_MATCH_POINTS, 1> b;
    A.setZero();
    b.setOnes();
    b *= -1.0f;

    // 填充点坐标矩阵
    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        A(j,0) = point[j].x;
        A(j,1) = point[j].y;
        A(j,2) = point[j].z;
    }

    // 使用列主元QR分解求解最小二乘问题
    Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

    // 归一化得到平面参数 [A, B, C, D]
    T n = normvec.norm();
    pca_result(0) = normvec(0) / n;  // A (法向量x分量)
    pca_result(1) = normvec(1) / n;  // B (法向量y分量)
    pca_result(2) = normvec(2) / n;  // C (法向量z分量)
    pca_result(3) = 1.0 / n;         // D (平面到原点的距离)

    // 验证拟合质量: 检查所有点到平面的距离是否在阈值内
    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) > threshold)
        {
            return false;  // 拟合质量不好
        }
    }
    return true;
}
// const bool time_list(PointType &x, PointType &y); // {return (x.curvature < y.curvature);};
// template<typename T>
// const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

#endif