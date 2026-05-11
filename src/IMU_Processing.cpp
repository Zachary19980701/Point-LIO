/**
 * @file IMU_Processing.cpp
 * @brief IMU数据处理模块 - 对应论文Section III-A 系统初始化
 *
 * 本文件实现了:
 * 1. IMU初始化: 静态条件下估计重力方向和传感器零偏
 * 2. 初始姿态确定: 根据重力向量计算初始旋转矩阵
 */

#include "IMU_Processing.h"

/**
 * @brief 点时间戳比较函数,用于排序
 */
const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_vel_scale = scaler;
}

/**
 * @brief 构造函数
 */
ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true)
{
  imu_en = true;
  init_iter_num = 1;
  mean_acc      = V3D(0, 0, 0.0);  // 平均加速度初始化
  mean_gyr      = V3D(0, 0, 0);    // 平均角速度初始化
  after_imu_init_ = false;
  state_cov.setIdentity();
}

ImuProcess::~ImuProcess() {}

/**
 * @brief 重置IMU处理器状态
 */
void ImuProcess::Reset()
{
  ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, 0.0);
  mean_gyr      = V3D(0, 0, 0);
  imu_need_init_    = true;
  init_iter_num     = 1;
  after_imu_init_   = false;

  time_last_scan = 0.0;
}

/**
 * @brief 设置初始姿态 - 根据重力向量确定初始旋转
 *
 * 对应论文Section III-A: 系统初始化
 *
 * 算法原理:
 * 1. 静止状态下,加速度计测量的是重力加速度的反方向
 * 2. 通过将测量到的重力向量与期望的重力方向对齐,可以得到初始姿态
 *
 * 数学推导:
 *   设测量重力为 g_meas, 期望重力为 g_ref
 *   需要找到旋转 R 使得 R * g_meas = g_ref
 *   使用向量对齐方法:
 *     旋转轴: n = g_meas × g_ref
 *     旋转角: θ = arccos(g_meas · g_ref / (|g_meas| * |g_ref|))
 *     旋转矩阵: R = Exp(θ * n)
 *
 * @param tmp_gravity 测量到的重力向量 (加速度计平均值的负方向)
 * @param rot 输出的初始旋转矩阵
 */
void ImuProcess::Set_init(Eigen::Vector3d &tmp_gravity, Eigen::Matrix3d &rot)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/

  // 构建参考重力的反对称矩阵,用于计算叉乘
  M3D hat_grav;
  hat_grav << 0.0, gravity_(2), -gravity_(1),
              -gravity_(2), 0.0, gravity_(0),
              gravity_(1), -gravity_(0), 0.0;

  // 计算对齐向量的范数 (用于判断是否平行或反向平行)
  double align_norm = (hat_grav * tmp_gravity).norm() / gravity_.norm() / tmp_gravity.norm();

  // 计算对齐向量的点积 (余弦值)
  double align_cos = gravity_.transpose() * tmp_gravity;
  align_cos = align_cos / gravity_.norm() / tmp_gravity.norm();

  // 特殊情况处理
  if (align_norm < 1e-6)  // 向量平行或反向平行
  {
    if (align_cos > 1e-6)  // 平行: 不需要旋转
    {
      rot = Eye3d;
    }
    else  // 反向平行: 旋转180度
    {
      rot = -Eye3d;
    }
  }
  else
  {
    // 一般情况: 使用Rodrigues公式计算旋转矩阵
    // 旋转角度向量 = 旋转轴 × 旋转角
    V3D align_angle = hat_grav * tmp_gravity / (hat_grav * tmp_gravity).norm() * acos(align_cos);

    // 使用指数映射将旋转向量转换为旋转矩阵
    rot = Exp(align_angle(0), align_angle(1), align_angle(2));
  }
}

/**
 * @brief IMU初始化 - 估计重力方向和传感器零偏
 *
 * 对应论文Section III-A: 系统初始化
 *
 * 初始化过程:
 * 1. 在静止状态下累积IMU测量数据
 * 2. 使用滑动平均估计:
 *    - 平均加速度 -> 重力方向 (静止时加速度计测量的是重力反方向)
 *    - 平均角速度 -> 陀螺仪零偏 (静止时角速度应为0)
 * 3. 当累积足够多的数据后 (N > MAX_INI_COUNT), 初始化完成
 *
 * 注意事项:
 * - 初始化期间系统应保持静止
 * - 初始化时间越长,估计越准确
 *
 * @param meas 当前测量数据组
 * @param N 累积的IMU数据计数
 */
void ImuProcess::IMU_init(const MeasureGroup &meas, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/

  ROS_INFO("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);
  V3D cur_acc, cur_gyr;

  // 第一帧特殊处理: 重置状态并初始化均值
  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;

    // 使用第一个IMU测量初始化均值
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
  }

  // 遍历当前测量组中的所有IMU数据
  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    // 使用滑动平均更新均值
    // 公式: mean = mean + (new_value - mean) / N
    // 这等价于: mean = (mean * (N-1) + new_value) / N
    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    N++;
  }
}

/**
 * @brief IMU数据处理主入口
 *
 * 功能:
 * 1. 在初始化阶段: 调用IMU_init()累积数据估计重力方向
 * 2. 初始化完成后: 返回原始点云数据供后续处理
 *
 * 注意: Point-LIO的IMU预积分是在laserMapping.cpp中逐点进行的,
 *       此函数主要负责初始化阶段的数据处理
 *
 * @param meas 测量数据组 (包含LiDAR和IMU数据)
 * @param cur_pcl_un_ 输出的点云数据
 */
void ImuProcess::Process(const MeasureGroup &meas, PointCloudXYZI::Ptr cur_pcl_un_)
{
  if (imu_en)
  {
    if(meas.imu.empty())  return;

    // ==================== IMU初始化阶段 ====================
    if (imu_need_init_)
    {
      {
        /// The very first lidar frame
        // 调用初始化函数累积IMU数据
        IMU_init(meas, init_iter_num);

        imu_need_init_ = true;

        // 检查是否累积了足够的数据
        if (init_iter_num > MAX_INI_COUNT)
        {
          ROS_INFO("IMU Initializing: %.1f %%", 100.0);
          imu_need_init_ = false;  // 初始化完成
          *cur_pcl_un_ = *(meas.lidar);
        }
      }
      return;
    }

    // ==================== 初始化完成后的处理 ====================
    if (!after_imu_init_) after_imu_init_ = true;

    // 返回原始点云 (点云去畸变在laserMapping.cpp中通过状态插值完成)
    *cur_pcl_un_ = *(meas.lidar);
    return;
  }
  else
  {
    // IMU未启用: 直接返回原始点云
    *cur_pcl_un_ = *(meas.lidar);
    return;
  }
}