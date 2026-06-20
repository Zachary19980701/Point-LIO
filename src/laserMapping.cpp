/**
 * @file laserMapping.cpp
 * @brief Point-LIO 激光雷达-惯性里程计与建图主节点 (laserMapping)
 *
 * === 整体功能 ===
 * 本节点是 Point-LIO 系统的核心，实现了一个基于 **迭代误差状态卡尔曼滤波器 (IESKF)**
 * 的实时 LiDAR-惯性里程计与增量式建图系统。主要功能包括：
 *
 *   1. **IMU 状态前向传播**：利用 IMU 测量的角速度和加速度，对系统状态（位置、姿态、速度、
 *      IMU bias、重力向量）进行先验预测。
 *   2. **点云预处理**：对 LiDAR 点云进行运动畸变校正（利用 IMU 反向传播）、空间降采样、
 *      时间戳排序和时间压缩（将相近时间戳的点合并为批次）。
 *   3. **逐点迭代卡尔曼滤波更新 (Point-by-Point IESKF)**：
 *      - 对每个特征点，利用当前估计的位姿将其投影到世界坐标系
 *      - 在 iVox (空间哈希体素栅格) 中搜索最近邻点，拟合局部平面
 *      - 以"点到平面"距离为残差，逐点进行卡尔曼滤波更新
 *      - 该方式是 Point-LIO 区别于 FAST-LIO2 的核心创新：不等待完整扫描结束，
 *        而是在每个点到达时立即进行状态更新
 *   4. **增量式建图**：将配准后的点云以体素去重的方式追加到全局 iVox 地图中
 *   5. **重定位模式**：支持加载预构建的 PCD 地图进行纯定位（不更新地图）
 *
 * === 两种工作模式 ===
 *   - use_imu_as_input = false (默认)：LiDAR 为主要输入源，使用 kf_output (30维状态)
 *   - use_imu_as_input = true：IMU 为主要输入源，使用 kf_input (24维状态)
 *
 * === 坐标帧说明 ===
 *   - Body 帧 (IMU 坐标系)
 *   - Lidar 帧 → 通过外参 Lidar_R_wrt_IMU / Lidar_T_wrt_IMU 变换到 Body 帧
 *   - World 帧 (camera_init)：以第一帧 Body 位姿为原点的世界坐标系
 *
 * === ROS 话题 ===
 *   - 订阅: LiDAR 点云 (livox/标准格式), IMU 数据, 初始位姿 (重定位模式)
 *   - 发布: /cloud_registered (世界系配准点云), /cloud_registered_body (Body 系点云),
 *           /Laser_map (地图点云), /aft_mapped_to_init (里程计位姿), /path (轨迹路径)
 */

// #include <so3_math.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include "li_initialization.h"
#include <malloc.h>
// #include <cv_bridge/cv_bridge.h>
// #include "matplotlibcpp.h"
// #include <ros/console.h>

using namespace std;

// =========================================================================
// 常量与宏定义
// =========================================================================

#define PUBFRAME_PERIOD     (20)     ///< 发布帧的周期间隔（未使用）

const float MOV_THRESHOLD = 1.5f;    ///< 运动阈值（m/s，未使用）

// =========================================================================
// 全局状态变量
// =========================================================================

string root_dir = ROOT_DIR;                    ///< 数据根目录

int time_log_counter = 0;                      ///< 时间日志计数器

bool init_map = false, flg_first_scan = true;  ///< 地图初始化标志 / 首次扫描标志

// 各阶段耗时统计 (用于性能分析)
double match_time = 0, solve_time = 0, propag_time = 0, update_time = 0;

bool flg_reset = false, flg_exit = false;      ///< 重置标志 / 退出标志

// =========================================================================
// 点云数据容器
// =========================================================================

// 运动畸变校正后的特征点云 (Body 系, 带时间戳)
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
// 降采样后的特征点云 (Body 系)
PointCloudXYZI::Ptr feats_down_body_space(new PointCloudXYZI());
// 初始化阶段累积的世界系点云
PointCloudXYZI::Ptr init_feats_world(new PointCloudXYZI());
// 深度特征点云队列（未使用）
std::deque<PointCloudXYZI::Ptr> depth_feats_world;

// 体素降采样滤波器
pcl::VoxelGrid<PointType> downSizeFilterSurf;  ///< 扫描内点云降采样
pcl::VoxelGrid<PointType> downSizeFilterMap;   ///< 地图降采样（未使用）

V3D euler_cur;                                 ///< 当前欧拉角 (调试用)

// ROS 消息对象 (复用以减少内存分配)
nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::PoseStamped msg_body_pose;

// =========================================================================
// 重定位 (Localization) 模式相关状态
// =========================================================================

PointCloudXYZI::Ptr localization_map_cloud(new PointCloudXYZI());  ///< 加载的预建地图
std::mutex localization_pose_mutex;                                 ///< 初始位姿互斥锁
V3D localization_pending_pos(Zero3d);                               ///< 待应用的位置
M3D localization_pending_rot(Eye3d);                                ///< 待应用的旋转
std::string localization_pending_source;                            ///< 位姿来源标识
bool localization_map_loaded = false;                                ///< 地图是否已加载
bool localization_pose_pending = false;                              ///< 是否有待应用位姿
bool localization_initial_pose_ready = false;                        ///< 初始位姿是否就绪


//=====================================================================
//  第1部分: 重定位模块
//  功能: 预建 PCD 地图的加载、初始位姿管理的完整流程
//  核心函数: loadLocalizationMap() + applyPendingLocalizationInitialPose()
//  数据流: RViz / params → initialPoseHandler / queueLocalizationParamsInitialPose
//          → queueLocalizationInitialPose (线程安全入队)
//          → applyPendingLocalizationInitialPose (主循环中应用)
//=====================================================================

/**
 * @brief 将 RPY 欧拉角转换为旋转矩阵
 *
 * 采用 ZYX 内旋顺序 (即: 先绕 Z 轴转 yaw, 再绕新的 Y 轴转 pitch,
 * 再绕新的 X 轴转 roll)。
 *
 * @param roll  绕 X 轴旋转角 (rad)
 * @param pitch 绕 Y 轴旋转角 (rad)
 * @param yaw   绕 Z 轴旋转角 (rad)
 * @return M3D  3x3 旋转矩阵
 */
M3D rotationFromRpy(double roll, double pitch, double yaw)
{
    Eigen::Quaterniond q =
        Eigen::AngleAxisd(yaw, V3D::UnitZ()) *
        Eigen::AngleAxisd(pitch, V3D::UnitY()) *
        Eigen::AngleAxisd(roll, V3D::UnitX());
    return q.toRotationMatrix();
}

/**
 * @brief 将初始位姿推入等待队列（线程安全）
 *
 * 此函数由 initialPoseHandler (RViz) 或 queueLocalizationParamsInitialPose (launch 参数)
 * 调用，将初始位姿写入共享变量，等待主循环通过 applyPendingLocalizationInitialPose() 应用。
 *
 * @param x     世界系 X 坐标 (m)
 * @param y     世界系 Y 坐标 (m)
 * @param z     世界系 Z 坐标 (m)
 * @param roll  初始 roll 角 (rad)
 * @param pitch 初始 pitch 角 (rad)
 * @param yaw   初始 yaw 角 (rad)
 * @param source 位姿来源标识 ("rviz" 或 "params")
 */
void queueLocalizationInitialPose(
        double x, double y, double z,
        double roll, double pitch, double yaw,
        const std::string &source)
{
    std::lock_guard<std::mutex> lock(localization_pose_mutex);
    localization_pending_pos << x, y, z;
    localization_pending_rot = rotationFromRpy(roll, pitch, yaw);
    localization_pending_source = source;
    localization_pose_pending = true;
}

/**
 * @brief 使用 ROS 参数 (localization_init_*) 设置初始位姿
 *
 * 当 localization_init_source == "params" 时调用，
 * 将 launch 文件中的定位初始位姿参数写入队列。
 *
 * @param source 位姿来源标识
 */
void queueLocalizationParamsInitialPose(const std::string &source)
{
    queueLocalizationInitialPose(
        localization_init_x, localization_init_y, localization_init_z,
        localization_init_roll, localization_init_pitch, localization_init_yaw,
        source);
}

/**
 * @brief 重置重定位初始位姿状态
 *
 * 在系统 reset 或重新加载地图时调用，清除所有待处理和已就绪的位姿标志。
 */
void resetLocalizationInitialPoseState()
{
    std::lock_guard<std::mutex> lock(localization_pose_mutex);
    localization_pose_pending = false;
    localization_initial_pose_ready = false;
    localization_pending_source.clear();
}

/**
 * @brief 查询重定位初始位姿是否已就绪 (线程安全)
 * @return true 位姿已通过 applyPendingLocalizationInitialPose() 应用
 * @return false 尚未应用初始位姿
 */
bool localizationInitialPoseReady()
{
    std::lock_guard<std::mutex> lock(localization_pose_mutex);
    return localization_initial_pose_ready;
}

/**
 * @brief 应用等待队列中的初始位姿到卡尔曼滤波器（线程安全）
 *
 * 这是重定位模式的核心操作：
 * 1. 从共享变量中取出待应用的位姿
 * 2. 同时设置 kf_input 和 kf_output 的状态（位置、旋转）
 * 3. 速度、角速度置零
 * 4. 清空历史轨迹路径
 *
 * @return true  成功应用了新的初始位姿
 * @return false 没有待应用的位姿
 */
bool applyPendingLocalizationInitialPose()
{
    V3D pos;
    M3D rot;
    std::string source;
    {
        std::lock_guard<std::mutex> lock(localization_pose_mutex);
        if (!localization_pose_pending)
        {
            return false;
        }
        pos = localization_pending_pos;
        rot = localization_pending_rot;
        source = localization_pending_source;
        localization_pose_pending = false;
        localization_initial_pose_ready = true;
    }

    // 设置 IMU 输入滤波器的状态
    kf_input.x_.pos << pos(0), pos(1), pos(2);
    kf_input.x_.rot = rot;
    kf_input.x_.vel << 0.0, 0.0, 0.0;

    // 设置 LiDAR 输出滤波器的状态
    kf_output.x_.pos << pos(0), pos(1), pos(2);
    kf_output.x_.rot = rot;
    kf_output.x_.vel << 0.0, 0.0, 0.0;
    kf_output.x_.omg << 0.0, 0.0, 0.0;

    // 清空历史路径
    path.poses.clear();
    path.header.stamp = ros::Time().fromSec(lidar_end_time);
    path.header.frame_id = "camera_init";

    ROS_INFO("Localization initial pose applied from %s: x=%.3f y=%.3f z=%.3f",
             source.c_str(), pos(0), pos(1), pos(2));
    return true;
}

/**
 * @brief ROS 回调: 接收 RViz 发送的初始位姿 (2D Pose Estimate)
 *
 * 用户通过 RViz 的 "2D Pose Estimate" 工具在地图上指定位姿，
 * 该回调被触发。从消息中提取 yaw 角和 (x, y) 位置，
 * z、roll、pitch 使用 launch 参数中的值。
 *
 * @param msg geometry_msgs::PoseWithCovarianceStamped 消息
 */
void initialPoseHandler(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr &msg)
{
    if (!localization_enable)
    {
        return;
    }

    tf::Quaternion q;
    tf::quaternionMsgToTF(msg->pose.pose.orientation, q);
    double unused_roll = 0.0, unused_pitch = 0.0, yaw = 0.0;
    tf::Matrix3x3(q).getRPY(unused_roll, unused_pitch, yaw);

    queueLocalizationInitialPose(
        msg->pose.pose.position.x,
        msg->pose.pose.position.y,
        localization_init_z,
        localization_init_roll,
        localization_init_pitch,
        yaw,
        "rviz");

    ROS_INFO("Received RViz initial pose on %s", localization_initial_pose_topic.c_str());
}

/**
 * @brief 发布重定位模式的预建地图点云
 *
 * 将 localization_map_cloud 以 PointCloud2 格式发布到 /Laser_map 话题，
 * 仅在 localization_enable 和 localization_publish_map 均为 true 时有效。
 *
 * @param pubLaserCloudMap 地图点云发布者
 */
void publishLocalizationMap(const ros::Publisher &pubLaserCloudMap)
{
    if (!localization_publish_map || localization_map_cloud->empty())
    {
        return;
    }

    sensor_msgs::PointCloud2 map_msg;
    pcl::toROSMsg(*localization_map_cloud, map_msg);
    map_msg.header.stamp = ros::Time::now();
    map_msg.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(map_msg);
}

/**
 * @brief 加载重定位模式使用的预构建 PCD 地图
 *
 * === 完整的数据处理流水线 ===
 * 1. 从 localization_map_path 读取 PCD 文件
 * 2. 过滤 NaN 点
 * 3. 可选体素降采样 (localization_map_voxel_size)
 * 4. 将 pcl::PointXYZI 转换为 PointType (添加 normal/curvature 字段)
 * 5. 将点加载到 iVox 空间哈希栅格 (ivox_->AddPoints)
 * 6. 设置 init_map = true 以跳过建图初始化阶段
 *
 * @param pubLaserCloudMap 地图发布者（用于发布加载的地图供可视化确认）
 * @return true  地图加载成功（或不需要重定位模式）
 * @return false 地图加载失败（文件不存在/为空/过滤后无有效点）
 */
bool loadLocalizationMap(const ros::Publisher &pubLaserCloudMap)
{
    // 非重定位模式直接返回成功
    if (!localization_enable)
    {
        return true;
    }

    // === 步骤1: 加载原始 PCD 文件 ===
    pcl::PointCloud<pcl::PointXYZI>::Ptr raw_map(new pcl::PointCloud<pcl::PointXYZI>());
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(localization_map_path, *raw_map) < 0)
    {
        ROS_ERROR("Failed to load localization map PCD: %s", localization_map_path.c_str());
        return false;
    }
    if (raw_map->empty())
    {
        ROS_ERROR("Localization map PCD is empty: %s", localization_map_path.c_str());
        return false;
    }

    // === 步骤2: 剔除 NaN 点 ===
    pcl::PointCloud<pcl::PointXYZI>::Ptr finite_map(new pcl::PointCloud<pcl::PointXYZI>());
    std::vector<int> finite_indices;
    pcl::removeNaNFromPointCloud(*raw_map, *finite_map, finite_indices);

    // === 步骤3: 可选体素降采样 ===
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_xyzi(new pcl::PointCloud<pcl::PointXYZI>());
    if (localization_map_voxel_size > 0.0)
    {
        pcl::VoxelGrid<pcl::PointXYZI> map_filter;
        map_filter.setLeafSize(localization_map_voxel_size, localization_map_voxel_size, localization_map_voxel_size);
        map_filter.setInputCloud(finite_map);
        map_filter.filter(*filtered_xyzi);
    }
    else
    {
        *filtered_xyzi = *finite_map;
    }

    if (filtered_xyzi->empty())
    {
        ROS_ERROR("Localization map has no valid points after filtering: %s", localization_map_path.c_str());
        return false;
    }

    // === 步骤4: 类型转换 pcl::PointXYZI → PointType ===
    PointCloudXYZI::Ptr filtered_map(new PointCloudXYZI());
    filtered_map->reserve(filtered_xyzi->size());
    for (const auto &point : filtered_xyzi->points)
    {
        PointType converted;
        converted.x = point.x;
        converted.y = point.y;
        converted.z = point.z;
        converted.intensity = point.intensity;
        converted.normal_x = 0.0;
        converted.normal_y = 0.0;
        converted.normal_z = 0.0;
        converted.curvature = 0.0;
        filtered_map->push_back(converted);
    }

    // === 步骤5: 检查 iVox 容量 ===
    if (filtered_map->size() > ivox_options_.capacity_)
    {
        ROS_WARN("Localization map points (%zu) exceed ivox capacity (%zu); oldest grids may be dropped.",
                 filtered_map->size(), ivox_options_.capacity_);
    }

    // === 步骤6: 加载到 iVox 空间哈希栅格 ===
    PointVector map_points;
    map_points.reserve(filtered_map->size());
    for (const auto &point : filtered_map->points)
    {
        map_points.emplace_back(point);
    }
    ivox_->AddPoints(map_points);

    // === 步骤7: 设置状态标志 ===
    localization_map_cloud = filtered_map;
    localization_map_loaded = true;
    init_map = true;  // 跳过建图初始化阶段

    publishLocalizationMap(pubLaserCloudMap);
    ROS_INFO("Loaded localization map: raw=%zu filtered=%zu ivox_grids=%zu path=%s",
             raw_map->size(), localization_map_cloud->size(), ivox_->NumValidGrids(), localization_map_path.c_str());
    return true;
}


//=====================================================================
//  第2部分: 信号处理与状态日志
//=====================================================================

/**
 * @brief 信号处理函数 — 优雅退出
 *
 * 捕获 SIGINT (Ctrl+C) 信号，设置退出标志，
 * 通知所有等待在 sig_buffer 上的线程。
 *
 * @param sig 信号编号
 */
void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

/**
 * @brief 将当前 LIO 系统状态写入日志文件
 *
 * 根据 use_imu_as_input 选择输出 kf_output 或 kf_input 的状态。
 * 日志格式 (空格分隔):
 *   相对时间 roll pitch yaw px py pz [omega] vx vy vz [acc] bgx bgy bgz bax bay baz gravity_x gravity_y gravity_z
 *
 * @param fp 已打开的日志文件指针
 */
inline void dump_lio_state_to_log(FILE *fp)
{
    V3D rot_ang;
    if (!use_imu_as_input)
    {
        rot_ang = SO3ToEuler(kf_output.x_.rot);
    }
    else
    {
        rot_ang = SO3ToEuler(kf_input.x_.rot);
    }

    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    if (use_imu_as_input)
    {
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.pos(0), kf_input.x_.pos(1), kf_input.x_.pos(2)); // Pos
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.vel(0), kf_input.x_.vel(1), kf_input.x_.vel(2)); // Vel
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.bg(0), kf_input.x_.bg(1), kf_input.x_.bg(2));    // Bias_g
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.ba(0), kf_input.x_.ba(1), kf_input.x_.ba(2));    // Bias_a
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.gravity(0), kf_input.x_.gravity(1), kf_input.x_.gravity(2)); // Bias_a
    }
    else
    {
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.pos(0), kf_output.x_.pos(1), kf_output.x_.pos(2)); // Pos
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.vel(0), kf_output.x_.vel(1), kf_output.x_.vel(2)); // Vel
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.bg(0), kf_output.x_.bg(1), kf_output.x_.bg(2));    // Bias_g
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.ba(0), kf_output.x_.ba(1), kf_output.x_.ba(2));    // Bias_a
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.gravity(0), kf_output.x_.gravity(1), kf_output.x_.gravity(2)); // Bias_a
    }
    fprintf(fp, "\r\n");
    fflush(fp);
}


//=====================================================================
//  第3部分: 坐标变换工具
//=====================================================================

/**
 * @brief 将点从 LiDAR 坐标系转换到 IMU (Body) 坐标系
 *
 * 支持两种外参获取方式：
 *   1. extrinsic_est_en = true: 使用卡尔曼滤波器在线估计的外参
 *      (kf_output.x_.offset_R_L_I / offset_T_L_I)
 *   2. extrinsic_est_en = false: 使用配置文件中固定的外参
 *      (Lidar_R_wrt_IMU / Lidar_T_wrt_IMU)
 *
 * p_body_imu = R * p_body_lidar + T
 *
 * @param pi 输入点 (LiDAR 系)
 * @param po 输出点 (IMU/Body 系) — 仅修改 x,y,z; intensity 直接拷贝
 */
void pointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu;
    if (extrinsic_est_en)
    {
        if (!use_imu_as_input)
        {
            p_body_imu = kf_output.x_.offset_R_L_I * p_body_lidar + kf_output.x_.offset_T_L_I;
        }
        else
        {
            p_body_imu = kf_input.x_.offset_R_L_I * p_body_lidar + kf_input.x_.offset_T_L_I;
        }
    }
    else
    {
        p_body_imu = Lidar_R_wrt_IMU * p_body_lidar + Lidar_T_wrt_IMU;
    }
    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}


//=====================================================================
//  第4部分: 地图管理
//=====================================================================

/**
 * @brief 增量式地图更新 — 将当前帧的特征点插入全局 iVox 地图
 *
 * === 算法原理 ===
 * 该函数执行基于体素的去重式地图插入：
 * 1. 遍历当前帧在世界系中的特征点 feats_down_world
 * 2. 对每个点，查找其在 iVox 中的最近邻集合 (Nearest_Points[i])
 * 3. 计算该点所在体素的中心坐标
 * 4. 如果该体素内已存在足够近的点，则跳过（去重）
 * 5. 否则将该点标记为需要添加到地图
 * 6. 批量调用 ivox_->AddPoints() 将所有新点插入 iVox 栅格
 *
 * === 为什么需要去重 ===
 * - iVox 栅格的存储容量有限 (ivox_options_.capacity_)
 * - 避免同一空间位置重复存储大量点云，浪费内存和查询时间
 * - 去重粒度为 filter_size_map_min (默认约 0.1~0.5m)
 *
 * === 重定位模式说明 ===
 * 当 localization_enable = true 时此函数被跳过，
 * 因为重定位模式下不应修改预建地图。
 */
void MapIncremental() {
    PointVector points_to_add;
    int cur_pts = feats_down_world->size();
    points_to_add.reserve(cur_pts);

    for (size_t i = 0; i < cur_pts; ++i) {
        /* 判断该点是否需要加入地图 */
        PointType &point_world = feats_down_world->points[i];
        if (!Nearest_Points[i].empty()) {
            const PointVector &points_near = Nearest_Points[i];

            // 计算该点所在体素的中心坐标
            Eigen::Vector3f center =
                ((point_world.getVector3fMap() / filter_size_map_min).array().floor() + 0.5) * filter_size_map_min;
            bool need_add = true;
            // 检查体素内是否已有足够近的邻近点
            for (int readd_i = 0; readd_i < points_near.size(); readd_i++) {
                Eigen::Vector3f dis_2_center = points_near[readd_i].getVector3fMap() - center;
                if (fabs(dis_2_center.x()) < 0.5 * filter_size_map_min &&
                    fabs(dis_2_center.y()) < 0.5 * filter_size_map_min &&
                    fabs(dis_2_center.z()) < 0.5 * filter_size_map_min) {
                    need_add = false;
                    break;
                }
            }
            if (need_add) {
                points_to_add.emplace_back(point_world);
            }
        } else {
            // 该点附近没有任何地图点，直接添加
            points_to_add.emplace_back(point_world);
        }
    }
    // 批量插入 iVox 空间哈希栅格
    ivox_->AddPoints(points_to_add);
}

/**
 * @brief 发布初始化阶段构建的初始地图
 *
 * 当累积足够的初始帧点云 (init_feats_world 达到 init_map_size) 后调用。
 * 将 init_feats_world 发布为 PointCloud2 消息。
 *
 * @param pubLaserCloudFullRes 地图点云发布者
 */
void publish_init_map(const ros::Publisher & pubLaserCloudFullRes)
{
    int size_init_map = init_feats_world->size();

    sensor_msgs::PointCloud2 laserCloudmsg;

    pcl::toROSMsg(*init_feats_world, laserCloudmsg);

    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudFullRes.publish(laserCloudmsg);
}


//=====================================================================
//  第5部分: 数据发布
//  功能: 世界系点云、Body 系点云、里程计位姿、运动轨迹的发布
//=====================================================================

// 发布和保存用的点云缓冲区（预分配以减少内存分配）
PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());

/**
 * @brief 发布世界坐标系下的配准点云 + 可选 PCD 保存
 *
 * 功能1: 将 feats_down_world (世界系) 发布到 /cloud_registered
 * 功能2: 如果 pcd_save_en = true, 每 pcd_save_interval 帧将点云保存为 PCD 文件
 *
 * @param pubLaserCloudFullRes 世界点云发布者
 */
void publish_frame_world(const ros::Publisher & pubLaserCloudFullRes)
{
    // === 发布世界系点云 ===
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(feats_down_body);
        int size = laserCloudFullRes->points.size();

        PointCloudXYZI::Ptr   laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            laserCloudWorld->points[i].x = feats_down_world->points[i].x;
            laserCloudWorld->points[i].y = feats_down_world->points[i].y;
            laserCloudWorld->points[i].z = feats_down_world->points[i].z;
            laserCloudWorld->points[i].intensity = feats_down_world->points[i].intensity; // feats_down_world->points[i].y; //
        }
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);

        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFullRes.publish(laserCloudmsg);
        // publish_count -= PUBFRAME_PERIOD;
    }

    // === 可选 PCD 文件保存 ===
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_down_world->points.size();
        PointCloudXYZI::Ptr   laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            laserCloudWorld->points[i].x = feats_down_world->points[i].x;
            laserCloudWorld->points[i].y = feats_down_world->points[i].y;
            laserCloudWorld->points[i].z = feats_down_world->points[i].z;
            laserCloudWorld->points[i].intensity = feats_down_world->points[i].intensity;
        }

        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

/**
 * @brief 发布 Body (IMU) 坐标系下的点云
 *
 * 将 feats_undistort 通过 pointBodyLidarToIMU() 从 LiDAR 系转到 IMU 系后发布到
 * /cloud_registered_body。用于调试和可视化传感器安装关系。
 *
 * @param pubLaserCloudFull_body Body 系点云发布者
 */
void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        pointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    // publish_count -= PUBFRAME_PERIOD;
}

/**
 * @brief 将当前卡尔曼滤波器状态写入 ROS Pose 消息 (模板函数)
 *
 * 根据 use_imu_as_input 标志选择读取 kf_output 或 kf_input 的状态。
 * 将旋转矩阵转换为 Eigen::Quaterniond 四元数格式。
 *
 * @tparam T 支持 .position 和 .orientation 的 ROS 消息类型 (如 geometry_msgs::Pose)
 * @param out 输出 ROS Pose 消息 (引用)
 */
template<typename T>
void set_posestamp(T & out)
{
    if (!use_imu_as_input)
    {
        out.position.x = kf_output.x_.pos(0);
        out.position.y = kf_output.x_.pos(1);
        out.position.z = kf_output.x_.pos(2);
        Eigen::Quaterniond q(kf_output.x_.rot);
        out.orientation.x = q.coeffs()[0];
        out.orientation.y = q.coeffs()[1];
        out.orientation.z = q.coeffs()[2];
        out.orientation.w = q.coeffs()[3];
    }
    else
    {
        out.position.x = kf_input.x_.pos(0);
        out.position.y = kf_input.x_.pos(1);
        out.position.z = kf_input.x_.pos(2);
        Eigen::Quaterniond q(kf_input.x_.rot);
        out.orientation.x = q.coeffs()[0];
        out.orientation.y = q.coeffs()[1];
        out.orientation.z = q.coeffs()[2];
        out.orientation.w = q.coeffs()[3];
    }
}

/**
 * @brief 发布里程计位姿 (/aft_mapped_to_init) 和 TF 变换
 *
 * 将当前最优状态估计发布为 nav_msgs::Odometry 消息，同时通过 TF 广播
 * camera_init → body 的坐标变换。
 *
 * @param pubOdomAftMapped 里程计发布者
 */
void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    if (publish_odometry_without_downsample)
    {
        odomAftMapped.header.stamp = ros::Time().fromSec(time_current);
    }
    else
    {
        odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
    }
    set_posestamp(odomAftMapped.pose.pose);

    pubOdomAftMapped.publish(odomAftMapped);

    // === 同时广播 TF 变换 ===
    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body") );
}

/**
 * @brief 发布运动轨迹路径 (/path)
 *
 * 在每个处理帧将当前位姿追加到 path.poses 并发布。
 * 注意：路径会持续增长，在长时间运行中可能占用大量内存。
 *
 * @param pubPath 路径发布者
 */
void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose.pose);
    // msg_body_pose.header.stamp = ros::Time::now();
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";
    static int jjj = 0;
    jjj++;
    // if (jjj % 2 == 0) // if path is too large, the rvis will crash
    {
        path.poses.emplace_back(msg_body_pose);
        pubPath.publish(path);
    }
}


//=====================================================================
//  第6部分: 主函数 (main)
//  功能: ROS 节点初始化、卡尔曼滤波器设置、主循环运行
//
//  主循环内部的数据流:
//  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐
//  │ sync_    │───→│ 点云预处理    │───→│ IESKF 逐点   │───→│ 增量建图  │
//  │ packages │    │ (畸变+降采样) │    │ 预测+更新     │    │          │
//  └──────────┘    └──────────────┘    └─────────────┘    └──────────┘
//                                                              ↓
//                                        ┌──────────────────────┘
//                                        ↓
//                              发布里程计/点云/路径
//=====================================================================

int main(int argc, char** argv)
{
    // =====================================================================
    // 6a. ROS 节点和参数初始化
    // =====================================================================
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh("~");
    ros::AsyncSpinner spinner(0);  ///< 多线程异步 spinner (线程数=CPU核数)
    spinner.start();
    readParameters(nh);
    cout<<"lidar_type: "<<lidar_type<<endl;

    // 构建 iVox (空间哈希体素栅格) 用于高效地图查询
    ivox_ = std::make_shared<IVoxType>(ivox_options_);

    path.header.stamp    = ros::Time().fromSec(lidar_end_time);
    path.header.frame_id ="camera_init";

    // 帧计数器与各阶段平均耗时累积变量
    int frame_num = 0;
    double aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_propag = 0;

    // =====================================================================
    // 6b. 卡尔曼滤波器初始化
    // =====================================================================

    // 初始化点选择标志
    memset(point_selected_surf, true, sizeof(point_selected_surf));

    // 体素降采样滤波器参数设置
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

    // 加载 LiDAR→IMU 外参
    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);

    // 如果启用在线外参估计，将初始外参赋值给滤波器状态
    if (extrinsic_est_en)
    {
        if (!use_imu_as_input)
        {
            kf_output.x_.offset_R_L_I = Lidar_R_wrt_IMU;
            kf_output.x_.offset_T_L_I = Lidar_T_wrt_IMU;
        }
        else
        {
            kf_input.x_.offset_R_L_I = Lidar_R_wrt_IMU;
            kf_input.x_.offset_T_L_I = Lidar_T_wrt_IMU;
        }
    }

    // 设置 LiDAR 类型和 IMU 使能标志
    p_imu->lidar_type = p_pre->lidar_type = lidar_type;
    p_imu->imu_en = imu_en;

    // 初始化两个卡尔曼滤波器的系统模型
    // kf_input:  24维状态 (IMU-中心模型, use_imu_as_input=true 时使用)
    // kf_output: 30维状态 (LiDAR-中心模型, use_imu_as_input=false 时使用)
    kf_input.init_dyn_share_modified_2h(get_f_input, df_dx_input, h_model_input);
    kf_output.init_dyn_share_modified_3h(get_f_output, df_dx_output, h_model_output, h_model_IMU_output);

    // 初始协方差矩阵设置
    Eigen::Matrix<double, 24, 24> P_init;
    reset_cov(P_init);
    kf_input.change_P(P_init);
    Eigen::Matrix<double, 30, 30> P_init_output;
    reset_cov_output(P_init_output);
    kf_output.change_P(P_init_output);

    // 过程噪声协方差矩阵 (来自 IMU 噪声参数)
    Eigen::Matrix<double, 24, 24> Q_input = process_noise_cov_input();
    Eigen::Matrix<double, 30, 30> Q_output = process_noise_cov_output();

    // =====================================================================
    // 6b附加: 调试日志文件初始化
    // =====================================================================
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");
    open_file();

    // =====================================================================
    // 6c. ROS 发布者/订阅者注册
    // =====================================================================

    // --- 订阅者 ---
    // LiDAR 点云: 根据 lidar_type 选择 AVIA (Livox) 或标准回调
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    // IMU 数据
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    // 重定位模式下的 RViz 初始位姿
    ros::Subscriber sub_initial_pose;
    if (localization_enable)
    {
        sub_initial_pose = nh.subscribe(localization_initial_pose_topic, 10, initialPoseHandler);
    }

    // --- 发布者 ---
    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 1000);          ///< 世界系配准点云
    ros::Publisher pubLaserCloudFullRes_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 1000);     ///< Body 系配准点云
    // ros::Publisher pubLaserCloudEffect  = nh.advertise<sensor_msgs::PointCloud2>
            // ("/cloud_effected", 1000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 1000, localization_enable && localization_publish_map);  ///< 地图点云
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>
            ("/aft_mapped_to_init", 1000);        ///< 里程计位姿
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path>
            ("/path", 1000);                      ///< 运动轨迹
    // ros::Publisher plane_pub = nh.advertise<visualization_msgs::Marker>
            // ("/planner_normal", 1000);

    // =====================================================================
    // 6d. 重定位模式设置
    // =====================================================================
    if (localization_enable)
    {
        // 加载预建 PCD 地图到 iVox
        if (!loadLocalizationMap(pubLaserCloudMap))
        {
            return 1;  // 地图加载失败，退出
        }
        // 根据配置决定立即使用参数位姿还是等待 RViz 交互
        if (localization_init_source == "params" || !localization_wait_for_initial_pose)
        {
            queueLocalizationParamsInitialPose("params");
        }
        else
        {
            ROS_WARN("Localization map loaded. Waiting for RViz initial pose on %s",
                     localization_initial_pose_topic.c_str());
        }
    }

//------------------------------------------------------------------------------------------------------
    // =====================================================================
    // 6e. 主处理循环 (500Hz)
    // =====================================================================
    signal(SIGINT, SigHandle);
    ros::Rate loop_rate(500);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit) break;
        ros::spinOnce();

        // --- 等待并同步 LiDAR 和 IMU 数据包 ---
        if(sync_packages(Measures))
        {
            // =============================================================
            // 6e1. 重置处理 (rosbag 回放时触发)
            // =============================================================
            if (flg_reset)
            {
                ROS_WARN("reset when rosbag play back");
                p_imu->Reset();                         // IMU 处理器重置
                feats_undistort.reset(new PointCloudXYZI());  // 清空点云

                // 重置对应模式下的卡尔曼滤波器状态和协方差
                if (use_imu_as_input)
                {
                    state_in = state_input();
                    kf_input.change_P(P_init);
                }
                else
                {
                    state_out = state_output();
                    kf_output.change_P(P_init_output);
                }

                // 重置所有流程控制标志
                flg_first_scan = true;
                is_first_frame = true;
                flg_reset = false;
                init_map = false;

                // 重建 iVox 地图
                {
                    ivox_.reset(new IVoxType(ivox_options_));
                }

                // 重定位模式：重新加载地图和初始位姿
                if (localization_enable)
                {
                    resetLocalizationInitialPoseState();
                    if (!loadLocalizationMap(pubLaserCloudMap))
                    {
                        flg_exit = true;
                        break;
                    }
                    if (localization_init_source == "params" || !localization_wait_for_initial_pose)
                    {
                        queueLocalizationParamsInitialPose("params");
                    }
                    else
                    {
                        ROS_WARN("Localization reset. Waiting for RViz initial pose on %s",
                                 localization_initial_pose_topic.c_str());
                    }
                }
            }

            // =============================================================
            // 6e2. 首帧初始化 — IMU 时间对齐与重力初始化
            // =============================================================
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;  // 记录首帧时间戳

                flg_first_scan = false;
                if (first_imu_time < 1)
                {
                    first_imu_time = imu_next.header.stamp.toSec();
                    printf("first imu time: %f\n", first_imu_time);
                }
                time_current = 0.0;

                if(imu_en)
                {
                    // IMU 使能时：设置重力向量初值，对齐 IMU 队列时间
                    kf_input.x_.gravity << VEC_FROM_ARRAY(gravity);
                    kf_output.x_.gravity << VEC_FROM_ARRAY(gravity);

                    // 丢弃早于首帧 LiDAR 时间的 IMU 数据
                    {
                        while (Measures.lidar_beg_time > imu_next.header.stamp.toSec())
                        {
                            imu_deque.pop_front();
                            if (imu_deque.empty())
                            {
                                break;
                            }
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                        }
                    }
                }
                else
                {
                    // 无 IMU 模式：从配置读取重力并标记 IMU 初始化完成
                    kf_input.x_.gravity << VEC_FROM_ARRAY(gravity);
                    kf_output.x_.gravity << VEC_FROM_ARRAY(gravity);
                    kf_output.x_.acc << VEC_FROM_ARRAY(gravity);
                    kf_output.x_.acc *= -1;
                    p_imu->imu_need_init_ = false;
                }
                // 计算重力模长
                G_m_s2 = std::sqrt(gravity[0] * gravity[0] + gravity[1] * gravity[1] + gravity[2] * gravity[2]);
            }

            // --- 初始化各阶段计时变量 ---
            double t0,t1,t2,t3,t4,t5,match_start, solve_start;
            match_time = 0;
            solve_time = 0;
            propag_time = 0;
            update_time = 0;
            t0 = omp_get_wtime();

            // =============================================================
            // 6e3. 点云预处理: 运动畸变校正 → 降采样 → 时间排序 → 时间压缩
            // =============================================================
            t1 = omp_get_wtime();

            // p_imu->Process() 执行:
            //   - 反向传播: 利用 IMU 积分对每个点进行运动畸变校正
            //   - 将点转换到 scan 结束时刻的 Body 坐标系
            //   - 输出: feats_undistort (畸变校正后的点云)
            p_imu->Process(Measures, feats_undistort);

            if(space_down_sample)
            {
                // 空间体素降采样 (filter_size_surf_min 体素大小)
                downSizeFilterSurf.setInputCloud(feats_undistort);
                downSizeFilterSurf.filter(*feats_down_body);
                sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);
            }
            else
            {
                // 跳过降采样，使用原始 LiDAR 点
                feats_down_body = Measures.lidar;
                sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);
            }

            // 时间压缩: 将相近时间戳的点合并为批次 (按时间间隔阈值分组)
            {
                time_seq = time_compressing<int>(feats_down_body);
                feats_down_size = feats_down_body->points.size();
            }

            // =============================================================
            // 6e4. IMU 初始化 — 重力对齐与初始姿态估计
            // =============================================================
            if (!p_imu->after_imu_init_)
            {
                if (!p_imu->imu_need_init_)
                {
                    V3D tmp_gravity;
                    if (imu_en)
                    {
                        // 使用 IMU 静止期间的加速度均值估计重力方向
                        tmp_gravity = - p_imu->mean_acc / p_imu->mean_acc.norm() * G_m_s2;
                    }
                    else
                    {
                        tmp_gravity << VEC_FROM_ARRAY(gravity_init);
                        p_imu->after_imu_init_ = true;
                    }
                    // 计算将重力对齐到世界系 Z 轴的旋转矩阵
                    M3D rot_init;
                    p_imu->Set_init(tmp_gravity, rot_init);
                    kf_input.x_.rot = rot_init;
                    kf_output.x_.rot = rot_init;
                    kf_output.x_.acc = - rot_init.transpose() * kf_output.x_.gravity;
                }
                else{
                continue;}  // IMU 尚未完成静止初始化，跳过当前帧
            }

            // =============================================================
            // 6e5. 重定位初始位姿应用
            // =============================================================
            if (localization_enable)
            {
                applyPendingLocalizationInitialPose();
                if (localization_wait_for_initial_pose && !localizationInitialPoseReady())
                {
                    ROS_WARN_THROTTLE(5.0, "Waiting for localization initial pose on %s",
                                      localization_initial_pose_topic.c_str());
                    continue;  // 等待用户在 RViz 中指定初始位姿
                }
            }

            // =============================================================
            // 6e6. 地图初始化 — 累积足够的初始帧后建立地图
            // =============================================================
            if(!init_map)
            {
                // 将畸变校正后的点转到世界系
                feats_down_world->resize(feats_undistort->size());
                for(int i = 0; i < feats_undistort->size(); i++)
                {
                    pointBodyToWorld(&(feats_undistort->points[i]), &(feats_down_world->points[i]));
                }

                // 累积到 init_feats_world
                for (size_t i = 0; i < feats_down_world->size(); i++)
                {
                    init_feats_world->points.emplace_back(feats_down_world->points[i]);
                }

                // 检查是否达到初始化所需的点数阈值
                if(init_feats_world->size() < init_map_size)
                {
                    init_map = false;  // 继续累积
                }
                else
                {
                    // 将累积的点插入 iVox 地图，完成初始化
                    ivox_->AddPoints(init_feats_world->points);
                    publish_init_map(pubLaserCloudMap);

                    init_feats_world.reset(new PointCloudXYZI());  // 释放内存
                    init_map = true;
                }
                continue;  // 初始化阶段不执行 IESKF 更新
            }

            // =============================================================
            // 6e7 & 6e8: IESKF 逐点迭代更新
            //
            // 两种模式共享的前置准备工作:
            //   - 分配 nearest points 容器
            //   - 预计算每个点的 Body 系坐标和叉乘矩阵 (用于雅可比)
            // =============================================================
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            t2 = omp_get_wtime();

            // 预计算: 将每个点转到 IMU/Body 系，并构建叉乘矩阵
            crossmat_list.resize(feats_down_size);
            pbody_list.resize(feats_down_size);

            for (size_t i = 0; i < feats_down_body->size(); i++)
            {
                V3D point_this(feats_down_body->points[i].x,
                            feats_down_body->points[i].y,
                            feats_down_body->points[i].z);
                pbody_list[i]=point_this;
                if (!extrinsic_est_en)
                {
                    // 使用固定外参: p_body = R * p_lidar + T
                    point_this = Lidar_R_wrt_IMU * point_this + Lidar_T_wrt_IMU;
                    M3D point_crossmat;
                    point_crossmat << SKEW_SYM_MATRX(point_this);  // 反对称矩阵，用于旋转雅可比
                    crossmat_list[i]=point_crossmat;
                }
            }

            // =============================================================
            // 6e7. 分支A: LiDAR 为主要输入 (use_imu_as_input = false)
            //       使用 kf_output (30维状态)
            //
            //       逐点处理流程:
            //       for each time_batch:
            //         1. IMU 时间推进 (consuming IMU data up to point's timestamp)
            //         2. 状态预测 predict(dt)
            //         3. 协方差预测 predict(dt_cov, update=false)
            //         4. 迭代卡尔曼更新 update_iterated_dyn_share_modified()
            //            - 搜索最近邻点 (iVox KNN)
            //            - 拟合局部平面
            //            - 计算点到平面残差
            //            - 迭代求解状态增量
            //         5. 将处理后的点投影到世界系
            // =============================================================
            if (!use_imu_as_input)
            {
                bool imu_upda_cov = false;
                effct_feat_num = 0;

                if (time_seq.size() > 0)
                {
                double pcl_beg_time = Measures.lidar_beg_time;
                idx = -1;
                for (k = 0; k < time_seq.size(); k++)
                {
                    PointType &point_body  = feats_down_body->points[idx+time_seq[k]];
                    // 当前批次的参考时间 (点的 curvature 字段存储相对时间, ms → s)
                    time_current = point_body.curvature / 1000.0 + pcl_beg_time;

                    // --- 首帧: 对齐 IMU 队列并初始化平均角速度/加速度 ---
                    if (is_first_frame)
                    {
                        if(imu_en)
                        {
                            while (time_current > imu_next.header.stamp.toSec())
                            {
                                imu_deque.pop_front();
                                if (imu_deque.empty()) break;
                                imu_last = imu_next;
                                imu_next = *(imu_deque.front());
                            }
                            angvel_avr<<imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                            acc_avr   <<imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                        }
                        is_first_frame = false;
                        imu_upda_cov = true;
                        time_update_last = time_current;
                        time_predict_last_const = time_current;
                    }

                    // --- IMU 时间推进: 消费早于当前点时间的所有 IMU 测量 ---
                    if(imu_en && !imu_deque.empty())
                    {
                        bool last_imu = imu_next.header.stamp.toSec() == imu_deque.front()->header.stamp.toSec();
                        // 丢弃过时的 IMU (时间在 predict 基准之前)
                        while (imu_next.header.stamp.toSec() < time_predict_last_const && !imu_deque.empty())
                        {
                            if (!last_imu)
                            {
                                imu_last = imu_next;
                                imu_next = *(imu_deque.front());
                                break;
                            }
                            else
                            {
                                imu_deque.pop_front();
                                if (imu_deque.empty()) break;
                                imu_last = imu_next;
                                imu_next = *(imu_deque.front());
                            }
                        }
                        // 消费位于当前点时间之前的 IMU 测量, 进行协方差传播+IMU更新
                        bool imu_comes = time_current > imu_next.header.stamp.toSec();
                        while (imu_comes)
                        {
                            imu_upda_cov = true;
                            angvel_avr<<imu_next.angular_velocity.x, imu_next.angular_velocity.y, imu_next.angular_velocity.z;
                            acc_avr   <<imu_next.linear_acceleration.x, imu_next.linear_acceleration.y, imu_next.linear_acceleration.z;

                            // 协方差传播 (仅更新协方差, 不更新状态)
                            double dt = imu_next.header.stamp.toSec() - time_predict_last_const;
                            kf_output.predict(dt, Q_output, input_in, true, false);
                            time_predict_last_const = imu_next.header.stamp.toSec();

                            {
                                double dt_cov = imu_next.header.stamp.toSec() - time_update_last;
                                if (dt_cov > 0.0)
                                {
                                    time_update_last = imu_next.header.stamp.toSec();
                                    double propag_imu_start = omp_get_wtime();
                                    kf_output.predict(dt_cov, Q_output, input_in, false, true);
                                    propag_time += omp_get_wtime() - propag_imu_start;

                                    double solve_imu_start = omp_get_wtime();
                                    kf_output.update_iterated_dyn_share_IMU();
                                    solve_time += omp_get_wtime() - solve_imu_start;
                                }
                            }
                            imu_deque.pop_front();
                            if (imu_deque.empty()) break;
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                            imu_comes = time_current > imu_next.header.stamp.toSec();
                        }
                    }
                    if (flg_reset) { break; }

                    // --- 状态前向传播到当前点的时间 ---
                    double dt = time_current - time_predict_last_const;
                    double propag_state_start = omp_get_wtime();
                    if(!prop_at_freq_of_imu)
                    {
                        double dt_cov = time_current - time_update_last;
                        if (dt_cov > 0.0)
                        {
                            kf_output.predict(dt_cov, Q_output, input_in, false, true);
                            time_update_last = time_current;
                        }
                    }
                    kf_output.predict(dt, Q_output, input_in, true, false);
                    propag_time += omp_get_wtime() - propag_state_start;
                    time_predict_last_const = time_current;

                    // --- 逐点 IESKF 迭代更新 (核心) ---
                    double t_update_start = omp_get_wtime();
                    if (feats_down_size < 1)
                    {
                        ROS_WARN("No point, skip this scan!\n");
                        idx += time_seq[k];
                        continue;
                    }
                    if (!kf_output.update_iterated_dyn_share_modified())
                    {
                        idx = idx+time_seq[k];
                        continue;  // 迭代不收敛, 跳过当前批次
                    }
                    solve_start = omp_get_wtime();

                    // --- 逐点发布里程计 (无降采样模式) ---
                    if (publish_odometry_without_downsample)
                    {
                        publish_odometry(pubOdomAftMapped);
                        if (runtime_pos_log)
                        {
                            euler_cur = SO3ToEuler(kf_output.x_.rot);
                            fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << kf_output.x_.pos.transpose() << " " << kf_output.x_.vel.transpose() \
                            <<" "<<kf_output.x_.omg.transpose()<<" "<<kf_output.x_.acc.transpose()<<" "<<kf_output.x_.gravity.transpose()<<" "<<kf_output.x_.bg.transpose()<<" "<<kf_output.x_.ba.transpose()<<" "<<feats_undistort->points.size()<<endl;
                        }
                    }

                    // --- 将当前批次的点投影到世界系 (使用更新后的位姿) ---
                    for (int j = 0; j < time_seq[k]; j++)
                    {
                        PointType &point_body_j  = feats_down_body->points[idx+j+1];
                        PointType &point_world_j = feats_down_world->points[idx+j+1];
                        pointBodyToWorld(&point_body_j, &point_world_j);
                    }

                    solve_time += omp_get_wtime() - solve_start;
                    update_time += omp_get_wtime() - t_update_start;
                    idx += time_seq[k];
                }
                }
                else
                {
                    // 降采样后没有有效时间批次, 仅推进 IMU
                    if (!imu_deque.empty())
                    {
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());

                    while (imu_next.header.stamp.toSec() > time_current && ((imu_next.header.stamp.toSec() < Measures.lidar_beg_time + lidar_time_inte )))
                    {
                        if (is_first_frame)
                        {
                            {
                                // 首帧: 跳过初始化窗口内的 IMU
                                {
                                    while (imu_next.header.stamp.toSec() < Measures.lidar_beg_time + lidar_time_inte)
                                    {
                                        imu_deque.pop_front();
                                        if(imu_deque.empty()) break;
                                        imu_last = imu_next;
                                        imu_next = *(imu_deque.front());
                                    }
                                }
                                break;
                            }
                            angvel_avr<<imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                            acc_avr   <<imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;

                            imu_upda_cov = true;
                            time_update_last = time_current;
                            time_predict_last_const = time_current;
                            is_first_frame = false;
                        }
                        time_current = imu_next.header.stamp.toSec();

                        if (!is_first_frame)
                        {
                        double dt = time_current - time_predict_last_const;
                        {
                            double dt_cov = time_current - time_update_last;
                            if (dt_cov > 0.0)
                            {
                                kf_output.predict(dt_cov, Q_output, input_in, false, true);
                                time_update_last = time_current;
                            }
                            kf_output.predict(dt, Q_output, input_in, true, false);
                        }
                        time_predict_last_const = time_current;

                        angvel_avr<<imu_next.angular_velocity.x, imu_next.angular_velocity.y, imu_next.angular_velocity.z;
                        acc_avr   <<imu_next.linear_acceleration.x, imu_next.linear_acceleration.y, imu_next.linear_acceleration.z;
                        kf_output.update_iterated_dyn_share_IMU();
                        imu_deque.pop_front();
                        if (imu_deque.empty()) break;
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                    }
                    else
                    {
                        imu_deque.pop_front();
                        if (imu_deque.empty()) break;
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                    }
                    }
                    }
                }
            }
            else
            // =============================================================
            // 6e8. 分支B: IMU 为主要输入 (use_imu_as_input = true)
            //       使用 kf_input (24维状态)
            //
            //       与分支A的核心区别:
            //       - IMU 测量作为系统输入 (predict 的 control input)
            //       - 使用 kf_input 进行预测和更新
            //       - IMU 积分在 predict 内部完成
            // =============================================================
            {
                bool imu_prop_cov = false;
                effct_feat_num = 0;
                if (time_seq.size() > 0)
                {
                double pcl_beg_time = Measures.lidar_beg_time;
                idx = -1;
                for (k = 0; k < time_seq.size(); k++)
                {
                    PointType &point_body  = feats_down_body->points[idx+time_seq[k]];
                    time_current = point_body.curvature / 1000.0 + pcl_beg_time;

                    if (is_first_frame)
                    {
                        while (time_current > imu_next.header.stamp.toSec())
                        {
                            imu_deque.pop_front();
                            if (imu_deque.empty()) break;
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                        }
                        imu_prop_cov = true;

                        is_first_frame = false;
                        t_last = time_current;
                        time_update_last = time_current;
                        {
                            input_in.gyro<<imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                            input_in.acc<<imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                            input_in.acc = input_in.acc * G_m_s2 / acc_norm;  // 加速度归一化
                        }
                    }

                    // --- IMU 时间推进 (消费早于当前点的 IMU) ---
                    while (time_current > imu_next.header.stamp.toSec())
                    {
                        imu_deque.pop_front();

                        input_in.gyro<<imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                        input_in.acc <<imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                        input_in.acc    = input_in.acc * G_m_s2 / acc_norm;
                        double dt = imu_last.header.stamp.toSec() - t_last;

                        double dt_cov = imu_last.header.stamp.toSec() - time_update_last;
                        if (dt_cov > 0.0)
                        {
                            kf_input.predict(dt_cov, Q_input, input_in, false, true);
                            time_update_last = imu_last.header.stamp.toSec();
                        }
                        kf_input.predict(dt, Q_input, input_in, true, false);
                        t_last = imu_last.header.stamp.toSec();
                        imu_prop_cov = true;

                        if (imu_deque.empty()) break;
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                    }
                    if (flg_reset) { break; }

                    // --- 状态传播到当前点时间 ---
                    double dt = time_current - t_last;
                    t_last = time_current;
                    double propag_start = omp_get_wtime();

                    if(!prop_at_freq_of_imu)
                    {
                        double dt_cov = time_current - time_update_last;
                        if (dt_cov > 0.0)
                        {
                            kf_input.predict(dt_cov, Q_input, input_in, false, true);
                            time_update_last = time_current;
                        }
                    }
                    kf_input.predict(dt, Q_input, input_in, true, false);
                    propag_time += omp_get_wtime() - propag_start;

                    // --- 逐点 IESKF 迭代更新 (kf_input 版本) ---
                    double t_update_start = omp_get_wtime();

                    if (feats_down_size < 1)
                    {
                        ROS_WARN("No point, skip this scan!\n");
                        idx += time_seq[k];
                        continue;
                    }
                    if (!kf_input.update_iterated_dyn_share_modified())
                    {
                        idx = idx+time_seq[k];
                        continue;
                    }

                    solve_start = omp_get_wtime();

                    // --- 逐点发布里程计 ---
                    if (publish_odometry_without_downsample)
                    {
                        publish_odometry(pubOdomAftMapped);
                        if (runtime_pos_log)
                        {
                            euler_cur = SO3ToEuler(kf_input.x_.rot);
                            fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << kf_input.x_.pos.transpose() << " " << kf_input.x_.vel.transpose() \
                            <<" "<<kf_input.x_.bg.transpose()<<" "<<kf_input.x_.ba.transpose()<<" "<<kf_input.x_.gravity.transpose()<<" "<<feats_undistort->points.size()<<endl;
                        }
                    }

                    // --- 将点投影到世界系 ---
                    for (int j = 0; j < time_seq[k]; j++)
                    {
                        PointType &point_body_j  = feats_down_body->points[idx+j+1];
                        PointType &point_world_j = feats_down_world->points[idx+j+1];
                        pointBodyToWorld(&point_body_j, &point_world_j);
                    }
                    solve_time += omp_get_wtime() - solve_start;

                    update_time += omp_get_wtime() - t_update_start;
                    idx = idx + time_seq[k];
                }
                }
                else
                {
                    // 无时间批次: 仅推进 IMU (与分支A逻辑对称)
                    if (!imu_deque.empty())
                    {
                    imu_last = imu_next;
                    imu_next = *(imu_deque.front());
                    while (imu_next.header.stamp.toSec() > time_current && ((imu_next.header.stamp.toSec() < Measures.lidar_beg_time + lidar_time_inte)))
                    {
                        if (is_first_frame)
                        {
                            {
                                {
                                    while (imu_next.header.stamp.toSec() < Measures.lidar_beg_time + lidar_time_inte)
                                    {
                                        imu_deque.pop_front();
                                        if(imu_deque.empty()) break;
                                        imu_last = imu_next;
                                        imu_next = *(imu_deque.front());
                                    }
                                }
                                break;
                            }
                            imu_prop_cov = true;
                            t_last = time_current;
                            time_update_last = time_current;
                            input_in.gyro<<imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                            input_in.acc   <<imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                            input_in.acc = input_in.acc * G_m_s2 / acc_norm;
                            is_first_frame = false;
                        }
                        time_current = imu_next.header.stamp.toSec();

                        if (!is_first_frame)
                        {
                        double dt = time_current - t_last;
                        double dt_cov = time_current - time_update_last;
                        if (dt_cov > 0.0)
                        {
                            time_update_last = imu_next.header.stamp.toSec();
                        }
                        t_last = imu_next.header.stamp.toSec();

                        input_in.gyro<<imu_next.angular_velocity.x, imu_next.angular_velocity.y, imu_next.angular_velocity.z;
                        input_in.acc<<imu_next.linear_acceleration.x, imu_next.linear_acceleration.y, imu_next.linear_acceleration.z;
                        input_in.acc = input_in.acc * G_m_s2 / acc_norm;
                        imu_deque.pop_front();
                        if (imu_deque.empty()) break;
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                        }
                        else
                        {
                            imu_deque.pop_front();
                            if (imu_deque.empty()) break;
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                        }
                    }
                    }
                }
            }

            // =============================================================
            // 6e9. 降采样模式下的里程计发布
            //
            // publish_odometry_without_downsample = true 时,
            // 每次 IESKF 更新后立即发布 (已在 6e7/6e8 内部完成)
            // publish_odometry_without_downsample = false 时,
            // 每帧仅在扫描结束后发布一次 (在此处发布)
            // =============================================================
            if (!publish_odometry_without_downsample)
            {
                publish_odometry(pubOdomAftMapped);
            }

            // =============================================================
            // 6e10. 增量地图更新 (非重定位模式)
            //
            // 将配准后的 feats_down_world 按体素去重后插入全局 iVox 地图。
            // 重定位模式 (localization_enable=true) 下跳过, 不改动预建地图。
            // feats_down_size > 4 作为最小有效点数检查。
            // =============================================================
            t3 = omp_get_wtime();

            if(!localization_enable && feats_down_size > 4)
            {
                MapIncremental();
            }

            t5 = omp_get_wtime();

            // =============================================================
            // 6e11. 帧数据发布
            // =============================================================
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFullRes);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFullRes_body);

            // =============================================================
            // 6e12. 性能日志 — 移动平均统计各阶段耗时
            //
            // 统计项:
            //   - t5 - t0: 总耗时 (整体)
            //   - t1 - t0: 点云预处理 (畸变校正)
            //   - t3 - t1: ICP + IESKF 求解 + 地图增量 (solve + map)
            //   - t5 - t3: 地图增量 (单独统计)
            //   - propagate_time / solve_time / update_time: 分别从内部累积得到
            // =============================================================
            if (runtime_pos_log)
            {
                frame_num ++;
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                {aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + update_time/frame_num;}
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + solve_time/frame_num;
                aver_time_propag = aver_time_propag * (frame_num - 1)/frame_num + propag_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = aver_time_consu;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f propogate: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu, aver_time_icp, aver_time_propag);
                if (!publish_odometry_without_downsample)
                {
                    if (!use_imu_as_input)
                    {
                        euler_cur = SO3ToEuler(kf_output.x_.rot);
                        fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << kf_output.x_.pos.transpose() << " " << kf_output.x_.vel.transpose() \
                        <<" "<<kf_output.x_.omg.transpose()<<" "<<kf_output.x_.acc.transpose()<<" "<<kf_output.x_.gravity.transpose()<<" "<<kf_output.x_.bg.transpose()<<" "<<kf_output.x_.ba.transpose()<<" "<<feats_undistort->points.size()<<endl;
                    }
                    else
                    {
                        euler_cur = SO3ToEuler(kf_input.x_.rot);
                        fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << kf_input.x_.pos.transpose() << " " << kf_input.x_.vel.transpose() \
                        <<" "<<kf_input.x_.bg.transpose()<<" "<<kf_input.x_.ba.transpose()<<" "<<kf_input.x_.gravity.transpose()<<" "<<feats_undistort->points.size()<<endl;
                    }
                }
                dump_lio_state_to_log(fp);
            }
        }
        status = ros::ok();
        loop_rate.sleep();
    }

    // =====================================================================
    // 6f. 退出 — 保存累积的 PCD 文件
    // =====================================================================
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }
    fout_out.close();
    fout_imu_pbp.close();
    return 0;
}
