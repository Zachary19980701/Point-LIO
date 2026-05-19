/**
 * @file li_initialization.cpp
 * @brief 数据的预处理和同步模块
 * 文件函数说明：
 * standard_pcl_cbk： 标准的pcd2消息的回调函数，处理雷达点云数据，支持分块、合并和普通的点云处理
 * livox_pcl_cbk： 处理livox_ros_driver2::CustomMsg消息的回调函数，处理雷达点云数据，支持分块、合并和普通的点云处理
 * imu_cbk： 处理imu消息的回调函数，处理IMU数据并
 * sync_packages： 同步imu和lidar数据，保证每次处理的数据都是时间上对齐的，返回是否成功同步
 */

#include "li_initialization.h"
bool data_accum_finished = false, data_accum_start = false, online_calib_finish = false, refine_print = false;
int frame_num_init = 0;
double time_lag_IMU_wtr_lidar = 0.0, move_start_time = 0.0, online_calib_starts_time = 0.0; //, mean_acc_norm = 9.81;
double imu_first_time = 0.0;
bool lose_lid = false;
double timediff_imu_wrt_lidar = 0.0;
bool timediff_set_flg = false;
V3D gravity_lio = V3D::Zero();
mutex mtx_buffer;
sensor_msgs::Imu imu_last, imu_next;
// sensor_msgs::Imu::ConstPtr imu_last_ptr;
PointCloudXYZI::Ptr  ptr_con(new PointCloudXYZI());
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot11[MAXN];

condition_variable sig_buffer;
int scan_count = 0;
int frame_ct = 0, wait_num = 0;
std::mutex m_time;
bool lidar_pushed = false, imu_pushed = false;
std::deque<PointCloudXYZI::Ptr>  lidar_buffer;
std::deque<double>               time_buffer;
std::deque<sensor_msgs::Imu::Ptr> imu_deque;

/**
    @brief 处理标准pcd2消息的回调函数
*/
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    // mtx_buffer.lock();
    scan_count ++; //更新scan的计数器
    double preprocess_start_time = omp_get_wtime(); // 获取当前的时间，用于处理预处理的时间

    // 判断雷达的时间戳是否回退，回退返回错误信息
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        // lidar_buffer.shrink_to_fit();

        // mtx_buffer.unlock();
        // sig_buffer.notify_all();
        return;
    }

    last_timestamp_lidar = msg->header.stamp.toSec(); // 更新lidar的时间戳
    // printf("check lidar time %f\n", last_timestamp_lidar);
    // if (abs(last_timestamp_imu - last_timestamp_lidar) > 1.0 && !timediff_set_flg && !imu_deque.empty()) {
    //     timediff_set_flg = true;
    //     timediff_imu_wrt_lidar = last_timestamp_imu - last_timestamp_lidar;
    //     printf("Self sync IMU and LiDAR, HARD time lag is %.10lf \n \n", timediff_imu_wrt_lidar);
    // }

    // 根据不同的雷达类型和是否使用切帧关键词选择进行雷达点云的切帧
    if ((lidar_type == VELO16 || lidar_type == OUST64 || lidar_type == HESAIxt32 || lidar_type == LIVOX_POINTCLOUD2) && cut_frame_init) {
        deque<PointCloudXYZI::Ptr> ptr;
        deque<double> timestamp_lidar;
        p_pre->process_cut_frame_pcl2(msg, ptr, timestamp_lidar, cut_frame_num, scan_count);
        while (!ptr.empty() && !timestamp_lidar.empty()) {
            lidar_buffer.push_back(ptr.front());
            ptr.pop_front();
            time_buffer.push_back(timestamp_lidar.front() / double(1000));//unit:s
            timestamp_lidar.pop_front();
        }
    }
    else
    {
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI(20000,1));
    p_pre->process(msg, ptr);
    if (con_frame)   // 合帧处理
    {   
        // 合帧处理逻辑，能够增加随机扫描的点云稠密度
        if (frame_ct == 0)
        {
            time_con = last_timestamp_lidar; //msg->header.stamp.toSec();
        }
        if (frame_ct < 10)
        {
            for (int i = 0; i < ptr->size(); i++)
            {
                ptr->points[i].curvature += (last_timestamp_lidar - time_con) * 1000;
                ptr_con->push_back(ptr->points[i]);
            }
            frame_ct ++;
        }
        else
        {
            PointCloudXYZI::Ptr  ptr_con_i(new PointCloudXYZI(10000,1));
            // cout << "ptr div num:" << ptr_div->size() << endl;
            *ptr_con_i = *ptr_con;
            lidar_buffer.push_back(ptr_con_i);
            double time_con_i = time_con;
            time_buffer.push_back(time_con_i);
            ptr_con->clear();
            frame_ct = 0;
        }
    }
    else // 普通模式进行处理
    { 
        if (ptr->points.size() > 0)
        {
            lidar_buffer.emplace_back(ptr);
            time_buffer.emplace_back(msg->header.stamp.toSec());
        }
    }
    }
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    // mtx_buffer.unlock();
    // sig_buffer.notify_all();
}


/**
    @brief 处理livox_ros_driver2::CustomMsg消息的回调函数
*/
void livox_pcl_cbk(const livox_ros_driver2::CustomMsg::ConstPtr &msg) 
{
    // mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");

        // mtx_buffer.unlock();
        // sig_buffer.notify_all();
        return;
        // lidar_buffer.shrink_to_fit();
    }

    last_timestamp_lidar = msg->header.stamp.toSec();    
    // if (abs(last_timestamp_imu - last_timestamp_lidar) > 1.0 && !timediff_set_flg && !imu_deque.empty()) {
    //     timediff_set_flg = true;
    //     timediff_imu_wrt_lidar = last_timestamp_imu - last_timestamp_lidar;
    //     printf("Self sync IMU and LiDAR, HARD time lag is %.10lf \n \n", timediff_imu_wrt_lidar);
    // }

    if (cut_frame_init) {
        deque<PointCloudXYZI::Ptr> ptr;
        deque<double> timestamp_lidar;
        p_pre->process_cut_frame_livox(msg, ptr, timestamp_lidar, cut_frame_num, scan_count);

        while (!ptr.empty() && !timestamp_lidar.empty()) {
            lidar_buffer.push_back(ptr.front());
            ptr.pop_front();
            time_buffer.push_back(timestamp_lidar.front() / double(1000));//unit:s
            timestamp_lidar.pop_front();
        }
    }
    else
    {
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI(10000,1));
    p_pre->process(msg, ptr); 
    if (con_frame)
    {
        if (frame_ct == 0)
        {
            time_con = last_timestamp_lidar; //msg->header.stamp.toSec();
        }
        if (frame_ct < 10)
        {
            for (int i = 0; i < ptr->size(); i++)
            {
                ptr->points[i].curvature += (last_timestamp_lidar - time_con) * 1000;
                ptr_con->push_back(ptr->points[i]);
            }
            frame_ct ++;
        }
        else
        {
            PointCloudXYZI::Ptr  ptr_con_i(new PointCloudXYZI(10000,1));
            // cout << "ptr div num:" << ptr_div->size() << endl;
            *ptr_con_i = *ptr_con;
            double time_con_i = time_con;
            lidar_buffer.push_back(ptr_con_i);
            time_buffer.push_back(time_con_i);
            ptr_con->clear();
            frame_ct = 0;
        }
    }
    else
    {
        if (ptr->points.size() > 0)
        {
            lidar_buffer.emplace_back(ptr);
            time_buffer.emplace_back(msg->header.stamp.toSec());
        }
    }
    }
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    // mtx_buffer.unlock();
    // sig_buffer.notify_all();
}


/**
    @brief 处理imu消息的回调函数
*/
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    // mtx_buffer.lock();

    // publish_count ++;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    msg->header.stamp = ros::Time().fromSec(msg->header.stamp.toSec() - timediff_imu_wrt_lidar - time_lag_IMU_wtr_lidar);

    double timestamp = msg->header.stamp.toSec();
    // printf("time_diff%f, %f, %f\n", last_timestamp_imu - timestamp, last_timestamp_imu, timestamp);

    if (timestamp < last_timestamp_imu)
    {
        ROS_ERROR("imu loop back, clear deque");
        // imu_deque.shrink_to_fit();
        // cout << "check time:" << timestamp << ";" << last_timestamp_imu << endl;
        // printf("time_diff%f, %f, %f\n", last_timestamp_imu - timestamp, last_timestamp_imu, timestamp);
        
        // mtx_buffer.unlock();
        // sig_buffer.notify_all();
        return;
    }
    imu_deque.emplace_back(msg);
    last_timestamp_imu = timestamp;
    // mtx_buffer.unlock();
    // sig_buffer.notify_all();
}

/**
    @brief 同步imu和lidar数据，保证每次处理的数据都是时间上对齐的
*/
bool sync_packages(MeasureGroup &meas)
{
    {
    if (!imu_en)
    {
        if (!lidar_buffer.empty())
        {
            if (!lidar_pushed)
            {
                meas.lidar = lidar_buffer.front();
                meas.lidar_beg_time = time_buffer.front();
                lose_lid = false;
                if(meas.lidar->points.size() < 1) 
                {
                    cout << "lose lidar" << std::endl;
                    // return false;
                    lose_lid = true;
                }
                else
                {
                    double end_time = meas.lidar->points.back().curvature;
                    for (auto pt: meas.lidar->points)
                    {
                        if (pt.curvature > end_time)
                        {
                            end_time = pt.curvature;
                        }
                    }
                    lidar_end_time = meas.lidar_beg_time + end_time / double(1000);
                    meas.lidar_last_time = lidar_end_time;
                }
                lidar_pushed = true;
            }
            
            time_buffer.pop_front();
            lidar_buffer.pop_front();
            lidar_pushed = false;
            if (!lose_lid)
            {
                return true;
            }
            else
            {
                return false;
            }
        }        
        return false;
    }

    if (lidar_buffer.empty() || imu_deque.empty())
    {
        return false;
    }
    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        lose_lid = false;
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if(meas.lidar->points.size() < 1) 
        {
            cout << "lose lidar" << endl;
            lose_lid = true;
            // lidar_buffer.pop_front();
            // time_buffer.pop_front();
            // return false;
        }
        else
        {
            double end_time = meas.lidar->points.back().curvature;
            for (auto pt: meas.lidar->points)
            {
                if (pt.curvature > end_time)
                {
                    end_time = pt.curvature;
                }
            }
            lidar_end_time = meas.lidar_beg_time + end_time / double(1000);
            // cout << "check time lidar:" << end_time << endl;
            meas.lidar_last_time = lidar_end_time;
        }
        lidar_pushed = true;
    }

    if (!lose_lid && (last_timestamp_imu < lidar_end_time))
    {
        return false;
    }
    if (lose_lid && last_timestamp_imu < meas.lidar_beg_time + lidar_time_inte)
    {
        return false;
    }

    if (!lose_lid && !imu_pushed)
    { 
        /*** push imu data, and pop from imu buffer ***/
        if (p_imu->imu_need_init_)
        {
            double imu_time = imu_deque.front()->header.stamp.toSec();
            imu_next = *(imu_deque.front());
            meas.imu.shrink_to_fit();
            while (imu_time < lidar_end_time)
            {
                meas.imu.emplace_back(imu_deque.front());
                imu_last = imu_next;
                imu_deque.pop_front();
                if(imu_deque.empty()) break;
                imu_time = imu_deque.front()->header.stamp.toSec(); // can be changed
                imu_next = *(imu_deque.front());
            }
        }
        imu_pushed = true;
    }

    if (lose_lid && !imu_pushed)
    { 
        /*** push imu data, and pop from imu buffer ***/
        if (p_imu->imu_need_init_)
        {
            double imu_time = imu_deque.front()->header.stamp.toSec();
            meas.imu.shrink_to_fit();

            imu_next = *(imu_deque.front());
            while (imu_time < meas.lidar_beg_time + lidar_time_inte)
            {
                meas.imu.emplace_back(imu_deque.front());
                imu_last = imu_next;
                imu_deque.pop_front();
                if(imu_deque.empty()) break;
                imu_time = imu_deque.front()->header.stamp.toSec(); // can be changed
                imu_next = *(imu_deque.front());
            }
        }
        imu_pushed = true;
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    imu_pushed = false;
    return true;
    }
}
