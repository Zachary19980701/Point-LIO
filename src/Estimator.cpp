/**
 * @file Estimator.cpp
 * @brief Point-LIO 状态估计器实现 - 论文核心算法对应
 *
 * 本文件实现了Point-LIO的状态估计核心算法,包括:
 * 1. 状态转移方程 (论文公式3)
 * 2. 状态雅可比矩阵 (论文公式5)
 * 3. 测量方程 - 点到面距离模型 (论文公式8)
 * 4. 过程噪声协方差 (论文公式4)
 */

#include "Estimator.h"

// ==================== 全局变量定义 ====================
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));  // 平面法向量存储
std::vector<int> time_seq;  // 时间序列压缩后的索引
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI(10000, 1));   // Body坐标系下的降采样点云
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI(10000, 1));  // World坐标系下的降采样点云
std::vector<V3D> pbody_list;  // 点在Body坐标系下的坐标列表
std::vector<PointVector> Nearest_Points;  // 每个点的最近邻点集
std::shared_ptr<IVoxType> ivox_ = nullptr;  // iVox增量体素地图 (论文Section IV)
std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);  // 最近邻搜索距离
bool point_selected_surf[100000] = {0};  // 点是否被选为有效特征点
std::vector<M3D> crossmat_list;  // 点坐标的反对称矩阵列表 (用于雅可比计算)
int effct_feat_num = 0;  // 有效特征点数量
int k = 0;  // 时间序列索引
int idx = -1;  // 点云索引

// 两个卡尔曼滤波器实例:
// kf_input: 输入模式,状态向量24维 (论文Section III-B)
// kf_output: 输出模式,状态向量30维 (论文Section III-B)
esekfom::esekf<state_input, 24, input_ikfom> kf_input;
esekfom::esekf<state_output, 30, input_ikfom> kf_output;

input_ikfom input_in;  // 输入向量 (加速度, 角速度)
V3D angvel_avr, acc_avr, acc_avr_norm;  // IMU测量平均值
int feats_down_size = 0;  // 降采样后点云大小
V3D Lidar_T_wrt_IMU(Zero3d);  // LiDAR到IMU的平移外参
M3D Lidar_R_wrt_IMU(Eye3d);   // LiDAR到IMU的旋转外参
double G_m_s2 = 9.81;  // 重力加速度常量

/**
 * @brief 输入模式过程噪声协方差矩阵
 *
 * 对应论文公式(4): Q = diag(σ_g², σ_a², σ_bg², σ_ba²)
 *
 * 噪声模型:
 * - 陀螺仪噪声 n_g: 影响姿态估计精度
 * - 加速度计噪声 n_a: 影响速度估计精度
 * - 陀螺仪零偏随机游走 n_bg: 陀螺仪零偏的缓慢漂移
 * - 加速度计零偏随机游走 n_ba: 加速度计零偏的缓慢漂移
 *
 * @return 24×24 过程噪声协方差矩阵
 */
Eigen::Matrix<double, 24, 24> process_noise_cov_input()
{
	Eigen::Matrix<double, 24, 24> cov;
	cov.setZero();

	// 状态向量索引 (输入模式):
	// [0:3]   pos     - 位置
	// [3:6]   rot     - 姿态 (SO3)
	// [6:9]   offset_R_L_I - LiDAR-IMU旋转外参
	// [9:12]  offset_T_L_I - LiDAR-IMU平移外参
	// [12:15] vel     - 速度
	// [15:18] bg      - 陀螺仪零偏
	// [18:21] ba      - 加速度计零偏
	// [21:24] gravity - 重力向量

	// 陀螺仪噪声协方差 (影响姿态,索引3-6)
	cov.block<3, 3>(3, 3).diagonal() << gyr_cov_input, gyr_cov_input, gyr_cov_input;

	// 加速度计噪声协方差 (影响速度,索引12-15)
	cov.block<3, 3>(12, 12).diagonal() << acc_cov_input, acc_cov_input, acc_cov_input;

	// 陀螺仪零偏随机游走 (索引15-18)
	cov.block<3, 3>(15, 15).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;

	// 加速度计零偏随机游走 (索引18-21)
	cov.block<3, 3>(18, 18).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;

	return cov;
}

/**
 * @brief 输出模式过程噪声协方差矩阵
 *
 * 输出模式状态向量比输入模式多了角速度(omg)和加速度(acc)
 *
 * @return 30×30 过程噪声协方差矩阵
 */
Eigen::Matrix<double, 30, 30> process_noise_cov_output()
{
	Eigen::Matrix<double, 30, 30> cov;
	cov.setZero();

	// 状态向量索引 (输出模式):
	// [0:3]   pos     - 位置
	// [3:6]   rot     - 姿态 (SO3)
	// [6:9]   offset_R_L_I - LiDAR-IMU旋转外参
	// [9:12]  offset_T_L_I - LiDAR-IMU平移外参
	// [12:15] vel     - 速度
	// [15:18] omg     - 角速度 (输出模式特有)
	// [18:21] acc     - 加速度 (输出模式特有)
	// [21:24] gravity - 重力向量
	// [24:27] bg      - 陀螺仪零偏
	// [27:30] ba      - 加速度计零偏

	// 速度噪声协方差
	cov.block<3, 3>(12, 12).diagonal() << vel_cov, vel_cov, vel_cov;

	// 角速度噪声协方差
	cov.block<3, 3>(15, 15).diagonal() << gyr_cov_output, gyr_cov_output, gyr_cov_output;

	// 加速度噪声协方差
	cov.block<3, 3>(18, 18).diagonal() << acc_cov_output, acc_cov_output, acc_cov_output;

	// 陀螺仪零偏随机游走
	cov.block<3, 3>(24, 24).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;

	// 加速度计零偏随机游走
	cov.block<3, 3>(27, 27).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;

	return cov;
}

/**
 * @brief 输入模式状态转移方程 f(x,u)
 *
 * 对应论文公式(3): 状态微分方程
 *
 * 状态方程:
 *   ṗ = v                              (位置导数 = 速度)
 *   Ṙ = ω                              (姿态导数 = 角速度, SO3流形上)
 *   v̇ = R(a - ba) + g                  (速度导数 = 惯性加速度 + 重力)
 *
 * 其中:
 *   p: 位置, R: 姿态矩阵, v: 速度
 *   ω: 角速度 (陀螺仪测量 - 零偏)
 *   a: 加速度 (加速度计测量 - 零偏)
 *   g: 重力向量
 *   bg, ba: 陀螺仪/加速度计零偏
 *
 * @param s 当前状态
 * @param in 输入 (加速度, 角速度)
 * @return 状态导数向量 (24维)
 */
Eigen::Matrix<double, 24, 1> get_f_input(state_input &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();

	// 计算真实角速度: ω = ω_meas - bg
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);

	// 计算惯性系下的加速度: a_inertial = R * (a_meas - ba)
	vect3 a_inertial = s.rot * (in.acc - s.ba);

	// 构建状态导数向量
	for(int i = 0; i < 3; i++ ){
		res(i) = s.vel[i];                    // ṗ = v (位置导数)
		res(i + 3) = omega[i];                // Ṙ = ω (姿态导数)
		res(i + 12) = a_inertial[i] + s.gravity[i];  // v̇ = R(a-ba) + g (速度导数)
	}
	return res;
}

Eigen::Matrix<double, 30, 1> get_f_output(state_output &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 30, 1> res = Eigen::Matrix<double, 30, 1>::Zero();
	vect3 a_inertial = s.rot * s.acc; // .normalized()
	for(int i = 0; i < 3; i++ ){
		res(i) = s.vel[i];
		res(i + 3) = s.omg[i]; 
		res(i + 12) = a_inertial[i] + s.gravity[i]; 
	}
	return res;
}

/**
 * @brief 输入模式状态雅可比矩阵 ∂f/∂x
 *
 * 对应论文公式(5): F矩阵 (状态转移矩阵)
 *
 * 用于协方差传播: P = FPF^T + Q
 *
 * 主要的非零块:
 *   ∂ṗ/∂v = I           : 位置对速度的雅可比
 *   ∂v̇/∂R = -R[a]×     : 速度对姿态的雅可比 (旋转向量的反对称矩阵)
 *   ∂v̇/∂ba = -R        : 速度对加速度计零偏的雅可比
 *   ∂v̇/∂g = I          : 速度对重力的雅可比
 *   ∂ω/∂bg = -I         : 角速度对陀螺仪零偏的雅可比
 *
 * @param s 当前状态
 * @param in 输入
 * @return 24×24 状态雅可比矩阵
 */
Eigen::Matrix<double, 24, 24> df_dx_input(state_input &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 24> cov = Eigen::Matrix<double, 24, 24>::Zero();

	// ∂ṗ/∂v = I (位置导数对速度的雅可比)
	cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();

	// 计算去偏后的加速度
	vect3 acc_;
	in.acc.boxminus(acc_, s.ba);
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);

	// ∂v̇/∂R = -R * [a]× (速度导数对姿态的雅可比)
	// [a]× 是加速度的反对称矩阵,用于表示叉乘
	cov.template block<3, 3>(12, 3) = -s.rot * MTK::hat(acc_);

	// ∂v̇/∂ba = -R (速度导数对加速度计零偏的雅可比)
	cov.template block<3, 3>(12, 18) = -s.rot;

	// ∂v̇/∂g = I (速度导数对重力的雅可比)
	cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity();

	// ∂ω/∂bg = -I (角速度对陀螺仪零偏的雅可比)
	cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity();

	return cov;
}

Eigen::Matrix<double, 30, 30> df_dx_output(state_output &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 30, 30> cov = Eigen::Matrix<double, 30, 30>::Zero();
	cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
	cov.template block<3, 3>(12, 3) = -s.rot*MTK::hat(s.acc); // .normalized().toRotationMatrix()
	cov.template block<3, 3>(12, 18) = s.rot; //.normalized().toRotationMatrix();
	// Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
	// Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
	// s.S2_Mx(grav_matrix, vec, 21);
	cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity(); // grav_matrix; 
	cov.template block<3, 3>(3, 15) = Eigen::Matrix3d::Identity(); 
	return cov;
}

/**
 * @brief 输入模式测量模型 - 点到平面距离
 *
 * 对应论文公式(8): 测量方程
 *
 * 测量模型:
 *   h(x) = n^T * (R*p_b + t) + d = 0
 *
 * 其中:
 *   n: 平面法向量 (通过最近邻点拟合得到)
 *   d: 平面到原点的距离
 *   p_b: 点在Body坐标系下的坐标
 *   R, t: 当前姿态和位置估计
 *
 * 测量残差:
 *   z = n^T * p_w + d
 *
 * 测量雅可比矩阵:
 *   H = [∂h/∂p, ∂h/∂R, ∂h/∂R_LI, ∂h/∂t_LI]
 *
 * @param s 当前状态
 * @param cov_p 位置协方差 (未使用)
 * @param cov_R 姿态协方差 (未使用)
 * @param ekfom_data 输出的测量数据结构
 */
void h_model_input(state_input &s, Eigen::Matrix3d cov_p, Eigen::Matrix3d cov_R, esekfom::dyn_share_modified<double> &ekfom_data)
{
	bool match_in_map = false;
	VF(4) pabcd;  // 平面参数 [A, B, C, D]
	pabcd.setZero();
	normvec->resize(time_seq[k]);
	int effect_num_k = 0;  // 当前时间组的有效特征点数

	// ==================== 步骤1: 遍历当前时间组的所有点 ====================
	for (int j = 0; j < time_seq[k]; j++)
	{
		PointType &point_body_j  = feats_down_body->points[idx+j+1];
		PointType &point_world_j = feats_down_world->points[idx+j+1];

		// 将点从Body坐标系变换到World坐标系
		pointBodyToWorld(&point_body_j, &point_world_j);

		V3D p_body = pbody_list[idx+j+1];
		double p_norm = p_body.norm();  // 点到传感器中心的距离
		V3D p_world;
		p_world << point_world_j.x, point_world_j.y, point_world_j.z;

		{
			// ==================== 步骤2: 在iVox地图中搜索最近邻点 ====================
			auto &points_near = Nearest_Points[idx+j+1];
			ivox_->GetClosestPoint(point_world_j, points_near, NUM_MATCH_POINTS);

			// 检查最近邻点数量是否足够
			if ((points_near.size() < NUM_MATCH_POINTS))
			{
				point_selected_surf[idx+j+1] = false;
			}
			else
			{
				point_selected_surf[idx+j+1] = false;

				// ==================== 步骤3: 使用最近邻点拟合局部平面 ====================
				// 求解平面方程: Ax + By + Cz + D = 0
				if (esti_plane(pabcd, points_near, plane_thr))
				{
					// ==================== 步骤4: 计算点到平面的距离 ====================
					// pd2 = |A*x + B*y + C*z + D| / sqrt(A²+B²+C²)
					// 由于esti_plane已经归一化,所以 pd2 = |A*x + B*y + C*z + D|
					float pd2 = fabs(pabcd(0) * point_world_j.x + pabcd(1) * point_world_j.y + pabcd(2) * point_world_j.z + pabcd(3));

					// ==================== 步骤5: 特征点有效性检验 ====================
					// 条件: 点到传感器的距离 > match_s * (点到平面距离)²
					// 这是为了过滤掉距离传感器太近或匹配质量不好的点
					if (p_norm > match_s * pd2 * pd2)
					{
						point_selected_surf[idx+j+1] = true;

						// 保存平面参数 (法向量 + 距离)
						normvec->points[j].x = pabcd(0);  // A (法向量x分量)
						normvec->points[j].y = pabcd(1);  // B (法向量y分量)
						normvec->points[j].z = pabcd(2);  // C (法向量z分量)
						normvec->points[j].intensity = pabcd(3);  // D (平面距离)

						effect_num_k++;
					}
				}
			}
		}
	}

	// 如果没有有效特征点,标记测量无效
	if (effect_num_k == 0)
	{
		ekfom_data.valid = false;
		return;
	}

	// ==================== 步骤6: 构建测量雅可比矩阵和残差 ====================
	ekfom_data.M_Noise = laser_point_cov;  // 测量噪声
	ekfom_data.h_x.resize(effect_num_k, 12);  // 测量雅可比矩阵 (每行12维,对应位置+姿态+外参)
	ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_num_k, 12);
	ekfom_data.z.resize(effect_num_k);  // 测量残差
	int m = 0;

	for (int j = 0; j < time_seq[k]; j++)
	{
		if(point_selected_surf[idx+j+1])
		{
			// 获取平面法向量
			V3D norm_vec(normvec->points[j].x, normvec->points[j].y, normvec->points[j].z);

			// ==================== 步骤7: 计算测量雅可比矩阵 ====================
			// 测量方程: h(x) = n^T * (R*p_b + t) + d
			// 雅可比矩阵: H = [∂h/∂t, ∂h/∂R, ∂h/∂R_LI, ∂h/∂t_LI]

			if (extrinsic_est_en)  // 如果估计外参
			{
				// 外参估计模式下的雅可比计算
				V3D p_body = pbody_list[idx+j+1];
				M3D p_crossmat, p_imu_crossmat;
				p_crossmat << SKEW_SYM_MATRX(p_body);
				V3D point_imu = s.offset_R_L_I * p_body + s.offset_T_L_I;
				p_imu_crossmat << SKEW_SYM_MATRX(point_imu);
				V3D C(s.rot.transpose() * norm_vec);
				V3D A(p_imu_crossmat * C);  // ∂h/∂R
				V3D B(p_crossmat * s.offset_R_L_I.transpose() * C);  // ∂h/∂R_LI

				// 雅可比矩阵: [n^T, A, B, C] 对应 [位置, 姿态, 外参旋转, 外参平移]
				ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
			}
			else  // 外参固定模式
			{
				// 使用预计算的反对称矩阵
				M3D point_crossmat = crossmat_list[idx+j+1];
				V3D C(s.rot.transpose() * norm_vec);
				V3D A(point_crossmat * C);  // ∂h/∂R

				// 雅可比矩阵: [n^T, A, 0, 0] (外参部分为0)
				ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
			}

			// ==================== 步骤8: 计算测量残差 ====================
			// z = -(n^T * p_w + d) = -h(x)
			ekfom_data.z(m) = -norm_vec(0) * feats_down_world->points[idx+j+1].x
			                - norm_vec(1) * feats_down_world->points[idx+j+1].y
			                - norm_vec(2) * feats_down_world->points[idx+j+1].z
			                - normvec->points[j].intensity;

			m++;
		}
	}
	effct_feat_num += effect_num_k;
}

void h_model_output(state_output &s, Eigen::Matrix3d cov_p, Eigen::Matrix3d cov_R, esekfom::dyn_share_modified<double> &ekfom_data)
{
	bool match_in_map = false;
	VF(4) pabcd;
	pabcd.setZero();
	normvec->resize(time_seq[k]);
	int effect_num_k = 0;
	for (int j = 0; j < time_seq[k]; j++)
	{
		PointType &point_body_j  = feats_down_body->points[idx+j+1];
		PointType &point_world_j = feats_down_world->points[idx+j+1];
		pointBodyToWorld(&point_body_j, &point_world_j); 
		V3D p_body = pbody_list[idx+j+1];
		double p_norm = p_body.norm();
		V3D p_world;
		p_world << point_world_j.x, point_world_j.y, point_world_j.z;
		{
			auto &points_near = Nearest_Points[idx+j+1];
			
            ivox_->GetClosestPoint(point_world_j, points_near, NUM_MATCH_POINTS); // 
			
			if ((points_near.size() < NUM_MATCH_POINTS)) // || pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5)
			{
				point_selected_surf[idx+j+1] = false;
			}
			else
			{
				point_selected_surf[idx+j+1] = false;
				if (esti_plane(pabcd, points_near, plane_thr)) //(planeValid)
				{
					float pd2 = fabs(pabcd(0) * point_world_j.x + pabcd(1) * point_world_j.y + pabcd(2) * point_world_j.z + pabcd(3));
					// V3D norm_vec;
					// M3D Rpf, pf;
					// pf = crossmat_list[idx+j+1];
					// // pf << SKEW_SYM_MATRX(p_body);
					// Rpf = s.rot * pf;
					// norm_vec << pabcd(0), pabcd(1), pabcd(2);
					// double noise_state = norm_vec.transpose() * (cov_p+Rpf*cov_R*Rpf.transpose())  * norm_vec + sqrt(p_norm) * 0.001;
					// // if (p_norm > match_s * pd2 * pd2)
					// double epsilon = pd2 / sqrt(noise_state);
					// double weight = 1.0; // epsilon / sqrt(epsilon * epsilon+1);
					// if (epsilon > 1.0) 
					// {
					// 	weight = sqrt(2 * epsilon - 1) / epsilon;
					// 	pabcd(0) = weight * pabcd(0);
					// 	pabcd(1) = weight * pabcd(1);
					// 	pabcd(2) = weight * pabcd(2);
					// 	pabcd(3) = weight * pabcd(3);
					// }
					if (p_norm > match_s * pd2 * pd2)
					{
						// point_selected_surf[i] = true;
						point_selected_surf[idx+j+1] = true;
						normvec->points[j].x = pabcd(0);
						normvec->points[j].y = pabcd(1);
						normvec->points[j].z = pabcd(2);
						normvec->points[j].intensity = pabcd(3);
						effect_num_k ++;
					}
				}  
			}
		}
	}
	if (effect_num_k == 0) 
	{
		ekfom_data.valid = false;
		return;
	}
	ekfom_data.M_Noise = laser_point_cov;
	ekfom_data.h_x.resize(effect_num_k, 12);
	ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_num_k, 12);
	ekfom_data.z.resize(effect_num_k);
	int m = 0;
	for (int j = 0; j < time_seq[k]; j++)
	{
		// ekfom_data.converge = false;
		if(point_selected_surf[idx+j+1])
		{
			V3D norm_vec(normvec->points[j].x, normvec->points[j].y, normvec->points[j].z);
			if (extrinsic_est_en)
			{
				V3D p_body = pbody_list[idx+j+1];
				M3D p_crossmat, p_imu_crossmat;
				p_crossmat << SKEW_SYM_MATRX(p_body);
				V3D point_imu = s.offset_R_L_I * p_body + s.offset_T_L_I;
				p_imu_crossmat << SKEW_SYM_MATRX(point_imu);
				V3D C(s.rot.transpose() * norm_vec);
				V3D A(p_imu_crossmat * C);
				V3D B(p_crossmat * s.offset_R_L_I.transpose() * C);
				ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
			}
			else
			{   
				M3D point_crossmat = crossmat_list[idx+j+1];
				V3D C(s.rot.transpose() * norm_vec); // conjugate().normalized()
				V3D A(point_crossmat * C);
				ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
			}
			ekfom_data.z(m) = -norm_vec(0) * feats_down_world->points[idx+j+1].x -norm_vec(1) * feats_down_world->points[idx+j+1].y -norm_vec(2) * feats_down_world->points[idx+j+1].z-normvec->points[j].intensity;
			
			m++;
		}
	}
	effct_feat_num += effect_num_k;
}

void h_model_IMU_output(state_output &s, esekfom::dyn_share_modified<double> &ekfom_data)
{
    std::memset(ekfom_data.satu_check, false, 6);
	ekfom_data.z_IMU.block<3,1>(0, 0) = angvel_avr - s.omg - s.bg;
	ekfom_data.z_IMU.block<3,1>(3, 0) = acc_avr * G_m_s2 / acc_norm - s.acc - s.ba;
    ekfom_data.R_IMU << imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_acc_cov, imu_meas_acc_cov, imu_meas_acc_cov;
	if(check_satu)
	{
		if(fabs(angvel_avr(0)) >= 0.99 * satu_gyro)
		{
			ekfom_data.satu_check[0] = true; 
			ekfom_data.z_IMU(0) = 0.0;
		}
		
		if(fabs(angvel_avr(1)) >= 0.99 * satu_gyro) 
		{
			ekfom_data.satu_check[1] = true;
			ekfom_data.z_IMU(1) = 0.0;
		}
		
		if(fabs(angvel_avr(2)) >= 0.99 * satu_gyro)
		{
			ekfom_data.satu_check[2] = true;
			ekfom_data.z_IMU(2) = 0.0;
		}
		
		if(fabs(acc_avr(0)) >= 0.99 * satu_acc)
		{
			ekfom_data.satu_check[3] = true;
			ekfom_data.z_IMU(3) = 0.0;
		}

		if(fabs(acc_avr(1)) >= 0.99 * satu_acc) 
		{
			ekfom_data.satu_check[4] = true;
			ekfom_data.z_IMU(4) = 0.0;
		}

		if(fabs(acc_avr(2)) >= 0.99 * satu_acc) 
		{
			ekfom_data.satu_check[5] = true;
			ekfom_data.z_IMU(5) = 0.0;
		}
	}
}

void pointBodyToWorld(PointType const * const pi, PointType * const po)
{    
    V3D p_body(pi->x, pi->y, pi->z);
    
    V3D p_global;
	if (extrinsic_est_en)
	{	
		if (!use_imu_as_input)
		{
			p_global = kf_output.x_.rot * (kf_output.x_.offset_R_L_I * p_body + kf_output.x_.offset_T_L_I) + kf_output.x_.pos;
		}
		else
		{
			p_global = kf_input.x_.rot * (kf_input.x_.offset_R_L_I * p_body + kf_input.x_.offset_T_L_I) + kf_input.x_.pos;
		}
	}
	else
	{
		if (!use_imu_as_input)
		{
			p_global = kf_output.x_.rot * (Lidar_R_wrt_IMU * p_body + Lidar_T_wrt_IMU) + kf_output.x_.pos; // .normalized()
		}
		else
		{
			p_global = kf_input.x_.rot * (Lidar_R_wrt_IMU * p_body + Lidar_T_wrt_IMU) + kf_input.x_.pos; // .normalized()
		}
	}

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}