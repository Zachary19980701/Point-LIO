/**
 * @file so3_math.h
 * @brief SO(3)流形数学运算库
 *
 * 本文件实现了SO(3)旋转群上的数学运算:
 * 1. 指数映射: 李代数 so(3) -> 李群 SO(3) (Rodrigues公式)
 * 2. 对数映射: 李群 SO(3) -> 李代数 so(3)
 * 3. 反对称矩阵: 向量 -> 反对称矩阵
 * 4. 欧拉角转换
 *
 * 这些运算是卡尔曼滤波状态估计的基础
 */

#ifndef SO3_MATH_H
#define SO3_MATH_H

#include <math.h>
#include <Eigen/Core>

// 反对称矩阵宏: v^ = [0, -v[2], v[1]; v[2], 0, -v[0]; -v[1], v[0], 0]
#define SKEW_SYM_MATRX(v) 0.0,-v[2],v[1],v[2],0.0,-v[0],-v[1],v[0],0.0

/**
 * @brief 构建反对称矩阵
 *
 * 对于向量 v = [v1, v2, v3]^T,其反对称矩阵为:
 *     [  0, -v3,  v2 ]
 * v^= [ v3,   0, -v1 ]
 *     [-v2,  v1,   0 ]
 *
 * 反对称矩阵满足: v^T = -v^, 且对于任意向量u: v^*u = v × u (叉乘)
 *
 * @param v 输入向量
 * @return 反对称矩阵
 */
template<typename T>
Eigen::Matrix<T, 3, 3> skew_sym_mat(const Eigen::Matrix<T, 3, 1> &v)
{
    Eigen::Matrix<T, 3, 3> skew_sym_mat;
    skew_sym_mat<<0.0,-v[2],v[1],v[2],0.0,-v[0],-v[1],v[0],0.0;
    return skew_sym_mat;
}

/**
 * @brief SO(3)指数映射 - Rodrigues公式
 *
 * 将李代数 so(3) 中的元素映射到李群 SO(3):
 *   R = exp(θ * [n]×) = I + sin(θ) * [n]× + (1 - cos(θ)) * [n]×²
 *
 * 其中:
 *   θ = ||ang|| 旋转角度
 *   n = ang / θ  旋转轴 (单位向量)
 *   [n]× = n的反对称矩阵
 *
 * @param ang 旋转向量 (角度*轴 = θ*n)
 * @return 旋转矩阵 R
 */
template<typename T>
Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1> &ang)
{
    T ang_norm = ang.norm();  // 旋转角度 θ
    Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();

    if (ang_norm > 0.0000001)
    {
        Eigen::Matrix<T, 3, 1> r_axis = ang / ang_norm;  // 旋转轴 n
        Eigen::Matrix<T, 3, 3> K;
        K << SKEW_SYM_MATRX(r_axis);  // [n]×

        /// Rodrigues公式: R = I + sin(θ)*K + (1-cos(θ))*K²
        return Eye3 + std::sin(ang_norm) * K + (1.0 - std::cos(ang_norm)) * K * K;
    }
    else
    {
        // 小角度近似: R ≈ I
        return Eye3;
    }
}

template<typename T, typename Ts>
Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1> &ang_vel, const Ts &dt)
{
    T ang_vel_norm = ang_vel.norm();
    Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();

    if (ang_vel_norm > 0.0000001)
    {
        Eigen::Matrix<T, 3, 1> r_axis = ang_vel / ang_vel_norm;
        Eigen::Matrix<T, 3, 3> K;

        K << SKEW_SYM_MATRX(r_axis);

        T r_ang = ang_vel_norm * dt;

        /// Roderigous Tranformation
        return Eye3 + std::sin(r_ang) * K + (1.0 - std::cos(r_ang)) * K * K;
    }
    else
    {
        return Eye3;
    }
}

template<typename T>
Eigen::Matrix<T, 3, 3> Exp(const T &v1, const T &v2, const T &v3)
{
    T &&norm = sqrt(v1 * v1 + v2 * v2 + v3 * v3);
    Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();
    if (norm > 0.00001)
    {
        T r_ang[3] = {v1 / norm, v2 / norm, v3 / norm};
        Eigen::Matrix<T, 3, 3> K;
        K << SKEW_SYM_MATRX(r_ang);

        /// Roderigous Tranformation
        return Eye3 + std::sin(norm) * K + (1.0 - std::cos(norm)) * K * K;
    }
    else
    {
        return Eye3;
    }
}

/**
 * @brief SO(3)对数映射
 *
 * 将李群 SO(3) 中的旋转矩阵映射到李代数 so(3):
 *   θ = arccos((tr(R) - 1) / 2)
 *   n = (R - R^T)^vee / (2 * sin(θ))
 *   ang = θ * n
 *
 * @param R 旋转矩阵
 * @return 旋转向量 (李代数元素)
 */
template<typename T>
Eigen::Matrix<T,3,1> Log(const Eigen::Matrix<T, 3, 3> R)
{
    // 计算旋转角度 θ = arccos((tr(R) - 1)/2)
    T theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1));

    // 计算旋转轴相关向量 K = (R - R^T)^vee
    Eigen::Matrix<T,3,1> K(R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1));

    // 小角度使用线性近似,否则使用精确公式
    return (std::abs(theta) < 0.001) ? (0.5 * K) : (0.5 * theta / std::sin(theta) * K);
}

/**
 * @brief 旋转矩阵转欧拉角 (ZYX顺序,即yaw-pitch-roll)
 *
 * 欧拉角定义 (ZYX顺序):
 *   R = Rz(yaw) * Ry(pitch) * Rx(roll)
 *
 * 从旋转矩阵提取欧拉角:
 *   yaw   = atan2(R(1,0), R(0,0))
 *   pitch = atan2(-R(2,0), sqrt(R(0,0)² + R(1,0)²))
 *   roll  = atan2(R(2,1), R(2,2))
 *
 * @param rot 旋转矩阵
 * @return 欧拉角 [roll, pitch, yaw]
 */
template<typename T>
Eigen::Matrix<T, 3, 1> RotMtoEuler(const Eigen::Matrix<T, 3, 3> &rot)
{
    T sy = sqrt(rot(0,0)*rot(0,0) + rot(1,0)*rot(1,0));
    bool singular = sy < 1e-6;  // 判断是否为奇异点 (pitch接近±90°)
    T x, y, z;

    if(!singular)
    {
        // 正常情况
        x = atan2(rot(2, 1), rot(2, 2));   // roll
        y = atan2(-rot(2, 0), sy);         // pitch
        z = atan2(rot(1, 0), rot(0, 0));   // yaw
    }
    else
    {
        // 奇异点处理 (万向节锁死)
        x = atan2(-rot(1, 2), rot(1, 1));
        y = atan2(-rot(2, 0), sy);
        z = 0;
    }

    Eigen::Matrix<T, 3, 1> ang(x, y, z);
    return ang;
}

/**
 * @brief SO(3)右逆雅可比矩阵
 *
 * 用于在流形上进行优化时的梯度计算:
 *   J_r^{-1}(φ) = I + 0.5*[φ]× + (1 - ||φ||cos(||φ||/2)/(2sin(||φ||/2))) * [φ]×² / ||φ||²
 *
 * @param vec 旋转向量
 * @return 右逆雅可比矩阵
 */
template<typename T>
Eigen::Matrix3d Jacob_right_inv(Eigen::Vector3d &vec){
    Eigen::Matrix3d hat_v, res;
    hat_v << SKEW_SYM_MATRX(vec);

    if(vec.norm() > 1e-6)
    {
        res = Eigen::Matrix<double, 3, 3>::Identity() + 0.5 * hat_v +
              (1 - vec.norm() * std::cos(vec.norm() / 2) / 2 / std::sin(vec.norm() / 2)) *
              hat_v * hat_v / vec.squaredNorm();
    }
    else
    {
        // 小角度近似
        res = Eigen::Matrix<double, 3, 3>::Identity();
    }
    return res;
}

#endif
