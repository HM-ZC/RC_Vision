#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/transforms.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>  // 新增的头文件，用于PassThrough滤波
#include <pcl/segmentation/sac_segmentation.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Vector3.h>
#include <Eigen/Dense>
#include <cmath>
#include <visualization_msgs/Marker.h>
#include <deque>

class SphereDetection
{
public:
    SphereDetection()
    {
        // 初始化订阅和发布
        sub_ = nh_.subscribe("/livox/lidar", 1, &SphereDetection::pointCloudCallback, this);
        marker_pub_ = nh_.advertise<visualization_msgs::Marker>("visualization_marker", 1);
        filtered_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("filtered_pointcloud", 1);

        // 增加轨迹可视化的发布器
        trajectory_pub_ = nh_.advertise<visualization_msgs::Marker>("trajectory_marker", 1);
        predicted_trajectory_pub_ = nh_.advertise<visualization_msgs::Marker>("predicted_trajectory_marker", 1);
backboard_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("backboard_marker", 1);
basket_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("basket_marker", 1);
three_point_line_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("three_point_line_marker", 1);

        // 初始化卡尔曼滤波器
        initialized_ = false;
        dt_ = 0.1; // 时间间隔，需根据实际情况调整

        // 状态向量：[x, y, z, vx, vy, vz]
        state_ = Eigen::VectorXd::Zero(6);
        // 状态转移矩阵
        F_ = Eigen::MatrixXd::Identity(6, 6);
        F_(0, 3) = dt_;
        F_(1, 4) = dt_;
        F_(2, 5) = dt_;
        // 过程噪声协方差矩阵
        Q_ = Eigen::MatrixXd::Identity(6, 6) * 0.01;
        // 观测矩阵
        H_ = Eigen::MatrixXd::Zero(3, 6);
        H_(0, 0) = 1;
        H_(1, 1) = 1;
        H_(2, 2) = 1;
        // 观测噪声协方差矩阵
        R_ = Eigen::MatrixXd::Identity(3, 3) * 0.1;
        // 状态协方差矩阵
        P_ = Eigen::MatrixXd::Identity(6, 6);

// 设置篮筐和篮板的位置和尺寸，以雷达为原点
basket_position_ = Eigen::Vector3d(0, 7.24, 3.05);       // 篮筐位于雷达前方7.24米处，高度3.05米
basket_radius_ = 0.23;                                   // 篮筐半径
backboard_position_ = Eigen::Vector3d(0, 7.39, 3.05);    // 篮板位于篮筐后方0.15米处
backboard_height_ = 2.7;                                 // 篮板高度

        // 初始化轨迹 Marker
        initializeTrajectoryMarkers();

        max_height_reached_ = false;  // 轨迹是否到达最高点的标记
        last_z_ = 0;  // 用于记录上一次的高度
    }

    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
        // 将ROS点云消息转换为PCL点云格式
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_raw(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*cloud_msg, *cloud_raw);

        // 预处理点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        preprocessPointCloud(cloud_raw, cloud_filtered);

        // 发布预处理后的点云
        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*cloud_filtered, output);
        output.header = cloud_msg->header;  // 保持相同的header，确保坐标系一致
        filtered_cloud_pub_.publish(output);

        // 球体检测
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        bool found = detectSphere(cloud_filtered, coefficients, inliers);

        if (!found)
        {
            ROS_INFO("no basketball ");
            return;
        }

        // 获取球体中心坐标和半径
        float x = coefficients->values[0];
        float y = coefficients->values[1];
        float z = coefficients->values[2];
        float radius = coefficients->values[3];

        // 发布可视化的Marker
        publishMarker(x, y, z, radius);

        // 卡尔曼滤波器更新
        updateKalmanFilter(x, y, z);

        // 发布预测轨迹
        Eigen::Vector3d angular_velocity(0, 0, 20);  // 假设球有旋转
        publishPredictedTrajectory(state_.head(3), state_.tail(3), angular_velocity);


    auto [success_prob, no_score_prob, left_prob, right_prob] = monteCarloSimulation(100, state_.head(3), state_.tail(3), angular_velocity);

    ROS_INFO("Predict Results:");
    ROS_INFO(" - Scored Probability: %.2f%%", success_prob * 100);
    ROS_INFO(" - Missed Probability: %.2f%%", no_score_prob * 100);
    if (no_score_prob > 0)
    {
        ROS_INFO("   - Landing Left Probability: %.2f%%", left_prob * 100);
        ROS_INFO("   - Landing Right Probability: %.2f%%", right_prob * 100);
    }
        ROS_INFO("center: (%f, %f, %f), radius: %f, v: (%f, %f, %f)",
                 state_(0), state_(1), state_(2), radius,
                 state_(3), state_(4), state_(5));
                     // 发布可视化的篮球场元素
    publishBackboardMarker();
    publishBasketMarker();
    publishThreePointLineMarker();
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Publisher marker_pub_;  // 发布可视化Marker
    ros::Publisher filtered_cloud_pub_;
    ros::Publisher trajectory_pub_;  // 发布历史轨迹
    ros::Publisher predicted_trajectory_pub_;  // 发布预测轨迹
ros::Publisher backboard_marker_pub_;     // 发布篮板可视化Marker
ros::Publisher basket_marker_pub_;         // 发布篮筐可视化Marker
ros::Publisher three_point_line_marker_pub_; // 发布三分线可视化Marker
    // 卡尔曼滤波器相关变量
    bool initialized_;
    double dt_;
    Eigen::VectorXd state_;
    Eigen::MatrixXd F_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd H_;
    Eigen::MatrixXd R_;
    Eigen::MatrixXd P_;

    // 篮筐和篮板位置与尺寸
    Eigen::Vector3d basket_position_;
    double basket_radius_;
    Eigen::Vector3d backboard_position_;
    double backboard_height_;

    visualization_msgs::Marker trajectory_marker_;
    visualization_msgs::Marker predicted_trajectory_marker_;

    bool max_height_reached_;  // 是否达到最高点
    double last_z_;  // 上一次球的高度
	// 用于记录投篮结果的结构体
struct ShotResult
{
    bool scored;
    std::string landing_side;  // "left" 或 "right"
};
    void initializeTrajectoryMarkers()
    {
        // 设置历史轨迹 Marker
        trajectory_marker_.header.frame_id = "livox_frame";
        trajectory_marker_.ns = "trajectory";
        trajectory_marker_.id = 1;
        trajectory_marker_.type = visualization_msgs::Marker::LINE_STRIP;
        trajectory_marker_.action = visualization_msgs::Marker::ADD;
        trajectory_marker_.scale.x = 0.01;  // 线条宽度
        trajectory_marker_.color.r = 0.0f;
        trajectory_marker_.color.g = 0.0f;
        trajectory_marker_.color.b = 1.0f;  // 蓝色轨迹
        trajectory_marker_.color.a = 1.0;

        // 设置预测轨迹 Marker
        predicted_trajectory_marker_.header.frame_id = "livox_frame";
        predicted_trajectory_marker_.ns = "predicted_trajectory";
        predicted_trajectory_marker_.id = 2;
        predicted_trajectory_marker_.type = visualization_msgs::Marker::LINE_STRIP;
        predicted_trajectory_marker_.action = visualization_msgs::Marker::ADD;
        predicted_trajectory_marker_.scale.x = 0.01;
        predicted_trajectory_marker_.color.r = 1.0f;  // 红色轨迹
        predicted_trajectory_marker_.color.g = 0.0f;
        predicted_trajectory_marker_.color.b = 0.0f;
        predicted_trajectory_marker_.color.a = 1.0;
    }

void publishPredictedTrajectory(Eigen::Vector3d position, Eigen::Vector3d velocity, const Eigen::Vector3d& angular_velocity)
{
    predicted_trajectory_marker_.points.clear();
    double time_step = 0.01;

    // 模拟直到球体落地
    while (position.z() > 0)
    {
        geometry_msgs::Point point;
        point.x = position.x();
        point.y = position.y();
        point.z = position.z();
        predicted_trajectory_marker_.points.push_back(point);

        position += rungeKuttaStep(velocity, angular_velocity, time_step);

        // 检查篮板或篮筐的碰撞并反弹
        if (checkCollisionWithBackboard(position, velocity) || checkCollisionWithRim(position, velocity))
        {
            Eigen::Vector3d normal = (checkCollisionWithBackboard(position, velocity))
                                     ? Eigen::Vector3d(0, -1, 0)
                                     : (position - basket_position_).normalized();
            bounceBall(velocity, normal, angular_velocity);
        }
    }

    predicted_trajectory_marker_.header.frame_id = "livox_frame";  // 坐标系设为雷达原点
    predicted_trajectory_marker_.header.stamp = ros::Time::now();
    predicted_trajectory_pub_.publish(predicted_trajectory_marker_);
}
    // 处理空气阻力
    void applyAirResistance(Eigen::Vector3d& velocity, double dt)
    {
        double air_density = 1.2;  // 空气密度（kg/m^3）
        double drag_coefficient = 0.47;  // 球的阻力系数
        double radius = 0.8;  // 球的半径（米）
        double area = M_PI * radius * radius;  // 球的正面投影面积

        double speed = velocity.norm();
        Eigen::Vector3d drag_force = -0.5 * air_density * drag_coefficient * area * speed * velocity;
        velocity += drag_force * dt;
    }

    // 处理Magnus效应
    void applyMagnusEffect(Eigen::Vector3d& velocity, const Eigen::Vector3d& angular_velocity, double dt)
    {
        double magnus_coefficient = 0.0004;  // 假设的Magnus效应系数
        Eigen::Vector3d magnus_force = magnus_coefficient * angular_velocity.cross(velocity);
        velocity += magnus_force * dt;
    }

    // Runge-Kutta法来计算运动
Eigen::Vector3d rungeKuttaStep(Eigen::Vector3d& velocity, const Eigen::Vector3d& angular_velocity, double dt)
{
    Eigen::Vector3d k1 = velocity * dt;
    Eigen::Vector3d k2 = (velocity + 0.5 * k1) * dt;
    Eigen::Vector3d k3 = (velocity + 0.5 * k2) * dt;
    Eigen::Vector3d k4 = (velocity + k3) * dt;

    applyAirResistance(velocity, dt);
    applyMagnusEffect(velocity, angular_velocity, dt);  // 考虑Magnus效应
    velocity.z() -= 9.81 * dt;  // 重力作用

    return (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;
}

    // 蒙特卡罗模拟来统计进篮概率
std::tuple<double, double, double, double> monteCarloSimulation(int num_simulations, const Eigen::Vector3d& initial_position, const Eigen::Vector3d& initial_velocity, const Eigen::Vector3d& angular_velocity)
{
    int success_count = 0;
    int left_count = 0;
    int right_count = 0;

    for (int i = 0; i < num_simulations; ++i)
    {
        Eigen::Vector3d perturbed_position = initial_position + Eigen::Vector3d::Random() * 0.1;  // 添加微小扰动
        Eigen::Vector3d perturbed_velocity = initial_velocity + Eigen::Vector3d::Random() * 0.1;

        ShotResult result = predictBallTrajectory(perturbed_position, perturbed_velocity, angular_velocity);

        if (result.scored)
        {
            success_count++;
        }
        else if (result.landing_side == "left")
        {
            left_count++;
        }
        else if (result.landing_side == "right")
        {
            right_count++;
        }
    }

    double success_probability = static_cast<double>(success_count) / num_simulations;
    double no_score_probability = 1.0 - success_probability;
    double left_probability = (no_score_probability > 0) ? static_cast<double>(left_count) / (num_simulations - success_count) : 0.0;
    double right_probability = (no_score_probability > 0) ? static_cast<double>(right_count) / (num_simulations - success_count) : 0.0;

    return {success_probability, no_score_probability, left_probability, right_probability};
}

    // 预测球的运动轨迹
ShotResult predictBallTrajectory(Eigen::Vector3d position, Eigen::Vector3d velocity, Eigen::Vector3d angular_velocity)
{
    double time_step = 0.01;
    while (position.z() > 0)  // 模拟到球体触地
    {
        position += rungeKuttaStep(velocity, angular_velocity, time_step);

        if (checkIfBallWillScore(position))  // 检查是否进篮
        {
            return {true, ""};  // 进球时，不需要判断左右
        }

        if (checkCollisionWithBackboard(position, velocity))  // 检查篮板碰撞
        {
            Eigen::Vector3d normal(0, -1, 0);  // 假设篮板法线
            bounceBall(velocity, normal, angular_velocity);
        }

        if (checkCollisionWithRim(position, velocity))  // 检查篮筐碰撞
        {
            Eigen::Vector3d normal = (position - basket_position_).normalized();
            bounceBall(velocity, normal, angular_velocity);
        }
    }

    // 当未进球时，基于最终的水平位置判断落在场地的左边还是右边
    std::string landing_side = (position.x() < 0) ? "left" : "right";
    return {false, landing_side};
}

// 检查球是否与篮筐碰撞
bool checkCollisionWithRim(const Eigen::Vector3d& position, const Eigen::Vector3d& velocity)
{
    // 计算球与篮筐中心的水平距离
    double distance_to_basket = std::sqrt(std::pow(position.x() - basket_position_.x(), 2) +
                                          std::pow(position.y() - basket_position_.y(), 2));

    // 如果距离小于等于篮筐半径且球在篮筐水平面附近，则认为发生碰撞
    return (distance_to_basket <= basket_radius_) && (std::abs(position.z() - basket_position_.z()) < 0.1);
}
    // 检查球是否会进入篮筐
    bool checkIfBallWillScore(const Eigen::Vector3d& position)
    {
        double distance_to_basket = std::sqrt(std::pow(position.x() - basket_position_.x(), 2) +
                                              std::pow(position.y() - basket_position_.y(), 2));
        return distance_to_basket <= basket_radius_ && position.z() >= basket_position_.z();
    }

    // 检查球是否与篮板碰撞
    bool checkCollisionWithBackboard(const Eigen::Vector3d& position, const Eigen::Vector3d& velocity)
    {
        return position.y() >= backboard_position_.y() && position.z() <= backboard_height_;
    }

    // 模拟球与篮板的反弹
void bounceBall(Eigen::Vector3d& velocity, const Eigen::Vector3d& normal, const Eigen::Vector3d& angular_velocity)
{
    // 计算速度在碰撞法线方向的分量
    double normal_velocity = velocity.dot(normal);
    
    // 反转法线方向的速度，模拟反弹
    velocity -= 2 * normal * normal_velocity;
    
    // 考虑自旋效应，增加反弹后的横向速度分量
    Eigen::Vector3d magnus_effect = angular_velocity.cross(normal) * 0.2; // 假设的Magnus效应系数
    velocity += magnus_effect;

    // 模拟能量损失，减少速度幅值
    velocity *= 0.8;
}

    // 点云预处理，包含x, y, z轴筛选
    void preprocessPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out)
    {
        pcl::PassThrough<pcl::PointXYZ> pass;

        // 过滤x轴范围 [-2, 2]
        pass.setInputCloud(cloud_in);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(0.0, 7.0);
        pass.filter(*cloud_out);

        // 过滤y轴范围 [-2, 2]
        pass.setInputCloud(cloud_out);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(-7.0, 7.0);
        pass.filter(*cloud_out);

        // 过滤z轴范围 [0, 5]
        pass.setInputCloud(cloud_out);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(0.2, 5.0);
        pass.filter(*cloud_out);
    }

    bool detectSphere(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                      pcl::ModelCoefficients::Ptr& coefficients,
                      pcl::PointIndices::Ptr& inliers)
    {
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_SPHERE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.01);
        seg.setRadiusLimits(0.05, 0.10); // 根据实际球体大小调整
        seg.setMaxIterations(46000);
        //seg.setOptimizeCoefficients(true);
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.size() == 0)
        {
            return false;
        }
        return true;
    }

void updateKalmanFilter(float x, float y, float z)
{
    if (!initialized_)
    {
        // 从任意位置初始化状态
        state_(0) = x;
        state_(1) = y;
        state_(2) = z;
        initialized_ = true;
        return;
    }

    state_ = F_ * state_;
    P_ = F_ * P_ * F_.transpose() + Q_;

    Eigen::VectorXd z_measure(3);
    z_measure << x, y, z;
    Eigen::VectorXd y_residual = z_measure - H_ * state_;
    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
    Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();
    state_ = state_ + K * y_residual;
    P_ = (Eigen::MatrixXd::Identity(6, 6) - K * H_) * P_;
}

    // 发布球体的Marker可视化
    void publishMarker(float x, float y, float z, float radius)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "livox_frame";  // 确保frame_id与点云的坐标系一致
        marker.header.stamp = ros::Time::now();
        marker.ns = "sphere_detection";
        marker.id = 0;  // 每个marker的唯一id

        marker.type = visualization_msgs::Marker::SPHERE;  // 设置为球体
        marker.action = visualization_msgs::Marker::ADD;

        // 设置球体的位置
        marker.pose.position.x = x;
        marker.pose.position.y = y;
        marker.pose.position.z = z;

        // 设置球体的尺寸
        marker.scale.x = radius * 2;  // 直径
        marker.scale.y = radius * 2;
        marker.scale.z = radius * 2;

        // 设置颜色
        marker.color.r = 0.0f;
        marker.color.g = 1.0f;  // 绿色球体
        marker.color.b = 0.0f;
        marker.color.a = 1.0;   // 透明度

        marker.lifetime = ros::Duration();  // 永久存在

        marker_pub_.publish(marker);  // 发布Marker
    }
    void publishBackboardMarker()
{
    visualization_msgs::Marker backboard_marker;
    backboard_marker.header.frame_id = "livox_frame";  // 基于雷达原点
    backboard_marker.header.stamp = ros::Time::now();
    backboard_marker.ns = "court";
    backboard_marker.id = 1;
    backboard_marker.type = visualization_msgs::Marker::CUBE;
    backboard_marker.action = visualization_msgs::Marker::ADD;

    backboard_marker.pose.position.x = backboard_position_.x();
    backboard_marker.pose.position.y = backboard_position_.y();
    backboard_marker.pose.position.z = backboard_position_.z();

    backboard_marker.scale.x = 0.05;  // 厚度
    backboard_marker.scale.y = 1.83;  // 宽度
    backboard_marker.scale.z = 1.22;  // 高度

    backboard_marker.color.r = 0.6f;
    backboard_marker.color.g = 0.6f;
    backboard_marker.color.b = 0.6f;
    backboard_marker.color.a = 1.0;

    backboard_marker_pub_.publish(backboard_marker);
}
void publishBasketMarker()
{
    visualization_msgs::Marker basket_marker;
    basket_marker.header.frame_id = "livox_frame";  // 基于雷达原点
    basket_marker.header.stamp = ros::Time::now();
    basket_marker.ns = "court";
    basket_marker.id = 2;
    basket_marker.type = visualization_msgs::Marker::CYLINDER;
    basket_marker.action = visualization_msgs::Marker::ADD;

    basket_marker.pose.position.x = basket_position_.x();
    basket_marker.pose.position.y = basket_position_.y();
    basket_marker.pose.position.z = basket_position_.z();

    basket_marker.scale.x = basket_radius_ * 2;  // 直径
    basket_marker.scale.y = basket_radius_ * 2;
    basket_marker.scale.z = 0.05;  // 厚度

    basket_marker.color.r = 1.0f;
    basket_marker.color.g = 0.5f;
    basket_marker.color.b = 0.0f;
    basket_marker.color.a = 1.0;

    basket_marker_pub_.publish(basket_marker);
}
void publishThreePointLineMarker()
{
    visualization_msgs::Marker three_point_line_marker;
    three_point_line_marker.header.frame_id = "livox_frame";  // 基于雷达原点
    three_point_line_marker.header.stamp = ros::Time::now();
    three_point_line_marker.ns = "court";
    three_point_line_marker.id = 3;
    three_point_line_marker.type = visualization_msgs::Marker::LINE_STRIP;
    three_point_line_marker.action = visualization_msgs::Marker::ADD;

    three_point_line_marker.scale.x = 0.02;  // 线条宽度
    three_point_line_marker.color.r = 1.0f;
    three_point_line_marker.color.g = 1.0f;
    three_point_line_marker.color.b = 1.0f;
    three_point_line_marker.color.a = 1.0;

    // 三分线圆弧的半径
    double three_point_radius = 7.24;

    // 绘制三分线的半圆弧
    for (double angle = -M_PI / 2; angle <= M_PI / 2; angle += 0.1)
    {
        geometry_msgs::Point point;
        point.x = three_point_radius * cos(angle);
        point.y = three_point_radius * sin(angle) + basket_position_.y();  // 圆弧中心位于篮筐 y 方向
        point.z = 0;
        three_point_line_marker.points.push_back(point);
    }

    three_point_line_marker_pub_.publish(three_point_line_marker);
}

};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "basketball");
    SphereDetection sd;
    ros::spin();
    return 0;
}
