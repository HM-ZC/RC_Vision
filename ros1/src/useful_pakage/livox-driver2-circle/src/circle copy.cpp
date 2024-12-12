#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <thread>
#include <vector>

ros::Publisher circle_pub;
ros::Publisher velocity_pub;
ros::Publisher acceleration_pub;
ros::Publisher cloud_pub;
ros::Publisher cloud_pub2; 

pcl::PointXYZ prev_center;
ros::Time prev_time;
geometry_msgs::TwistStamped prev_velocity;

double computeDistance(const pcl::PointXYZ& point) {
    return sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
}

void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");

    for (double z_min = 0; z_min <= 1.0; z_min += 0.01) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pass.setFilterLimits(z_min, z_min + 0.1);
        pass.filter(*filtered_cloud);
        *merged_cloud += *filtered_cloud;
    }

    sensor_msgs::PointCloud2 output_cloud_msg2;
    pcl::toROSMsg(*merged_cloud, output_cloud_msg2);
    output_cloud_msg2.header.frame_id = "livox_frame";
    output_cloud_msg2.header.stamp = ros::Time::now();
    cloud_pub2.publish(output_cloud_msg2);

    for (auto& point : merged_cloud->points) {
        point.z = 0.0;
    }

    sensor_msgs::PointCloud2 output_cloud_msg;
    pcl::toROSMsg(*merged_cloud, output_cloud_msg);
    output_cloud_msg.header.frame_id = "livox_frame";
    output_cloud_msg.header.stamp = ros::Time::now();
    cloud_pub.publish(output_cloud_msg);

    pcl::PointXYZ nearest_center;
    double nearest_distance = std::numeric_limits<double>::max();
    bool found_circle = false;

    while (merged_cloud->points.size() > 0) {
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_CIRCLE2D);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.1);
        seg.setInputCloud(merged_cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.size() == 0) {
            break;
        }

        pcl::PointXYZ current_center;
        current_center.x = coefficients->values[0];
        current_center.y = coefficients->values[1];
        current_center.z = 0.0;

        double distance = computeDistance(current_center);
        if (distance < nearest_distance) {
            nearest_distance = distance;
            nearest_center = current_center;
            found_circle = true;
        }

        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(merged_cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*merged_cloud);
    }

    if (!found_circle) {
        ROS_WARN("Could not find a circular inliers in the point cloud.");
        return;
    }

    geometry_msgs::PointStamped circle_center;
    circle_center.header.stamp = ros::Time::now();
    circle_center.point.x = nearest_center.x;
    circle_center.point.y = nearest_center.y;
    circle_center.point.z = nearest_center.z;
    circle_pub.publish(circle_center);

    ros::Time current_time = ros::Time::now();
    double dt = (current_time - prev_time).toSec();

    if (dt > 0) {
        geometry_msgs::TwistStamped velocity;
        velocity.header.stamp = current_time;
        velocity.twist.linear.x = (nearest_center.x - prev_center.x) / dt;
        velocity.twist.linear.y = (nearest_center.y - prev_center.y) / dt;
        velocity_pub.publish(velocity);

        geometry_msgs::TwistStamped acceleration;
        acceleration.header.stamp = current_time;
        acceleration.twist.linear.x = (velocity.twist.linear.x - prev_velocity.twist.linear.x) / dt;
        acceleration.twist.linear.y = (velocity.twist.linear.y - prev_velocity.twist.linear.y) / dt;
        acceleration_pub.publish(acceleration);

        prev_velocity = velocity;
    }

    prev_center = nearest_center;
    prev_time = current_time;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "circle_fitting_node");
    ros::NodeHandle nh;

    ros::Subscriber cloud_sub = nh.subscribe("/livox/lidar", 1, pointCloudCallback);

    circle_pub = nh.advertise<geometry_msgs::PointStamped>("circle_center", 1);
    velocity_pub = nh.advertise<geometry_msgs::TwistStamped>("circle_velocity", 1);
    acceleration_pub = nh.advertise<geometry_msgs::TwistStamped>("circle_acceleration", 1);
    cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("filtered_cloud", 1);
    cloud_pub2 = nh.advertise<sensor_msgs::PointCloud2>("filtered_cloud2", 1);

    prev_time = ros::Time::now();

    ros::spin();
    return 0;
}