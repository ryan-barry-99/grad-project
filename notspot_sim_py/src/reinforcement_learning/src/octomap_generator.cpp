#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <octomap/octomap.h>
#include <octomap_msgs/conversions.h>
#include <octomap_msgs/Octomap.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseStamped.h>

class OctomapGenerator {
public:
    OctomapGenerator() {
        octree = new octomap::OcTree(resolution);

        // Initialize the ROS publisher for the OctoMap
        octomap_pub = nh.advertise<octomap_msgs::Octomap>("octomap_binary", 1, true);
        grid_pub = nh.advertise<nav_msgs::OccupancyGrid>("occupancy_grid", 1, true);

        // Subscribe to PointCloud2 data from the Velodyne scanner
        pc_sub = nh.subscribe("/velodyne_points", 1, &OctomapGenerator::pointCloudCallback, this);

        robot_pose_sub = nh.subscribe("/gazebo/model_poses/robot/notspot", 1, &OctomapGenerator::robotPoseCallback, this);
    }

    void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
        ros::Time now = ros::Time::now();
        if ((now - last_update).toSec() >= 0.1) {  // Check if at least 0.1 seconds have passed
            last_update = now;  // Update the time of the last update
            // Clear the existing octree
            octree->clear();

            // Convert ROS PointCloud2 to PCLPointCloud2
            pcl::PCLPointCloud2 pcl_pc2;
            pcl_conversions::toPCL(*cloud_msg, pcl_pc2);

            // Convert PCLPointCloud2 to PCL PointCloud
            pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromPCLPointCloud2(pcl_pc2, *temp_cloud);

            // Now convert PCL PointCloud to octomap::Pointcloud
            octomap::Pointcloud octo_cloud;
            for (const auto& point : temp_cloud->points) {
                octo_cloud.push_back(point.x, point.y, point.z);
            }

            double x_pos = robot_pose.pose.position.x;
            double y_pos = robot_pose.pose.position.y;
            double z_pos = robot_pose.pose.position.z;

            // Insert the pointcloud into the octree
            octree->insertPointCloud(octo_cloud, octomap::point3d(x_pos, y_pos, z_pos));

            // Publish the octomap
            octomap_msgs::Octomap map_msg;
            map_msg.header.frame_id = "map";  // Set the frame ID to your map frame
            map_msg.header.stamp = ros::Time::now();  // Timestamp the message
            // if (octomap_msgs::binaryMapToMsg(*octree, map_msg)) {
            //     octomap_pub.publish(map_msg);
            // } else {
            //     ROS_ERROR("Error serializing OctoMap");
            // }

            // Now generate and publish the occupancy grid
            nav_msgs::OccupancyGrid grid_msg;
            grid_msg.header.frame_id = "map";  // Adjust as needed
            grid_msg.header.stamp = ros::Time::now();

            // Set the occupancy grid parameters (resolution, width, height, etc.)
            grid_msg.info.resolution = resolution;
            // For simplicity, let's assume a fixed grid size (adjust as needed)
            grid_msg.info.width = 100;  // 10m wide
            grid_msg.info.height = 100;  // 10m tall
            grid_msg.info.origin.position.x = -5;  // Center the grid
            grid_msg.info.origin.position.y = -5;

            // Allocate space for the occupancy data
            grid_msg.data.resize(grid_msg.info.width * grid_msg.info.height, -1);  // Unknown space

            // Iterate through the octree to fill in the occupancy grid
            for (octomap::OcTree::iterator it = octree->begin(octree->getTreeDepth()), end = octree->end(); it != end; ++it) {
                if (octree->isNodeOccupied(*it)) {
                    // Compute the 2D cell coordinates from the 3D node coordinates
                    int x = static_cast<int>((it.getX() - grid_msg.info.origin.position.x) / resolution);
                    int y = static_cast<int>((it.getY() - grid_msg.info.origin.position.y) / resolution);

                    if (x >= 0 && x < static_cast<int>(grid_msg.info.width) && y >= 0 && y < static_cast<int>(grid_msg.info.height)) {
                        int index = x + y * grid_msg.info.width;
                        grid_msg.data[index] = 100;  // Mark as occupied
                    }
                }
            }

            // Publish the occupancy grid
            grid_pub.publish(grid_msg);
        }
    }


    void robotPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        robot_pose = *msg;
    }


private:
    ros::NodeHandle nh;
    ros::Publisher octomap_pub, grid_pub;
    ros::Subscriber pc_sub, robot_pose_sub;
    octomap::OcTree* octree;
    double resolution = 0.1; // OctoMap resolution
    geometry_msgs::PoseStamped robot_pose;
    ros::Time last_update;  // Time of the last update
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "octomap_generator");
    OctomapGenerator generator;
    ros::spin();
    return 0;
}