#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <octomap/octomap.h>
#include <octomap_msgs/conversions.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseStamped.h>

class OctomapGenerator {
public:
    // Constructor with additional parameters for grid origin and occupancy grid topic
    OctomapGenerator(std::string point_cloud_topic, std::string robot_pose_topic, std::string occupancy_grid_topic, double origin_x, double origin_y) {
        octree = new octomap::OcTree(resolution);
        grid_pub = nh.advertise<nav_msgs::OccupancyGrid>(occupancy_grid_topic, 1, true);
        pc_sub = nh.subscribe(point_cloud_topic, 1, &OctomapGenerator::pointCloudCallback, this);
        robot_pose_sub = nh.subscribe(robot_pose_topic, 1, &OctomapGenerator::robotPoseCallback, this);
        grid_origin_x = origin_x;
        grid_origin_y = origin_y;
    }

    void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
        ros::Time now = ros::Time::now();
        if ((now - last_update).toSec() >= 0.1) {  // Check if at least 0.1 seconds have passed
            last_update = now;  // Update the time of the last update
            octree->clear();  // Clear the existing octree

            pcl::PCLPointCloud2 pcl_pc2;
            pcl_conversions::toPCL(*cloud_msg, pcl_pc2);
            pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromPCLPointCloud2(pcl_pc2, *temp_cloud);
            octomap::Pointcloud octo_cloud;
            for (const auto& point : temp_cloud->points) {
                octo_cloud.push_back(point.x, point.y, point.z);
            }
            
            double x_pos = robot_pose.pose.position.x;
            double y_pos = robot_pose.pose.position.y;
            double z_pos = robot_pose.pose.position.z;
            octree->insertPointCloud(octo_cloud, octomap::point3d(x_pos, y_pos, z_pos));

            nav_msgs::OccupancyGrid grid_msg;
            grid_msg.header.frame_id = "map";
            grid_msg.header.stamp = ros::Time::now();
            grid_msg.info.resolution = resolution;
            grid_msg.info.width = 100;  // Example width
            grid_msg.info.height = 100;  // Example height
            grid_msg.info.origin.position.x = grid_origin_x;  // Parameterized origin
            grid_msg.info.origin.position.y = grid_origin_y;  // Parameterized origin
            grid_msg.info.origin.position.z = 0;
            grid_msg.data.resize(grid_msg.info.width * grid_msg.info.height, -1);  // Initialize as unknown

            for (octomap::OcTree::iterator it = octree->begin(octree->getTreeDepth()), end = octree->end(); it != end; ++it) {
                if (octree->isNodeOccupied(*it)) {
                    int x = static_cast<int>((it.getX() - grid_msg.info.origin.position.x) / resolution);
                    int y = static_cast<int>((it.getY() - grid_msg.info.origin.position.y) / resolution);
                    if (x >= 0 && x < static_cast<int>(grid_msg.info.width) && y >= 0 && y < static_cast<int>(grid_msg.info.height)) {
                        int index = x + y * grid_msg.info.width;
                        grid_msg.data[index] = 100;  // Mark as occupied
                    }
                }
            }
            grid_pub.publish(grid_msg);
        }
    }

    void robotPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        robot_pose = *msg;
    }

private:
    ros::NodeHandle nh;
    ros::Publisher grid_pub;
    ros::Subscriber pc_sub, robot_pose_sub;
    octomap::OcTree* octree;
    double resolution = 0.05; // OctoMap resolution
    geometry_msgs::PoseStamped robot_pose;
    ros::Time last_update;  // Time of the last update
    double grid_origin_x, grid_origin_y; // Grid origin parameters
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "octomap_generator");

    if (argc < 6) {
        ROS_ERROR("Usage: octomap_generator point_cloud_topic robot_pose_topic occupancy_grid_topic origin_x origin_y");
        return -1;
    }

    std::string point_cloud_topic = argv[1];
    std::string robot_pose_topic = argv[2];
    std::string occupancy_grid_topic = argv[3];
    double origin_x = atof(argv[4]);
    double origin_y = atof(argv[5]);
    
    OctomapGenerator generator(point_cloud_topic, robot_pose_topic, occupancy_grid_topic, origin_x, origin_y);
    ros::spin();
    return 0;
}