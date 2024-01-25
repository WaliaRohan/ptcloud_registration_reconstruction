#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include<string>

int main()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

    std::string filepath = "/home/speedracer1702/Projects/Black_I/surface_reconstruction/data/living_room/pcd/livingRoomData_pt_1.pcd";

    if (pcl::io::loadPCDFile<pcl::PointXYZ> (filepath, *cloud) == -1)
    {
        PCL_ERROR("Couldn't read given file");
        return (-1);   
    }

    std::cout << "Loaded"
              << cloud->width * cloud->height
              << " data points from given file with the following fields: "
              << std::endl;

    for (const auto& point: *cloud)
        std::cout << "  " << point.x
                  << " " << point.y
                  << " " << point.z << std::endl;

    return (0);
}