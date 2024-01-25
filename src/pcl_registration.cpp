#include <iostream>
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

bool readPointCloud(std::string filePath, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPtr)
{  
    std::cout << "Attempting to load: " << filePath << std::endl;

    if (pcl::io::loadPCDFile<pcl::PointXYZ> (filePath, *cloudPtr) == -1)
    {
        PCL_ERROR("Couldn't read given file");
        return false;   
    }

    std::cout << "Loaded "
              << cloudPtr->width * cloudPtr->height
              << " data points from given file"
              << std::endl;

    return true;
}


int main ()
{

    std::string folderPath = 
    "/home/speedracer1702/Projects/Black_I/surface_reconstruction/data/living_room/pcd/";
    
    std::string file1 = folderPath + "livingRoomData_pt_1.pcd";
    std::string file2 = folderPath + "livingRoomData_pt_2.pcd";

    // Read first point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSource(new pcl::PointCloud<pcl::PointXYZ>);
    if (!readPointCloud(file1, cloudSource))
        return -1;

    // Read second point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudTarget(new pcl::PointCloud<pcl::PointXYZ>);
    if (!readPointCloud(file2, cloudTarget))
        return -1;

    return 0;
}