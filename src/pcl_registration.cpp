#include <iostream>
#include <string>
#include <pcl/io/pcd_io.h> // reading pointclouds
#include <pcl/point_types.h> // creating pointcloud shared ptrs

// estimate the SIFT points based on the Normal gradients
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/features/normal_3d.h> 

#include <pcl/visualization/pcl_visualizer.h> //visualization

#include <chrono> // high_resolution_clock

using XYZCloudPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;
using NormalCloudPtr = pcl::PointCloud<pcl::PointNormal>::Ptr;


void visualizePointCloud(XYZCloudPtr cloudPtr)
{
    // Visualization
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    viewer.setBackgroundColor( 0.0, 0.0, 0.0 );
    viewer.addPointCloud(cloudPtr, "cloud");
    
    while(!viewer.wasStopped())
    {
        viewer.spin();
    }
}

/*
Feature based registration

    use SIFT Keypoints (pcl::SIFT…something)

    use FPFH descriptors (pcl::FPFHEstimation) at the keypoints (see our tutorials for that, like http://www.pointclouds.org/media/rss2011.html)

    get the FPFH descriptors and estimate correspondences using pcl::CorrespondenceEstimation

    reject bad correspondences using one or many of the pcl::CorrespondenceRejectionXXX methods

    finally get a transformation as mentioned above
*/


NormalCloudPtr estimateNormalCloud(XYZCloudPtr cloudPtr)
{
    // Estimate the normals of the cloud_xyz -> normals help determine orientation of a surface -> https://pcl.readthedocs.io/projects/tutorials/en/master/normal_estimation.html#normal-estimation
    pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
    ne.setInputCloud(cloudPtr);

    std::cout << "Normal Estimation: Generated normal estimation object" << std::endl;

    // // Create an emptyKdTree and pass it to the estimation object. KdTree enables efficient range searches and nearest neighbor searches -> https://pcl.readthedocs.io/projects/tutorials/en/latest/walkthrough.html#kdtree
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_n(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree_n); 
    ne.setRadiusSearch(0.1); // Use all neighbors in a sphere of this radius in cm

    std::cout << "Normal Estimation: Computing normals" << std::endl;

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // // Output dataset to store the result
    NormalCloudPtr cloud_normals (new pcl::PointCloud<pcl::PointNormal>);
    ne.compute(*cloud_normals);

    std::chrono::high_resolution_clock::time_point finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(finish-start);

    // cloud_normals->size () should have the same size as the input cloud->size ()*
    std::cout << "Normal Estimation: Successfuly computed normals and generated cloud of size: "
              << cloud_normals->size()
              << " (original cloud size: "
              << cloudPtr->size() << ") in "
              << time_span.count() << " seconds."
              << std::endl;    

    // For more details on normal estimation, see: https://pcl.readthedocs.io/projects/tutorials/en/master/normal_estimation.html#normal-estimation 


    // Note that the compute function would have only computed ther normal_x,
    // normal_y, normal_z and curvature values for Copy the xyz info from
    // the original pointcloud (see
    // https://pointclouds.org/documentation/point__types_8hpp_source.html#l00869)

    // So, below we copy x, y and z fields from cloudPtr to cloud_normals
    for(std::size_t i = 0; i<cloud_normals->size(); ++i)
    {
        (*cloud_normals)[i].x = (*cloudPtr)[i].x;
        (*cloud_normals)[i].y = (*cloudPtr)[i].y;
        (*cloud_normals)[i].z = (*cloudPtr)[i].z;
    }

    return cloud_normals;
}

pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> getSIFTKeypoints(NormalCloudPtr cloud_normals)
{   // Parameters for sift computation - as defined in in pcl::SIFTKeypoint class

    /*
    standard deviation of the smallest scale in the scale space, or the
    standard deviation of the smallest Gaussian used in the 
    difference-of-Gaussian function (see original paper)
    */
    constexpr float min_scale = 0.01f; // not sure why this value is used
    constexpr int n_octaves = 3; // number of octaves
    constexpr int n_scales_per_octave = 4; // scales per octave
    constexpr float min_contrast = 0.001f; // not sure why this value is used

    // Estimate the sift interest points using normals values from xyz as the Intensity variants
    pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;
    pcl::PointCloud<pcl::PointWithScale> result;
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointNormal> ());
    sift.setSearchMethod(tree);
    sift.setScales(min_scale, n_octaves, n_scales_per_octave);
    sift.setMinimumContrast(min_contrast);
    sift.setInputCloud(cloud_normals);
    sift.compute(result);

    std::cout << "Keypoint Estimation: # of SIFT points in the result are "
              << result.size() 
              << std::endl;

    return sift;
}

void generatedFPFHEstimators()
{

}

bool readPointCloud(std::string filePath, XYZCloudPtr cloudPtr)
{  
    std::cout << "Attempting to load: " << filePath << std::endl;

    if (pcl::io::loadPCDFile<pcl::PointXYZ> (filePath, *cloudPtr) == -1)
    {
        PCL_ERROR("Couldn't read given file");
        return false;   
    }

    std::cout << "Loaded "
              << cloudPtr->size()
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

    // 1) Data Acquisition - acquire pair of pointclouds

    // Read first point cloud
    XYZCloudPtr cloudSource(new pcl::PointCloud<pcl::PointXYZ>);
    if (!readPointCloud(file1, cloudSource))
        return -1;

    // Read second point cloud
    XYZCloudPtr cloudTarget(new pcl::PointCloud<pcl::PointXYZ>);
    if (!readPointCloud(file2, cloudTarget))
        return -1;

    std::cout << "Estimating keypoints" << std::endl;

    // 2) Keypoint estimatation
    NormalCloudPtr normalCloudPtrSource = estimateNormalCloud(cloudSource);
    NormalCloudPtr normalCloudPtrTarget = estimateNormalCloud(cloudTarget);

    pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> siftKeypointsSource
     = getSIFTKeypoints(normalCloudPtrSource);

    pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> siftKeypointsTarget
     = getSIFTKeypoints(normalCloudPtrTarget);

    // 3) Get FPHP feature descriptors


    return 0;
}