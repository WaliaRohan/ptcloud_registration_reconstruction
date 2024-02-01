#include <iostream>
#include <string>
#include <pcl/io/pcd_io.h> // reading pointclouds
#include <pcl/point_types.h> // creating pointcloud shared ptrs

// estimate the SIFT points based on the Normal gradients
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/features/normal_3d.h> 

#include <pcl/filters/filter.h>

// feature matching
#include <pcl/features/fpfh.h>

#include <pcl/visualization/pcl_visualizer.h> //visualization

#include <chrono> // high_resolution_clock

#include <vector> // for debugging
#include <cmath>

#include <pcl/common/transforms.h> // for transformPointCloud

#include <pcl/registration/ia_ransac.h> // for sample consesus intial allignment
#include <pcl/registration/icp.h> // for iterative allignment

using XYZCloudPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;
using NormalCloudPtr = pcl::PointCloud<pcl::PointNormal>::Ptr;
using FPFH33Ptr = pcl::PointCloud<pcl::FPFHSignature33>::Ptr;

bool readPointCloud(std::string filePath, XYZCloudPtr cloudPtr)
{  
    std::cout << "Attempting to load: " << filePath << std::endl;

    if (pcl::io::loadPCDFile<pcl::PointXYZ> (filePath, *cloudPtr) == -1)
    {
        PCL_ERROR("Couldn't read given file");
        return false;   
    }

    std::vector<int> nan_idx;
    pcl::removeNaNFromPointCloud(*cloudPtr, *cloudPtr, nan_idx);

    std::cout << "Loaded "
              << cloudPtr->size()
              << " data points from given file"
              << std::endl;

    return true;
}

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


NormalCloudPtr estimateValidNormalCloud(XYZCloudPtr cloudPtr)
{
    // Estimate the normals of the cloud_xyz -> normals help determine orientation of a surface -> https://pcl.readthedocs.io/projects/tutorials/en/master/normal_estimation.html#normal-estimation
    pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
    ne.setInputCloud(cloudPtr);

    std::cout << "Normal Estimation: Generated normal estimation object" << std::endl;

    // // Create an emptyKdTree and pass it to the estimation object. KdTree enables efficient range searches and nearest neighbor searches -> https://pcl.readthedocs.io/projects/tutorials/en/latest/walkthrough.html#kdtree
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_n(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree_n); 
    // ne.setKSearch(5); // Search for given number of neighbors around each point -> faster
    ne.setRadiusSearch(0.03); // Use all neighbors in a sphere of this radius in cm
    // more info here: https://pcl.readthedocs.io/en/pcl-1.11.1/normal_estimation.html#selecting-the-right-scale

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

    // Note that the compute function would have only computed the normal_x,
    // normal_y, normal_z and curvature values for Copy the xyz info from
    // the original pointcloud (see
    // https://pointclouds.org/documentation/point__types_8hpp_source.html#l00869)

    std::vector<int> vec(3, 0); // for dubugging

    // So, below we copy x, y and z fields from cloudPtr to cloud_normals
    for(std::size_t i = 0; i<cloud_normals->size(); ++i)
    {
        (*cloud_normals)[i].x = (*cloudPtr)[i].x;
        (*cloud_normals)[i].y = (*cloudPtr)[i].y;
        (*cloud_normals)[i].z = (*cloudPtr)[i].z;

        if (std::isnan((*cloud_normals)[i].normal_x))
            vec[0]++;
        
        if (std::isnan((*cloud_normals)[i].normal_x))
            vec[1]++;

        if (std::isnan((*cloud_normals)[i].normal_x))
            vec[2]++;
    }

    std::cout << vec[0] << " " << vec[1] << " " << vec[2] << std::endl;

    // used for mapping points between input and output clouds -> not used here
    // since input and output clouds are the same
    std::vector<int> indices; 
    pcl::removeNaNNormalsFromPointCloud (*cloud_normals, *cloud_normals, indices);

    std::cout << cloud_normals->size() << std::endl;

    return cloud_normals;
}

pcl::PointCloud<pcl::PointWithScale>::Ptr getSIFTKeypoints(NormalCloudPtr cloud_normals)
{  
    /*
    Parameters for sift computation - as defined in in pcl::SIFTKeypoint class

    min_scale standard deviation of the smallest scale in the scale space, or the
              standard deviation of the smallest Gaussian used in the 
              difference-of-Gaussian function (see original paper)
    */
    constexpr float min_scale = 0.01f; // not sure why this value is used
    constexpr int n_octaves = 3; // number of octaves
    constexpr int n_scales_per_octave = 4; // scales per octave
    constexpr float min_contrast = 0.001f; // not sure why this value is used

    std::cout << "Keypoint detection: Generating SIFT keypoints" << std::endl;

    pcl::PointCloud<pcl::PointWithScale>::Ptr result (new pcl::PointCloud<pcl::PointWithScale>);

    // Estimate the sift interest points using normals values from xyz as the Intensity variants
    pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointNormal> ());
    sift.setSearchMethod(tree);
    sift.setScales(min_scale, n_octaves, n_scales_per_octave);
    sift.setMinimumContrast(min_contrast);
    sift.setInputCloud(cloud_normals);
    std::cout << "Going to compute result" << std::endl;
    sift.compute(*result);

    std::cout << "Keypoint detection: # of SIFT points in the result are "
              << result->size() 
              << std::endl;

    return result;
}

FPFH33Ptr generateFPFHfeatures(pcl::PointCloud<pcl::PointWithScale>::Ptr keypointCloud,
                                                     NormalCloudPtr normalCloud)
{
    // Create FPFH estimation class object
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    
    // Convert pointwithscale keypoint cloud to xyz keypoint cloud
    XYZCloudPtr xyzKeypointCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*keypointCloud, *xyzKeypointCloud);

    // Set keypointCloud as input cloud
    fpfh.setInputCloud(xyzKeypointCloud);

    // Generate xyz search surface from normalCloud
    XYZCloudPtr xyzCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*normalCloud, *xyzCloud);
    
    // Set xyzcloud as search surface
    fpfh.setSearchSurface(xyzCloud);
    
    // Convert pointNormal cloud to Normal cloud
    pcl::PointCloud<pcl::Normal>::Ptr normal (new pcl::PointCloud<pcl::Normal>);
    pcl::copyPointCloud(*normalCloud, *normal);

    // Set this cloud as normal cloud
    fpfh.setInputNormals(normal);

    // Since no other search surface is given, we will create an empty kdtree
    // rerpesentation and pass it to the FPFH estimation object.
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    fpfh.setSearchMethod(tree);
    fpfh.setRadiusSearch(0.05); // radius used here has to be larger than that used for estimating surface normals

    // Output dataset
    FPFH33Ptr features_ptr (new pcl::PointCloud<pcl::FPFHSignature33>());
    
    std::cout << "Feature estimation: Estimation object contstructed. Going to compute \n";
    
    fpfh.compute(*features_ptr);

    std::cout << "Feature estimation: Computed " 
              << features_ptr->size() 
              << " FPFH features for source cloud \n";

    return features_ptr;
}

Eigen::Matrix4f guessSampleConsesusInitialAlignment(pcl::PointCloud<pcl::PointWithScale>::Ptr sourceKeypoints,
                                         pcl::PointCloud<pcl::PointWithScale>::Ptr targetKeypoints, 
                                         FPFH33Ptr sourceFeatures,
                                         FPFH33Ptr targetFeatures,
                                         XYZCloudPtr alignedPointCloud)
{
    pcl::SampleConsensusInitialAlignment<pcl::PointWithScale, pcl::PointWithScale, pcl::FPFHSignature33> scia;

    // Sample Consensus Initial Alignment parameters (explanation below)
    const float min_sample_dist = 0.025f; // The minimum distance between any two random samples
    const float max_correspondence_dist = 0.01f; // The maximum distance between a point and its nearest neighbor correspondent in order to be considered in the alignment process
    const int nr_iters = 500; // The number of RANSAC iterations to perform

    // set input, output pointclouds and features
    scia.setInputSource(sourceKeypoints);
    scia.setSourceFeatures(sourceFeatures);

    scia.setInputTarget(targetKeypoints);
    scia.setTargetFeatures(targetFeatures);

    scia.setMinSampleDistance(min_sample_dist);
    scia.setMaxCorrespondenceDistance(max_correspondence_dist);
    scia.setMaximumIterations(nr_iters);

    pcl::PointCloud<pcl::PointWithScale> registration_output;
    scia.align(registration_output); 

    // Convert pointwithscale keypoint cloud to xyz cloud
    pcl::copyPointCloud(registration_output, *alignedPointCloud);

    Eigen::Matrix4f transformation = scia.getFinalTransformation();

    std::cout << "Sample Conensus: Calculatead new pointcloud of size: "
              << registration_output.size() << std::endl;

    std::cout << transformation << std::endl;

    return transformation;
}

// This might come in handy: https://stackoverflow.com/questions/44605198/pointcloud-library-compute-sift-keypoints-input-cloud-error

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
    NormalCloudPtr normalCloudPtrSource = estimateValidNormalCloud(cloudSource);
    NormalCloudPtr normalCloudPtrTarget = estimateValidNormalCloud(cloudTarget);

    pcl::PointCloud<pcl::PointWithScale>::Ptr sourceSIFTKeypoints = 
    getSIFTKeypoints(normalCloudPtrSource);

    pcl::PointCloud<pcl::PointWithScale>::Ptr targetSIFTKeypoints =
    getSIFTKeypoints(normalCloudPtrTarget);

    // 3) Get FPFH feature descriptors
    FPFH33Ptr sourceFPFHFeature =
     generateFPFHfeatures(sourceSIFTKeypoints, normalCloudPtrSource);

    FPFH33Ptr targetFPFHFeature = 
     generateFPFHfeatures(targetSIFTKeypoints, normalCloudPtrTarget);

    XYZCloudPtr alignedPointCloud(new pcl::PointCloud<pcl::PointXYZ>);

    Eigen::Matrix4f initialTransformation = 
    guessSampleConsesusInitialAlignment(sourceSIFTKeypoints, targetSIFTKeypoints,
                                    sourceFPFHFeature, targetFPFHFeature,
                                    alignedPointCloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());

    pcl::transformPointCloud (*cloudSource, *transformed_cloud, initialTransformation);

    pcl::visualization::PCLVisualizer viewer ("Matrix transformation example");

    // Define R,G,B colors for the point cloud
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler (cloudSource, 0, 255, 0); // Green
    // We add the point cloud to the viewer and pass the color handler
    viewer.addPointCloud (cloudSource, source_cloud_color_handler, "original_cloud");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transformed_cloud_color_handler (transformed_cloud, 0, 0, 255); // Blue
    viewer.addPointCloud (transformed_cloud, transformed_cloud_color_handler, "transformed_cloud");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_cloud_color_handler (cloudTarget, 255, 0, 0); // Red
    viewer.addPointCloud(cloudTarget, target_cloud_color_handler, "target_cloud");

    viewer.addCoordinateSystem (1.0, "cloud", 0);
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0); // Setting background to a dark grey
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "transformed_cloud");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target_cloud");

    //viewer.setPosition(800, 400); // Setting visualiser window position
    while (!viewer.wasStopped ()) { // Display the visualiser until 'q' key is pressed
        viewer.spin();
    }


    return 0;
}

// The Iterative Closest Point algorithm
    // std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    
    // int iterations = 10;  // Default number of ICP iterations

    // icp.setMaximumIterations (iterations);
    // icp.setInputSource (cloudSource);
    // icp.setInputTarget (cloudTarget);
    // icp.align (*cloudSource);
    // icp.setMaximumIterations (1);  // We set this variable to 1 for the next time we will call .align () function
    
    // std::chrono::high_resolution_clock::time_point finish = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(finish-start);
    
    // std::cout << "Applied " << iterations << " ICP iteration(s) in " << time_span.count() << " seconds." << std::endl;

    // if (icp.hasConverged ())
    // {
    //     std::cout << "\nICP has converged, score is " << icp.getFitnessScore () << std::endl;
    //     std::cout << "\nICP transformation " << iterations << " : cloud_icp -> cloud_in" << std::endl;
    //     transformation_matrix = icp.getFinalTransformation().cast<double>();
    //     // print4x4Matrix (transformation_matrix);
    // }

    // else
    // {
    //     PCL_ERROR ("\nICP has not converged.\n");
    //     return (-1);
    // }