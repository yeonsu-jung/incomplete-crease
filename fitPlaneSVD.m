function [normal, pointOnPlane] = fitPlaneSVD(pointCloud)
    % Compute the centroid of the point cloud
    centroid = mean(pointCloud, 1);
    
    % Translate the point cloud so that the centroid is at the origin
    translatedPointCloud = pointCloud - centroid;
    
    % Perform SVD on the translated point cloud
    [U,S,V] = svd(translatedPointCloud, 'econ');
    
    % The normal to the plane is the last column of V
    normal = V(:,end);
    
    % Any point on the plane can be used as a reference, we use the centroid
    pointOnPlane = centroid;
    
    % Display the result
    fprintf('Normal to the plane: [%f, %f, %f]\n', normal);
    fprintf('A point on the plane: [%f, %f, %f]\n', pointOnPlane);
end
