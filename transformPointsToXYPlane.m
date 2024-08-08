function transformedPointCloud = transformPointsToXYPlane(pointCloud)
    % Step 1: Fit plane and find normal using SVD
    centroid = mean(pointCloud, 1);
    translatedPointCloud = pointCloud - centroid;
    [~, ~, V] = svd(translatedPointCloud, 'econ');
    normal = V(:,end);

    % Step 2: Find the rotation needed to align the normal with the Z-axis
    % The target vector is the Z-axis
    target = [0; 0; 1];
    
    % Axis of rotation is the cross product of the normal and the target vector
    axisRotation = cross(normal, target);
    axisRotation = axisRotation / norm(axisRotation); % Normalize the axis
    
    % Angle of rotation is the arccos of the dot product of the normal and the target vector
    angle = acos(dot(normal, target));
    
    % Create the rotation matrix using the axis-angle representation
    K = [0, -axisRotation(3), axisRotation(2); axisRotation(3), 0, -axisRotation(1); -axisRotation(2), axisRotation(1), 0];
    rotationMatrix = eye(3) + sin(angle) * K + (1 - cos(angle)) * K^2;

    % Step 3: Apply the rotation to all points
    transformedPointCloud = (rotationMatrix * translatedPointCloud')';
    
    % Optional: Translate points back using the centroid, if desired
    % This step is optional and depends on whether you want the transformed
    % point cloud to be centered around the original centroid or the origin
    transformedPointCloud = transformedPointCloud + centroid;
    
    % Plot original and transformed point clouds for comparison (optional)
    figure;
    subplot(1,2,1);
    scatter3(pointCloud(:,1), pointCloud(:,2), pointCloud(:,3), 'filled');
    title('Original Point Cloud');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    axis equal;
    
    subplot(1,2,2);
    scatter3(transformedPointCloud(:,1), transformedPointCloud(:,2), transformedPointCloud(:,3), 'filled');
    title('Transformed Point Cloud');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    axis equal;
end
