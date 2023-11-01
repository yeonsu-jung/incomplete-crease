function plot_pointcloud(all_points)

X = all_points(:,1);
Y = all_points(:,2);
Z = all_points(:,3);

ptCloud = pointCloud([X,Y,Z]);
pcshow(ptCloud)


end