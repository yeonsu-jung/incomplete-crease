load('zstack_clean.mat')
%%
ind = find(zstack_clean);
[X,Y,Z] = ind2sub(size(zstack_clean),ind);
%%
close all;
plot3(X,Y,Z,'.');axis equal;

%%
ptCloud = pointCloud([X,Y,Z]);
%%
close all;
pcshow(ptCloud)

%%
% clc
% [mesh,depth,perVertexDensity] = pc2surfacemesh(ptCloud,"poisson",2);
%%
[M,I] = max(Z);


%%
close all;
orthosliceViewer(zstack_clean)
%%
center = [968,942,347];
point_cloud = [X,Y,Z];

point_cloud = point_cloud - center;

%%
addpath(genpath('../entanglement/functions'))
crease_line = linspacev([1001,276,194],[968,942,347],100);
crease_line = crease_line(:,[2,1,3]);


close all;
% plot3v(crease_line,'linewidth',5);hold on;
plot3v(point_cloud(1:1000:end,:),'.');
xlabel('x');
ylabel('y')
axis equal
%%
sparse_points = point_cloud(1:1000:end,:);
close all;
plot3v(sparse_points,'.'); hold on;

I = ( sparse_points(:,3) < -165 ) &...
( sparse_points(:,1) < 950 ) & ...
( sparse_points(:,1) < 1050 ) & ...
( sparse_points(:,2) < 0 ) & ...
( sparse_points(:,2) > -400 );

plot3v(sparse_points(I,:),'.');
%%
sparse_points(I,:) = [];
%%
axang = [0 0 1 pi/2];
rotm = axang2rotm(axang);

rotated = zeros(size(sparse_points));
for i = 1:size(sparse_points,1)
    rotated(i,:) = (rotm*sparse_points(i,:)')';
end
%%
close all;
plot3v(rotated,'.'); axis equal;
xlabel('x');
ylabel('y')
%%
axang = [0 1 0 1.1];
rotm = axang2rotm(axang);

rotated2 = zeros(size(rotated));
for i = 1:size(sparse_points,1)
    rotated2(i,:) = (rotm*rotated(i,:)')';
end

%%
close all;
plot3v(rotated,'.'); axis equal;
xlabel('x');
ylabel('y');

%%
cen_x = 942;
cen_y = 968;
cen_z = 347;

aligned = sparse_points - [cen_x,cen_y,cen_z];

%%

