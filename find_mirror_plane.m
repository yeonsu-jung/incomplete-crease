load('zstack_clean.mat')
%%
crop = imcrop3(zstack_clean,[700,700,1,400,400,370]);
volshow2(crop);
%%
ind = find(zstack_clean);
center = [968,942,347];
[X,Y,Z] = ind2sub(size(zstack_clean),ind);
point_cloud = [X,Y,Z];
point_cloud = point_cloud - center;

%% point_cloud - center;
addpath(genpath('../entanglement/functions'))
all_points = point_cloud(1:1:end,:);
close all;
plot3v(all_points,'.'); hold on;

I = ( all_points(:,3) < -165 ) &...
( all_points(:,1) < 950 ) & ...
( all_points(:,1) < 1050 ) & ...
( all_points(:,2) < 0 ) & ...
( all_points(:,2) > -400 );

plot3v(all_points(I,:),'.');
%%
all_points(I,:) = [];
%%
close all;
plot3v(all_points,'.');
%%
close all;
plot_pointcloud(all_points);

%%
centered = all_points - mean(all_points);
%%
centered = centered(1:100:end,:);
%%
stat = get_principal_axis_length(centered);
%%
plane_normal = stat.EigenVectors(:,2)';
plane_tangent1 = stat.EigenVectors(:,1)';
plane_tangent2 = stat.EigenVectors(:,3)';
%%
close all;
plot3v(centered,'.');
hold on;
quiver3v([0,0,0],plane_tangent1*1000,'r');
quiver3v([0,0,0],plane_tangent2*1000,'b');
quiver3v([0,0,0],plane_normal*1000,'k');
%%
[az,el,r] = cart2sph(centered(:,1),centered(:,2),centered(:,3));
%%
close all;
delta_phi = pi/100;


az2 = az + pi;
I = az2 > pi;
az2(I) = az2(I) - 2*pi;

%%
close all;
num_angles = 100;
angle_list = linspace(-pi/4,pi/4,num_angles);
delta_phi = pi / 200;
for i = 1:num_angles
    angl = angle_list(i);
    I = rwnorm(az2 - angl) < delta_phi;
    plot3v(centered(I,:),'.');hold on;

    [~,I_rmax] = max(r(I));

    tmp = find(I);
    
    text(centered(tmp(I_rmax),1),centered(tmp(I_rmax),2),centered(tmp(I_rmax),3),...
        sprintf('%d',i));
end
axis equal;
%%
close all;
num_angles = 1000;
angle_list = linspace(-pi,pi,num_angles);
delta_phi = pi / 200;

edge_point_list = zeros(num_angles,3);
for i = 1:num_angles
    angl = angle_list(i);
    I = rwnorm(az2 - angl) < delta_phi;
    [~,I_rmax] = max(r(I));
    
    tmp = find(I);    
    % plot3(centered(tmp(I_rmax),1),centered(tmp(I_rmax),2),centered(tmp(I_rmax),3),'o');hold on;
    edge_point_list(i,:) = centered(tmp(I_rmax),:);
end
axis equal;
%%
close all;
plot3v(edge_point_list);axis equal;
%%
num_points = size(centered,1);
t1 = sum(centered.*repmat(plane_tangent1,[num_points,1]),2);
t2 = sum(centered.*repmat(plane_tangent2,[num_points,1]),2);
n = sum(centered.*repmat(plane_normal,[num_points,1]),2);
%
close all;
plot(t1(n>0),t2(n>0),'.');
hold on;
plot(t1(n<0),t2(n<0),'.');

%%

plane_normal = [1,2,0];
plane_normal = plane_normal/norm(plane_normal);

plane_tangent1 = 1;

N = size(centered,1);
foot_dist = rwnorm( centered.*repmat(plane_normal,[N,1]) );

%%
centroid = mean(all_points);
centered = all_points - centroid;

%%
[az,el,r] = cart2sph(centered(:,1),centered(:,2),centered(:,3));
%%
clc
close all;
plot(az(1:100:end,:),el(1:100:end,:),'.');
%%
plane_normal = [1,2,0];
plane_normal = plane_normal/norm(plane_normal);

plane_tangent1 = 1;

N = size(centered,1);
foot_dist = rwnorm( centered.*repmat(plane_normal,[N,1]) );
%%
signed_foot_dist = sum(centered.*repmat(plane_normal,[N,1]),2);
I_positive = signed_foot_dist >= 0;
I_negative = signed_foot_dist < 0;
%%
positive_dist = sort(signed_foot_dist(I_positive,:));
negative_dist = sort(-signed_foot_dist(I_negative,:));

close all;
plot(positive_dist,'.');hold on;
plot(negative_dist,'.');


%%
foot_dist = sort(foot_dist);

close all;
plot(foot_dist,'.')
%%
project_to_plane(centered,[0,1,0])
%%
function objective_function(centered,plane_normal)

% tangent
% binormal


end

function project_to_plane(point_vector,plane_normal)
% point vectors are anchored on the origin
    N = size(point_vector);
    foot_dist = rwnorm( point_vector.*repmat(plane_normal,[N,1]) );
    ;
end


