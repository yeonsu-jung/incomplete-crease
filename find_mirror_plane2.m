load('zstack_clean.mat')
%%
eroded = imerode(zstack_clean,strel('sphere',2));
close all;
volshow(eroded);
close all;
volshow(zstack_clean&~eroded);

%%
crop = imcrop3(eroded,[700,700,1,400,400,370]);
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

% plot3v(all_points(I,:),'.');
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
stat = get_principal_axis_length(centered);
%%
plane_normal = stat.EigenVectors(:,2)';
plane_tangent1 = stat.EigenVectors(:,1)';
plane_tangent2 = stat.EigenVectors(:,3)';
%%
close all;
plot3v(centered(1:500:end,:),'.');
hold on;
quiver3v([0,0,0],-plane_tangent1*1000,'r');
quiver3v([0,0,0],plane_tangent2*1000,'b');
quiver3v([0,0,0],plane_normal*1000,'k');
%%
x1 = sum(centered.*stat.EigenVectors(:,1)',2);
y1 = sum(centered.*stat.EigenVectors(:,2)',2);

close all;
plot(x1(1:500:end),y1(1:500:end),'.');axis equal;
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
num_angles = 1000;
angle_list = linspace(-pi/4,pi/4,num_angles);
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
plot3v(edge_point_list,'o');axis equal;
%
[~,I_min] = min(edge_point_list(:,3));
tail = edge_point_list(I_min,:);
hold on;
plot3v(tail,'ro','markersize',10,'linewidth',5);
plot3v(centered(1:500:end,:),'.');
%%
% ruled surface



%%
I_head = rwnorm(centered(:,1:2) - [0,0]) < 5;
[~,I_max] = max(centered(I_head,3));
tmp = find(I_head);
%%
head = centered(tmp(I_max),:);
%%
close all;
plot3v(edge_point_list,'o');axis equal;
%
[~,I_min] = min(edge_point_list(:,3));
tail = edge_point_list(I_min,:);
hold on;
plot3v(head,'ro','markersize',10,'linewidth',5);
plot3v(tail,'ro','markersize',10,'linewidth',5);
plot3v(centered(1:500:end,:),'.');
%%
close all;
plot3v(edge_point_list,'.-');axis equal;hold on;
plot3v(tail,'ro','markersize',10);
%%
new_centered = centered - head;

close all;
plot3v(new_centered(1:500:end,:),'.');
%%
[az,el,distances_to_head] = cart2sph(new_centered(:,1),new_centered(:,2),new_centered(:,3));

az2 = az + pi;
I = az2 > pi;
az2(I) = az2(I) - 2*pi;
%%
distances_to_head = rwnorm(new_centered);

%%
close all;
num_R = 35;
R_list = linspace(10,850,num_R);

delta_R = 1;

for i = 1:num_R
    R = R_list(i);
    
    I_R = rwnorm(distances_to_head - R) < delta_R;
    I_az = rwnorm(az2) < pi/4;
    plot3v(new_centered(I_R&I_az,:),'.');hold on;
end
axis equal;
%%
i = 25;
R = R_list(i);
delta_R = 1;
I_R = rwnorm(distances_to_head - R) < delta_R;
plot3v(new_centered(I_R,:),'.');hold on;

close all;
plot3v(new_centered(I_R,:),'.');axis equal; 
%%
close all;
num_R = 20;
R_list = linspace(100,850,num_R);
% az_list = linspace();
clr = viridis(num_R);
az_range = pi/4;
for i = 1:num_R
    R = R_list(i);
    delta_R = 1;
    I_R = rwnorm(distances_to_head - R) < delta_R;
    I_az = rwnorm(az2) < az_range/2;
    
    plot(az2(I_R&I_az),el(I_R&I_az),'.','color',clr(i,:));hold on;

    % islocalmin(el(I_R&I_az))
    ;
end
%%
% binning and averaging...
addpath(genpath('../entanglement/functions'))
%%
num_az = 10;
delta_az = pi/100;
az_bin = linspace(-az_range/2,az_range/2,num_az);
for i = 1:num_az
    I = rwnorm(az2 - az_bin(i)) < delta_az;

    plot(az2(I_R&I_az),el(I_R&I_az),'.');hold on;
end
%%
close all;
plot3v(new_centered(I_R&I_az,:),'.');
%%
lb = point_segmentation(new_centered(I_R&I_az,:));
max(lb)


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


