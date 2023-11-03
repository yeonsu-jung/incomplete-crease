load('zstack_clean.mat')
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
stat = get_principal_axis_length(centered);
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

%%
% ruled surface



%%
I_head = rwnorm(centered(:,1:2) - [0,0]) < 5;
[~,I_max] = max(centered(I_head,3));
tmp = find(I_head);
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
R_list = linspace(350,850,num_R);
% az_list = linspace();
clr = viridis(num_R);
az_range = pi/4;
for i = 10%1:num_R
    R = R_list(i);
    delta_R = 1;
    I_R = rwnorm(distances_to_head - R) < delta_R;
    I_az = rwnorm(az2) < az_range/2;
    
    plot(az2(I_R&I_az),el(I_R&I_az),'.','color',clr(i,:));hold on;

    % islocalmin(el(I_R&I_az))
    ;
end
%%
num_R = 20;
R_list = linspace(350,850,num_R);
% binning and averaging...
num_az = 100;
az_range = pi/4;
delta_az = pi/100;
az_bin = linspace(-az_range/2,az_range/2,num_az);

binned_data = zeros(num_az,num_R);
binned_data_min = zeros(num_az,num_R);
binned_data_max = zeros(num_az,num_R);

% given fixed R
for i = 1:num_R
    I_R = rwnorm(distances_to_head - R_list(i)) < delta_R;
    binned_el = zeros(num_az,1);
    for j = 1:num_az
        I = rwnorm(az2 - az_bin(j)) < delta_az;    
        % plot(az2(I_R&I),el(I_R&I),'.');hold on;    
        binned_data(j,i) = mean(el(I_R&I));
        binned_data_min(j,i) = min(el(I_R&I));
        binned_data_max(j,i) = max(el(I_R&I));
    end
end

%%
close all;
plot(az_bin,binned_data(:,1:end),'.-');hold on;
plot(az_bin,binned_data_min(:,1:end),'ro-');
plot(az_bin,binned_data_max(:,1:end),'bo-');
%%
close all;
dip_positions = zeros(num_R,3);

az_bin_back = mod(az_bin-pi,2*pi);

num_R = 20;
R_list = linspace(350,850,num_R);
for i = 1:num_R
    tmp = binned_data(:,i);
    [el_min,I_min] = min(tmp);
    plot(az_bin,tmp,'.-');hold on;
    plot(az_bin(I_min),el_min,'.-');
    
    pp = spline(az_bin,tmp);
    xx = linspace(min(az_bin),max(az_bin),100);
    yy = spline(az_bin,tmp,xx);
    plot(xx,yy,'-');

    dfdx = fnder(pp);    
    x0 = fnzeros(dfdx);
    x0 = x0(1);
    y0 = ppval(pp,x0);
    plot(x0,y0,'o');    

    % [x,y,z] = sph2cart(az_bin_back(I_min),el_min,R_list(i));
    [x,y,z] = sph2cart(mod(x0-pi,2*pi),y0,R_list(i));
    dip_positions(i,:) = [x,y,z];
end
%%


%%
close all;
plot3v(dip_positions,'o');axis equal;hold on;

[cen,ori] = get_line_coord(dip_positions);
plot3v(cen + ori.*linspace(-1000,1000,100)','o-');


%%
% find the real center
close all;
plot3v(new_centered(1:500:end,:),'.');
hold on;
plot3v(dip_positions,'o');
plot3v(cen + ori.*linspace(-1000,1000,100)','-');
axis equal;

%%
num_line_points = 200;
line_points = cen + ori.*linspace(-750,500,num_line_points)';

R_search = 15;

close all;
plot3v(new_centered(1:500:end,:),'.');hold on;

for i = 1:num_line_points
    I = rwnorm(new_centered - line_points(i,:)) < R_search;
    plot3v(new_centered(I,:),'.');
end

%%
close all;
crease_positions = NaN(num_line_points,3);
for i = 1:num_line_points
    I = rwnorm(new_centered - line_points(i,:)) < R_search;
    % plot3v(new_centered(I,:),'.');hold on;
    % crease_positions(i,:) = mean(new_centered(I,:));
    if nnz(I) > 0
        [~,I_min] = min(new_centered(I,3));
        tmp = find(I);
        crease_positions(i,:) = new_centered(tmp(I_min),:);
    end
end
%%
close all;
plot3v(crease_positions,'.');hold on;

[~,I_max] = max(crease_positions(:,3));
[~,I_min] = min(crease_positions(:,3));

real_center = crease_positions(I_max,:);
tail = crease_positions(I_min,:);
plot3v(real_center,'ro','linewidth',3);
plot3v(tail,'ro','linewidth',3);
plot3v(new_centered(1:500:end,:),'.');
%%
[az_crease,el_crease,r_crease] = cart2sph(crease_positions(60:end,1),crease_positions(60:end,2),crease_positions(60:end,3));
% [az_crease,el_crease,r_crease] = cart2sph(dip_positions(:,1),dip_positions(:,2),dip_positions(:,3));

%%
mean(az_crease,'omitnan')
median(az_crease,'omitnan')
%%
close all;
plot(az_crease)
%%
close all;
plot3v(new_centered(1:500:end,:),'.'); hold on;
plot3v(real_center,'ro','linewidth',2);
plot3v(tail,'ro','linewidth',2);
%%
new_new_centered = new_centered - real_center;
new_new_tail = tail - real_center;

%%
close all;
plot3v(new_new_centered(1:500:end,:),'.');

%%
% [az_tail,el_tail,r_tail] = cart2sph(new_new_tail(:,1),new_new_tail(:,2),new_new_tail(:,3));
az_tail = median(az_crease,'omitnan');
%%
[new_az,new_el,new_r] = cart2sph(new_new_centered(:,1),new_new_centered(:,2),new_new_centered(:,3));
%%
[x,y,z] = sph2cart(new_az - az_tail,new_el - el_tail,new_r);
new_new_centered2 = [x,y,z];
close all;
plot3v(new_new_centered(1:500:end,:),'.');hold on;
plot3v(new_new_centered2(1:500:end,:),'.');
grid on
%%
close all;
plot(new_new_centered2(1:500:end,2),new_new_centered2(1:500:end,3),'.');
grid on

%%
% rotx = @(t) [1 0 0; 0 cos(t) -sin(t) ; 0 sin(t) cos(t)] ;
% roty = @(t) [cos(t) 0 sin(t) ; 0 1 0 ; -sin(t) 0  cos(t)] ;
% rotz = @(t) [cos(t) -sin(t) 0 ; sin(t) cos(t) 0 ; 0 0 1] ;

R = rotx(0.1);

rotated = zeros(size(new_new_centered2));
for i = 1:size(new_new_centered2,1)
    pt = new_new_centered2(i,:);
    rotated(i,:) = (R*pt')';
end
%%
clc
x0 = -0.005;
% options = ;
x_opt = fminsearch(@(x) objective_function(new_new_centered2,x),x0);
% objective_function(new_new_centered2,x_opt)

%%
rotated = zeros(size(new_new_centered2));
for i = 1:size(new_new_centered2,1)
    pt = new_new_centered2(i,:);
    rotated(i,:) = (rotx(x_opt)*pt')';
end
%
close all;
% plot3v(new_new_centered2(1:500:end,:),'.');hold on;
plot3v(rotated(1:500:end,:),'.');
axis equal;

%%
[az,el,r] = cart2sph(rotated(:,1),rotated(:,2),rotated(:,3));

%%
num_R = 35;
R_list = linspace(10,850,num_R);
delta_R = 1;

close all;
for i = 1:num_R
    R = R_list(i);    
    I_R = rwnorm(r - R) < delta_R;    
    plot3v(rotated(I_R,:),'.');hold on;

    % plot(az(I_R),el(I_R),'.');
    

end
xlabel('x');
ylabel('y');
zlabel('z');

%%

R = 300;
I_R = rwnorm(r - R) < delta_R;    

close all;
plot3v(rotated(I_R,:),'.');
%%
close all;
num_R = 5;
R_list = linspace(350,850,num_R);

I_az = abs(az) < 0.1;

for i = 2%1:num_R
    R = R_list(i);
    I_R = rwnorm(r - R) < delta_R;    
    plot(az(I_R&I_az),el(I_R&I_az),'.');hold on;
end

%%
num_az = 100;
az_range = 0.5;
delta_az = 0.01;
offset = -0.02;
az_bin = linspace(-az_range/2-offset,az_range/2-offset,num_az)';

num_R = 5;
binned_data = zeros(num_az,num_R);
binned_data2 = zeros(num_az,num_R);
R_list = linspace(350,850,num_R);

% az2 = mod(az+pi,-2*pi);
% given fixed R

binned_data = zeros(num_R,num_az);
binned_data2 = zeros(num_R,num_az);
binned_data3 = zeros(num_R,num_az);

for i = 1:num_R
    I_R = rwnorm(r - R_list(i)) < delta_R;
    binned_el = zeros(num_az,1);
    for j = 1:num_az
        I = rwnorm(az - az_bin(j)) < delta_az;
        % plot(az2(I_R&I),el(I_R&I),'.');hold on;
        binned_data(j,i) = min(el(I_R&I));
        binned_data2(j,i) = max(el(I_R&I));
        binned_data3(j,i) = mean(el(I_R&I));
    end
end

%%
RR = repmat(R_list(i),[numel(az_bin),1]);

clc
[x,y,z] = sph2cart(az(I_R&I),binned_data(:,i),RR);

%%
close all;
plot(binned_data3);hold on;
plot(binned_data);
plot(binned_data2);
%%
% crease angle - max
close all;
plot(az_bin,binned_data2,'.-');

%%
close all;
plot(az_bin,binned_data3(:,1),'o');hold on;

i = 1;
tmp = binned_data3(:,i);
[~,I_min] = min(tmp);

plot(az_bin(I_min),tmp(I_min),'ro','linewidth',3);

plot(az_bin(I_min-5:I_min-1),tmp(I_min-5:I_min-1),'ro-');
plot(az_bin(I_min:I_min+4),tmp(I_min:I_min+4),'ro-');

%%
pp = spline(az_bin,tmp);
dfdx = fnder(pp);

x0 = fnzeros(dfdx);
x0 = x0(1);
y0 = ppval(pp,x0);

close all;
plot(az_bin,binned_data3(:,1),'o');hold on;
fnplt(pp);
plot(x0,y0,'ro','linewidth',3);
%
[~,I_min] = min(abs(tmp - y0));
plot(az_bin(I_min+5:I_min+15),tmp(I_min+5:I_min+15),'ro-');
plot(az_bin(I_min-16:I_min-6),tmp(I_min-16:I_min-6),'bo-');

x_left = az_bin(I_min+5:I_min+15);
y_left = tmp(I_min+5:I_min+15);

x_right = az_bin(I_min-16:I_min-6);
y_right = tmp(I_min-16:I_min-6);

%%
p_left = polyfit(x_left,y_left,1);
p_right = polyfit(x_right,y_right,1);
%%
p_left
p_right
% goodness of fitting?

%%
close all;
plot3(x,y,z);
hold on;
plot3v(rotated(I_R,:),'.');
%%
close all;
plot(az_bin,binned_data,'ro-');
hold on;
plot(az_bin,binned_data2,'bo-');

% need a bit of re-adjustment; why?
%%
close all;
plot(az_bin,binned_data,'ro-');

%%
% so, crease angle...
num_az = 100;
az_range = pi/2;
delta_az = pi/100;

num_R = 1;
delta_R = 1;
binned_data_fine = zeros(num_az,num_R);
R_list = linspace(350,850,num_R);

az2 = mod(az+pi/2,-2*pi);
az2 = az;
az2(az > pi) = az(az > pi) - 2*pi;

az_bin = linspace(-az_range/2,az_range/2,num_az);

% given fixed R
close all;

clr = viridis(num_az);
for i = 1:num_R
    R = R_list(i);
    R = 500;
    I_R = rwnorm(r - R) < delta_R;
    binned_el = zeros(num_az,1);
    for j = 1:num_az
        I = rwnorm(az - az_bin(j)) < delta_az;
        binned_data_fine(j,i) = mean(el(I_R&I));
        % plot(az2(I_R&I),el(I_R&I),'.');hold on;

        plot3v(rotated(I_R&I,:),'.');hold on;

        % plot(az(I_R&I),el(I_R&I),'.');hold on;    
        % plot(mean(az(I_R&I)),mean(el(I_R&I)),'.');hold on;
        % plot3v(new_new_centered2(I_R&I,:),'.','colo/r',clr(j,:));hold on;

    end
    ;
end
% axis equal;

%%
function score = objective_function(centered,angx)

% tangent
% binormal
rotated = zeros(size(centered));
R = rotx(angx);
for i = 1:size(centered,1)
    pt = centered(i,:);
    rotated(i,:) = (R*pt')';
end

I_neg = rotated(:,2) < 0;
I_pos = rotated(:,2) >= 0;

score = abs(mean(rotated(I_pos,3)) - mean(rotated(I_neg,3)));


end

function out = rotx(t)
out = [1 0 0; 0 cos(t) -sin(t) ; 0 sin(t) cos(t)];

end

function project_to_plane(point_vector,plane_normal)
% point vectors are anchored on the origin
    N = size(point_vector);
    foot_dist = rwnorm( point_vector.*repmat(plane_normal,[N,1]) );
    ;
end


