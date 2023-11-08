addpath(genpath('../entanglement/functions'));
%%
dir('/Users/yeonsu/Dropbox (Harvard University)/xray-data/z-stacks')
%%
file_path='/Users/yeonsu/Dropbox (Harvard University)/xray-data/z-stacks/zStack_InverseHalfCrease_04/zstack.mat';
load(file_path)
%%
[folder_name,file_name] = fileparts(file_path);

%%
split_path = regexp(folder_name,'[\\/]','split');
sample_name = split_path{end};
split_path2 = regexp(sample_name,'[\_]','split');
sample_name = horzcat(split_path2{2:end});

%%
% cleaning zstack
% cleaned = imerode(bwmorph3(zstack,'clean'),1);
cleaned = bwmorph3(imerode(zstack,1),'clean');

close all;
volshow(cleaned);

%%
cc = bwconncomp(cleaned);
%%
num_pixels = cellfun(@numel,cc.PixelIdxList);

%%
[~,I_max] = max(num_pixels);

%%
ind = cc.PixelIdxList{I_max};
%%
pts = ind2sub2(size(zstack),ind);
%%
centered = pts - mean(pts);

rotz = @(t) [cos(t) -sin(t) 0 ; sin(t) cos(t) 0 ; 0 0 1] ;
R = rotz(pi);
rt = zeros(size(centered));
for i = 1:size(centered,1)
    pt = centered(i,:);
    rt(i,:) = (R*pt')';
end

close all;
plot3v(rt(1:500:end,:),'o');
%%
% crease at -y axis
;
%%
centered = rt;
clear rt

close all;
plot3v(centered(1:500:end,:),'.');

%%
close all;
plot(centered(1:500:end,1),centered(1:500:end,2),'.');hold on;
plot(0,0,'ro','linewidth',2);

%%
[az,el,r] = cart2sph(centered(:,1),centered(:,2),centered(:,3));
%%
% binning and averaging...
num_R = 20;
delta_R = 1;
R_list = linspace(350,max(r),num_R);

%%
i_start = 1;
i_end = 15;

close all;
for i = i_start:i_end
    I_R = rwnorm(r - R_list(i)) < delta_R;    
    tmp = find(I_R);    
    plot3v(centered(tmp(1:50:end),:),'.');hold on;
end
%%
close all;
for i = 1:num_R
    I_R = rwnorm(r - R_list(i)) < delta_R;
    tmp = find(I_R);
    
    plot(az(tmp),el(tmp),'.');hold on;
end

%%
num_az = 100;
az_range = 1;
offset = -0.02;
delta_az = pi/100;

az_bin = linspace(-az_range/2+offset,az_range/2+offset,num_az);

binned_data = zeros(num_az,num_R);
binned_data_min = zeros(num_az,num_R);
binned_data_max = zeros(num_az,num_R);

% given fixed R
for i = 1:num_R
    I_R = rwnorm(r - R_list(i)) < delta_R;
    binned_el = zeros(num_az,1);
    for j = 1:num_az
        I = rwnorm(az - az_bin(j)) < delta_az;
        % plot(az2(I_R&I),el(I_R&I),'.');hold on;
        if nnz(I_R&I) > 0
            binned_data(j,i) = mean(el(I_R&I));
            binned_data_min(j,i) = min(el(I_R&I));
            binned_data_max(j,i) = max(el(I_R&I));
        end
    end
end

%%
close all;
plot(az_bin,binned_data(:,i_start:i_end),'.-');hold on;
plot(az_bin,binned_data_min(:,i_start:i_end),'ro-');
plot(az_bin,binned_data_max(:,i_start:i_end),'bo-');

%%
close all;
dip_positions = zeros(i_start - i_end + 1,3);
num_R = 20;
for i = i_start:i_end
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
    [x,y,z] = sph2cart(x0,y0,R_list(i));
    dip_positions(i,:) = [x,y,z];
end
%%
close all;
plot3v(dip_positions,'o');axis equal;hold on;

[cen,ori] = get_line_coord(dip_positions);
plot3v(cen + ori.*linspace(-1000,1000,100)','o-');


%%
% find the real center
close all;
plot3v(centered(1:500:end,:),'.');
hold on;
plot3v(dip_positions,'o');
plot3v(cen + ori.*linspace(-1000,1000,100)','-');
axis equal;

%%
num_line_points = 200;
line_points = cen + ori.*linspace(-750,500,num_line_points)';

R_search = 30;
close all;
plot3v(centered(1:500:end,:),'.');hold on;

for i = 1:num_line_points
    I = rwnorm(centered - line_points(i,:)) < R_search;
    plot3v(centered(I,:),'.');
end

%%
close all;
crease_positions = NaN(num_line_points,3);
for i = 1:num_line_points
    I = rwnorm(centered - line_points(i,:)) < R_search;
    % plot3v(centered(I,:),'.');hold on;axis equal;
    ;
    % crease_positions(i,:) = mean(new_centered(I,:));
    if nnz(I) > 0
        % [~,I_min] = min(centered(I,3));
        % tmp = find(I);
        % crease_positions(i,:) = centered(tmp(I_min),:);
        % plot3v(centered(tmp(I_min),:),'ro','linewidth',2);
        crease_positions(i,:) = mean(centered(I,:),1);      
        % plot3v(crease_positions(i,:),'ro','linewidth',2);
        ;
    end
end
%%
% x0 = find_critical_points(1:num_line_points,crease_positions);

%%
close all;
plot3v(crease_positions,'.');hold on;
[~,I_max] = max(crease_positions(:,3));
[~,I_min] = min(crease_positions(:,3));

real_center = crease_positions(I_max,:);
tail = crease_positions(I_min-24,:);

plot3v(real_center,'ro','linewidth',3);
plot3v(tail,'bo','linewidth',3);
plot3v(centered(1:500:end,:),'.');
%%

%%
[az_crease,el_crease,r_crease] = cart2sph(dip_positions(:,1),dip_positions(:,2),dip_positions(:,3));
% [az_crease,el_crease,r_crease] = cart2sph(dip_positions(:,1),dip_positions(:,2),dip_positions(:,3));

mean(az_crease,'omitnan')
median(az_crease,'omitnan')
%%
close all;
plot(az_crease)
%%
close all;
plot3v(centered(1:500:end,:),'.'); hold on;
plot3v(real_center,'ro','linewidth',2);
plot3v(tail,'ro','linewidth',2);
%%
head_tail_distance = norm(tail-real_center);
new_centered = centered - real_center;
new_tail = tail - real_center;

[new_az,new_el,new_r] = cart2sph(new_centered(:,1),new_centered(:,2),new_centered(:,3));
%%
unwanted = new_centered(new_r > head_tail_distance,:);
wanted = new_centered(new_r <= head_tail_distance,:);

close all;
plot3v(unwanted,'.');hold on;
plot3v(wanted(1:500:end,:),'.');

%%
new_centered = wanted;
;
%%
;
%%
[az_tail,el_tail,r_tail] = cart2sph(new_tail(:,1),new_tail(:,2),new_tail(:,3));
az_tail = median(az_crease,'omitnan');
%%
[new_az,new_el,new_r] = cart2sph(new_centered(:,1),new_centered(:,2),new_centered(:,3));

%%
[x,y,z] = sph2cart(new_az - az_tail,new_el - el_tail,new_r);
new_centered2 = [x,y,z];
close all;
plot3v(new_centered(1:500:end,:),'.');hold on;
plot3v(new_centered2(1:500:end,:),'.');
grid on
%%
close all;
plot(new_centered2(1:500:end,2),new_centered2(1:500:end,3),'.');
grid on

%%
% roty = @(t) [cos(t) 0 sin(t) ; 0 1 0 ; -sin(t) 0  cos(t)] ;
% rotz = @(t) [cos(t) -sin(t) 0 ; sin(t) cos(t) 0 ; 0 0 1] ;

R = rotx(0.1);

rotated = zeros(size(new_centered2));
for i = 1:size(new_centered2,1)
    pt = new_centered2(i,:);
    rotated(i,:) = (R*pt')';
end

%%
clc
x0 = -0.005;
% options = ;
x_opt = fminsearch(@(x) objective_function(new_centered2,x),x0);
% objective_function(new_centered2,x_opt)

%%
rotated = zeros(size(new_centered2));
for i = 1:size(new_centered2,1)
    pt = new_centered2(i,:);
    rotated(i,:) = (rotx(x_opt)*pt')';
end
%
close all;
% plot3v(new_centered2(1:500:end,:),'.');hold on;
plot3v(rotated(1:500:end,:),'.');
axis equal;

%%
close all;
plot(rotated(1:500:end,2),rotated(1:500:end,3),'.');
print(sprintf('results/%s_yz_projection.png',sample_name),'-dpng','-r600');

% done with rotation

%%
[az,el,r] = cart2sph(rotated(:,1),rotated(:,2),rotated(:,3));

%%
num_R = 35;
R_list = linspace(50,head_tail_distance,num_R);
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
offset = +0.02;
az_bin = linspace(-az_range/2+offset,az_range/2+offset,num_az)';

num_R = 5;
R_list = linspace(350,700,num_R);

binned_data = zeros(num_R,num_az);
binned_data2 = zeros(num_R,num_az);
binned_data3 = zeros(num_R,num_az);

for i = 1:num_R
    I_R = rwnorm(r - R_list(i)) < delta_R;
    binned_el = zeros(num_az,1);
    for j = 1:num_az
        I = rwnorm(az - az_bin(j)) < delta_az;
        % plot(az2(I_R&I),el(I_R&I),'.');hold on;
        binned_data(i,j) = min(el(I_R&I));
        binned_data2(i,j) = max(el(I_R&I));
        binned_data3(i,j) = mean(el(I_R&I));
    end
end

%%
close all;
plot(binned_data3');hold on;
plot(binned_data');
plot(binned_data2');

%%
% crease angle - max
close all;
plot(az_bin,binned_data,'.-');

%%
close all;
plot(az_bin,binned_data3(1,:),'o');hold on;

i = 1;
tmp = binned_data3(i,:);
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
plot(az_bin,binned_data3(i,:),'o');hold on;
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
p_left = polyfit(x_left,y_left,1)
p_right = polyfit(x_right,y_right,1)
%%

% goodness of fitting?
%%

binned_data_all = zeros(num_R,num_az);
az_bin_all = linspace(-pi,pi,num_az)';

for i = 1:num_R
    I_R = rwnorm(r - R_list(i)) < delta_R;
    binned_el = zeros(num_az,1);

    for j = 1:num_az
        I = rwnorm(az - az_bin_all(j)) < delta_az;
        % plot(az2(I_R&I),el(I_R&I),'.');hold on;
        
        binned_data_all(i,j) = mean(el(I_R&I));        
    end
end
%%
close all;
plot(az_bin_all',binned_data_all','o-');
print(sprintf('results/%s_psi_plot.png',sample_name),'-dpng','-r600');

%%
data_output.p_left = p_left;
data_output.p_right = p_right;

data_output.R_list = R_list;
data_output.az_bin_all = az_bin_all;
data_output.binned_data_all = binned_data_all;

save(sprintf('results/%s_data_output',sample_name),'data_output','-v7.3');
