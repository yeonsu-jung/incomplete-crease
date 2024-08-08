function data_analysis_template(file_path)
addpath(genpath('../entanglement/functions'));
%%
% run('../entanglement/script/setup')
%%
% dir('/Users/yeonsu/Dropbox (Harvard University)/xray-data/z-stacks')
%%
% file_path='/Users/yeonsu/Dropbox (Harvard University)/xray-data/z-stacks/zStack_SS_Mid_Standing_01/zstack.mat';
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

%%
small = imresize3(cleaned,0.3);
volshow2(small);
%%
save(sprintf('small_stack_%s',sample_name),'small','-v7.3');
%%
% close all;
% volshow(cleaned);

pts = ind2sub2(size(cleaned),find(cleaned));
%%
%%
% Assuming your Nx3 point cloud is stored in a variable named 'yourPointCloud'
% [normal, pointOnPlane] = fitPlaneSVD(centered(1:250:end,:));

%%
centered = pts - mean(pts);
transformedPointCloud = transformPointsToXYPlane(centered(1:100:end,:));
%%
% rotz = @(t) [cos(t) -sin(t) 0 ; sin(t) cos(t) 0 ; 0 0 1] ;
% R = rotz(pi);
% rt = zeros(size(centered));
% for i = 1:size(centered,1)
%     pt = centered(i,:);
%     rt(i,:) = (R*pt')';
% end
%%
pc = pointCloud(transformedPointCloud);
pcshow(pc)

%%
close all;
plot3v(transformedPointCloud,'.');
axis equal;

%%
[az,el,r] = cart2sph(transformedPointCloud(:,1),transformedPointCloud(:,2),transformedPointCloud(:,3));

%%
% binning and averaging...
num_R = 20;
delta_R = 1;
R_list = linspace(100,max(r),num_R);

%%
i_start = 1;
i_end = 15;

figure;
for i = i_start:i_end
    I_R = rwnorm(r.*cos(el) - R_list(i)) < delta_R*2;
    tmp = find(I_R);
    plot3v(transformedPointCloud(tmp,:),'.');hold on;
end
hold on;

%%
figure;
lb = {};
for i = 1:3:num_R
    I_R = rwnorm(r.*cos(el) - R_list(i)) < delta_R;
    tmp = find(I_R);
    xx = az(tmp);
    yy = el(tmp);
    [xx2,I] = sort(xx);
    yy2 = yy(I)
    plot(xx2,yy2,'-');hold on;

    lb{end+1} = sprintf('r = %.2f mm', R_list(i)*0.2);
end
xlabel('Azimuthal angle, $\phi$','interpreter','latex')
ylabel('Elevation angle, $\theta$','interpreter','latex')
legend(lb);

print(gcf,sprintf('asymmetry_%s.png',sample_name),'-dpng','-r300');

end