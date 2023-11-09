%%
clear all
close all
 
out_path = '/Users/yeonsu/Dropbox (Harvard University)/xray-data/z-stacks';
image_dir_path = '/Users/yeonsu/Dropbox (Harvard University)/xray-data/zStack_InverseHalfCrease_08';

fname = dir([image_dir_path,'/*.tif']);
num_img = numel(fname)
 
split_path = regexp(image_dir_path,'[\\/]','split');
sample_name = split_path{end}
 
%%
clc
zl1 = 1;
zl2 = num_img;
 
img1 =  imread(fullfile(image_dir_path,fname(zl1).name),'tif');
img2 =  imread(fullfile(image_dir_path,fname(zl2).name),'tif');
figure(1);imshowpair(img1,img2,'montage')

%%
img = imread(fullfile(image_dir_path,fname(150).name));
img = imdiffusefilt(img);
figure(2)
imshow(img,[])
%%
close all;
histogram(img)
 
%%
T = graythresh(img)
T2 = T * 65535
BW = imbinarize(img,T);
% close all;imshowpair(img>3.24e4,BW,'montage')
TH = 1.35e4;
close all;imshowpair(img>TH,BW,'montage')
%%
filtered = imdiffusefilt(img);
close all;imshowpair(img>TH,filtered>TH,'montage')
%%
figure;imshow(img>3.2e4,[]);
%%
clc
% [m,n] = size(img);
[m,n] = size(img);
 
%%
close all
zstack = zeros(m,n,zl2-zl1+1,'logical');
 
pages = 1;
tic
for idx = zl1:zl2
    raw = imread( fullfile(image_dir_path,fname(idx).name) );
%     raw = raw.*uint16(mask);
%     raw = imcrop(raw,pos);
%     filtered = imdiffusefilt(raw);
%     binarized = imbinarize(filtered,T);
    binarized = imbinarize(raw,T*0.95);
    % binarized = raw > TH;
    zstack(:,:,pages) = binarized;
    pages = pages + 1;
end
toc
%
full_stack_folder = fullfile(out_path,sample_name)
mkdir(full_stack_folder)
%%
% stack_path = fullfile(full_stack_folder,'full_stack0.mat')
stack_path = fullfile(full_stack_folder,'zstack.mat')
save(stack_path,'zstack','-v7.3');
%%
close all;
crop = imcrop3(zstack,[400,400,400,499,499,499]);
volshow(crop)
%%
close all;
% small = imresize3(zstack,0.5);
volshow(zstack);
%%
eroded = imerode(stack,strel('sphere',2));
cc = bwconncomp(eroded);
volume_list = zeros(cc.NumObjects,1);
for i = 1:cc.NumObjects
    volume_list(i) = size(cc.PixelIdxList{i},1);
end
[M,I] = sort(volume_list,'descend');
 
M(1:8)
I(1:8)
%
num_obj = 7;
close all;
for ii = 1:num_obj
    [xx,yy,zz] = ind2sub(cc.ImageSize,cc.PixelIdxList{I(ii)});
    plot3(xx,yy,zz);hold on;
end
axis equal;
%
cc2 = cc;
cc2.NumObjects = num_obj;
cc2.PixelIdxList = cc.PixelIdxList([I(1:num_obj)]);
non_rods = imdilate(labelmatrix(cc2)>0,strel('sphere',6));
stack2 = stack&~non_rods;
stack2 = imopen(stack2,strel('sphere',1)); % erosion followed by dilation
%
close all;
volshow(imresize3(stack2,0.5));
 
%%
close all;h = volshow(imresize3(stack2,0.5));
h.BackgroundColor = 'w';
print(gcf,fullfile(out_path,'img_check.png'),'-dpng','-r300');
 
%%
cc3 = bwconncomp(stack);
stats = regionprops3(cc3,'volume');
i_small = find(stats.Volume < 20000);
size(i_small)
%%
[xx,yy,zz] = ind2sub(size(stack),vertcat(cc3.PixelIdxList{i_small}));
close all;plot3(xx,yy,zz,'o');
 
%%
stack = stack2;
stack_path = fullfile(full_stack_folder,'full_stack.mat')
save(stack_path,'stack','-v7.3')
 
%%
% stack = stack(:,:,88:end);
eroded = imerode(stack,strel('sphere',4));
cc = bwconncomp(eroded);
volume_list = zeros(cc.NumObjects,1);
for i = 1:cc.NumObjects
    volume_list(i) = size(cc.PixelIdxList{i},1);
end
[M,I] = sort(volume_list,'descend');
 
M(1:5)
I(1:5)
 
cc2 = cc;
cc2.NumObjects = 4;
cc2.PixelIdxList = cc.PixelIdxList([I(1:4)]);
non_rods = imdilate(labelmatrix(cc2)>0,strel('sphere',6));
stack = stack&~non_rods;
stack = imopen(stack,strel('sphere',2)); % erosion followed by dilation
 
stack_path = fullfile(full_stack_folder,'full_stack.mat')
save(stack_path,'stack','-v7.3')
 
%%
% close all
% figure
% orthosliceViewer(stack)
% 
% figure
% volshow(imresize3(eroded,0.5));
% 
% figure
% for i = 1:3
%     [x,y,z] = ind2sub(size(stack),cc.PixelIdxList{I(i)});
%     plot3(x,y,z);
%     hold on
% end
% axis equal
% 
% figure
% volshow(imresize3(rod_stack,0.5));
% 
figure
volshow(imresize3(stack,0.5));
 
%%
% size(img)
% 
% cx = size(img,2)/2;
% cy = size(img,1)/2;
% R = size(img,1)/2*0.92;
% 
% [X,Y] = meshgrid((1:3024)-cx,(1:3064)-cy);
% mask = sqrt(X.^2+Y.^2) < R;
% pos = [cx-R,cy-R,2*R,2*R];
% % cropped = img.*uint16(mask);
% 
% mask2 = sqrt(X.^2+Y.^2) > R;
% cropped = img;
% cropped(mask2) = NaN;
% 
% figure(3);
% imshow(img)
% rectangle('position',pos,'curvature',[1 1],'edgecolor','r')
% %%
% cx = 1000;
% cy = 1000;
% R = 960;
% 
% % cx = 1000;
% % cy = 1150;
% % R = 560;
% 
% [X,Y] = meshgrid((1:2000)-cx,(1:2000)-cy);
% mask = sqrt(X.^2+Y.^2) < R;
% pos = [cx-R,cy-R,2*R,2*R];
% cropped = img.*uint16(mask);
% 
% 
% figure(3);
% imshow(img)
% rectangle('position',pos,'curvature',[1 1],'edgecolor','r')
% 
% %% for rectangular crop
% 
% pos = [360 520 1250 1250];
% figure(3);
% imshow(img)
% rectangle('position',pos)
% 
% cropped = imcrop(img,pos);
% figure
% imshow(cropped,[])
% %%
% max_intensity = double(max(cropped(:)));
% bin_edges = linspace(1,max_intensity,50);
% 
% figure;histogram(cropped,bin_edges);
% %%
% thres = 2.2e4;
% figure;
% imshowpair(img>thres,cropped>thres,'montage');
% %%
% figure;
% imshowpair(img>1e4,img>1e4,'montage');
% 
% %%
% %%
% if ~isfile(stack_path)
%     save(stack_path,'stack','-v7.3')
% end
% 
% 
% %%
% figure
% volshow(stack2)
% 
% 
% %%
% figure
% volshow(stack)
% %%
% figure
% imshow(stack(:,:,900))
% 
% 
% %%
% resized_stack = imresize3(stack(:,:,160:end),0.3);
% figure
% volshow(resized_stack,'Renderer','VolumeRendering')
% 
% %%
% pos = [1000 1000 500 500 500 size(stack,3)-500];
% tic
% cropped_stack = imcrop3(stack, pos);
% toc
% 
% figure;volshow(cropped_stack)
% %%
% orthosliceViewer(stack)
 
%%
 
% stack_path = '/Users/yeonsu/Dropbox (Harvard University)/Entangled/xray_data/Alpha100_67mm_stack/full_stack.mat';
% 
% load(stack_path);
% stack = stack(:,:,160:end);
% save(stack_path,'stack','-v7.3');
% %%
% stack_path = '/Users/yeonsu/Dropbox (Harvard University)/Entangled/xray_data/Alpha100_68mm_stack/full_stack.mat';
% 
% load(stack_path);
% size(stack)
% 

