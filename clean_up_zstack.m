file_path='C:\Users\yjung\Dropbox (Harvard University)\xray-data\z-stacks\zStack_InverseHalfCrease_04\zstack.mat';

load(file_path)
%%
close all;
volshow(zstack)

%%
small = imresize3(zstack,0.3);
%%
close all;
volshow(small)

%%

summed = sum(zstack,3);
%%
center = [46.5 + 1813/2,93.5 + 1798/2];
[X,Y] = meshgrid(1:1942,1:1908);

I = (X-center(1)).^2 + (Y-center(2)).^2 > 885.^2;

crc = double(I>0);

tmp = summed;
tmp(I>0) = 0;

close all;
imshow(tmp);
%%
mask_3d = repmat(I>0,[1,1,size(zstack,3)]);

%%
zstack2 = zstack;
zstack2(mask_3d) = 0;

%%
zstack2 = bwmorph3(zstack2,'majority');

%%
cc = bwconncomp(zstack2);
%%
[M,I] = max(cellfun(@numel,cc.PixelIdxList));

%%
cc.NumObjects = 1;
cc.PixelIdxList{1} = cc.PixelIdxList{I};
%%
L = labelmatrix(cc);
%%
zstack2 = L > 0;

%%
small2 = imresize3(zstack2,0.3);
%%
close all;volshow(small2);

%%
zstack_clean = zstack2;
save('zstack_clean','zstack_clean','-v7.3');
%%



