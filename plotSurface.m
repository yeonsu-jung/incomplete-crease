function plotSurface()


%% load and parse
file_path='/Users/yeonsu/GitHub/incomplete-crease/results/HalfCrease04_data_output.mat';
load(file_path)

[folder_name,file_name] = fileparts(file_path);
split_path2 = regexp(file_name,'[\_]','split');
sample_name = horzcat(split_path2{1});

%%
centered = data_output.rotated;

x = centered(1:500:end,1);
y = centered(1:500:end,2);
z = centered(1:500:end,3);

lb = -1000;
ub = 1000;
xq = lb:50:ub;
yq = lb:50:ub;
[X,Y] = meshgrid(xq,yq);
Z = griddata(x, y, z, X, Y);

%%
[X,Y,Z] = GetMesh([x,y,z],xq,yq);
[K,H,Pmax,Pmin] = surfature(X,Y,Z);
%%
close all
% s = surf(X,Y,Z,-H);
% s.EdgeColor = 'none';
patch(X,Y,Z,'edgecolor','none');

axis equal;
axis off;
colorbar;
hold on
caxis([-0.2,0.2])

%%
% tris = delaunay(x, y, z);


tris = convhull(x,y,z);
close all;
axis vis3d
h = patch('Faces',tris,'Vertices',[x, y, z]);
h.EdgeColor = 'none';
view(3)
camlight


end