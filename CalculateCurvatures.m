function CalculateCurvatures(pts)

% save('centered_','centered');
%%
x = centered(1:500:end,1);
y = centered(1:500:end,2);
z = centered(1:500:end,3);

x_unique = unique(x);
y_unique = unique(y);

% Create a meshgrid from the unique x and y values
[X, Y] = meshgrid(x_unique, y_unique);

% Interpolate z values onto the meshgrid
Z = griddata(x, y, z, X, Y);
%%
tic
[K,H,Pmax,Pmin] = surfature(X,Y,Z);
toc

K(isnan(K)) = 0;
H(isnan(H)) = 0;

%%
max_H = max(H(:));
min_H = min(H(:));
min_H = max_H*0.999;

K_normalized = abs((H - min_H)/(max_H - min_H));
%%
close all;
figure;
surf(X, Y, Z, K_normalized);
shading interp;

% Applying a nice colormap
colormap(parula);  % Example colormap, you can use 'jet', 'hot', etc.
colorbar;

xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');