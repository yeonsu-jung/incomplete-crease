load('zstack_clean.mat')
%%
ind = find(zstack_clean);
[X,Y,Z] = ind2sub(size(zstack_clean),ind);

%%
ptCloud = pointCloud([X,Y,Z]);
%%
close all;
pcshow(ptCloud)
%%
addpath(genpath('../entanglement/functions'))
%%
clc
colormap(viridis)

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
X = all_points(:,1);
Y = all_points(:,2);
Z = all_points(:,3);

ptCloud = pointCloud([X,Y,Z]);
%%
close all;
pcshow(ptCloud)
%%
colormap(magma)
%%
% sampling
meshgrid()








%%
size(X)
%%
sparse_points = all_points(1:1:end,:);
%%
tail = [-857,68,-193];
% center = [17, 53, -1];
center = [mean(all_points(:,1:2)),max(all_points(:,3))];
%%
centered = all_points - center;

close all;
plot3v(centered);
%%
I = rwnorm(centered) < 150;
close all;
plot3v(centered(I,:),'.');axis equal;
%%
close all;
plot_pointcloud(centered(I,:));

%%
plane_vector = [0,1,0];
centered.*plane_vector

%%
stat = get_principal_axis_length(all_points);
%%
stat.EigenVectors

%%
close all;
plot3v(sparse_points,'.');
hold on;
plot3v([tail;center],'k','linewidth',2);

quiver3v(center,-stat.EigenVectors(:,1)'*1000,'k-')
quiver3v(center,stat.EigenVectors(:,2)'*1000)
quiver3v(center,stat.EigenVectors(:,3)'*1000)
%%
crease_length = norm(tail-center);
%%
sparse_points = all_points(1:100:end,:);
stat = get_principal_axis_length(sparse_points);
R = stat.EigenVectors';

transformed = zeros(size(sparse_points));
for i = 1:size(sparse_points,1)
    tmp = R*sparse_points(i,:)';
    transformed(i,:) = tmp(1:3)';
end
%%
close all;
% plot3v(sparse_points,'.');hold on;
plot3v(transformed,'.');
axis equal;

%%
% transformation matrix 

T = rigidTransformMatrix(center,tail,[0,0,0],[1,0,0]*crease_length);

T2 = rigidTransformFrom2Points([center;tail],[[0,0,0];[1,0,0]*crease_length]);
transformed = zeros(size(sparse_points));

twd = [0,1,1];
twd = twd/norm(twd);
R = rotMat([0,1,0]',twd',-pi/20);

for i = 1:size(sparse_points,1)
    tmp = (T2*[sparse_points(i,:),1]');
    transformed(i,:) = tmp(1:3)';
end

close all;
% plot(transformed(:,2),transformed(:,3),'.');

for i = 1:size(sparse_points,1)
    
    tmp = R*transformed(i,:)';

    transformed(i,:) = tmp';
end
hold on;
plot(transformed(:,2),transformed(:,3),'.');
grid on
%%
close all;
plot3v(transformed);
axis equal;
%
view([0,1])
%%
close all;
plot(transformed(:,2),transformed(:,3),'.');
%%



%%
X = transformed(:,1);
Y = transformed(:,2);
Z = transformed(:,3);

ptCloud = pointCloud([X,Y,Z]);

close all;
pcshow(ptCloud)
%%
[az,elev,~] = cart2sph(transformed(:,1),transformed(:,2),transformed(:,3));
%%
I = rwnorm(az - 0) < pi/100;

close all;
plot3v(transformed(I,:),'o');
axis equal;

%%
close all;
plot(transformed(I,2),transformed(I,3),'o');
%%
% crease angle

I_positive = transformed(:,2) >= 0;
I_negative = transformed(:,2) < 0;
%%
stats_positive = get_principal_axis_length(transformed(I_positive,:));
stats_negative = get_principal_axis_length(transformed(I_negative,:));
%%
symmetric_score(transformed)
%%
clc
sparse_points = all_points(1:1000:end,:);
clc
x0 = [center,tail,-pi/20];
options = optimset('PlotFcns',@optimplotfval);
x_opt = fminsearch(@(x) objective_function(sparse_points,x),x0,options);
%%
transformed = get_transformed_pointcloud(sparse_points,x_opt);
%%


%%
center = x_opt(1:3);
tail = x_opt(4:6);
close all;
plot3v(transformed,'.');
hold on;
plot3v([center;tail],'linewidth',2);
axis equal;
% view([0,1])


%%
I = rwnorm(az - theta0) < sampling_uncertainty;

close all;
plot3v(transformed(I,:),'o');
hold on;
plot3v(transformed(~I,:),'.');
axis equal;

mean(elev(I))
std(elev(I))
%%
num_sampling = 100; 
theta_list = linspace(-pi,pi,num_sampling);

psi_list = zeros(num_sampling,1);
psi_err_list = zeros(num_sampling,1);
for i_theta = 1:num_sampling
    th = theta_list(i_theta);
    I = rwnorm(az - th) < sampling_uncertainty;

    psi_list(i_theta) = mean(elev(I));
    psi_err_list(i_theta) = std(elev(I));
end
%%

%%
close all;
errorbar(theta_list,psi_list,psi_err_list,'o');
%%
close all;
errorbar(theta_list,-fftshift(psi_list),fftshift(psi_err_list),'o');











%%
% psi(theta)
% sampling uncertainty
sampling_uncertainty = pi/100;
theta0 = pi/8;

distance_list = zeros(size(transformed,1),1);
for i = 1:size(transformed,1);
    x1 = transformed(i,1);
    y1 = transformed(i,2);

    distance_list(i) = distancePointToLine(x1, y1, 0,0,cos(theta0),sin(theta0));
end

%%
% I = rwnorm(transformed(:,1:2) - [cos(theta0),sin(theta0)]) < sampling_uncertainty;
I = distance_list < sampling_uncertainty;

close all;
plot3v(transformed(I,:),'o');


%%
function score = objective_function(sparse_points,x)
    transformed = get_transformed_pointcloud(sparse_points,x);
    score = symmetric_score(transformed);
end

function transformed = get_transformed_pointcloud(sparse_points,x)

center = x(1:3);
tail = x(4:6);
rotation_angle = x(7);

crease_length = norm(tail-center);
T2 = rigidTransformFrom2Points([center;tail],[[0,0,0];[1,0,0]*crease_length]);
transformed = zeros(size(sparse_points));

twd = [0,1,1];
twd = twd/norm(twd);
R = rotMat([0,1,0]',twd',rotation_angle);

for i = 1:size(sparse_points,1)
    tmp = (T2*[sparse_points(i,:),1]');
    transformed(i,:) = tmp(1:3)';
end

for i = 1:size(sparse_points,1)    
    tmp = R*transformed(i,:)';
    transformed(i,:) = tmp';
end

end


function score3  = symmetric_score(transformed)
    I_positive = transformed(:,2) >= 0;
    I_negative = transformed(:,2) < 0;

    % transformed(I_positive,:) 
    
    stats_positive = get_principal_axis_length(transformed(I_positive,:));
    stats_negative = get_principal_axis_length(transformed(I_negative,:));
        
    score1 = norm(stats_positive.PrincipalAxisLength - stats_negative.PrincipalAxisLength);
    score2 = norm(stats_positive.Centroid([1,3]) - stats_negative.PrincipalAxisLength([1,3]));

    score3 = sqrt(score1^2 + score2^2);

end

function d = distancePointToLine(x1, y1, x2, y2, x3, y3)
    % Compute the numerator of the formula
    numerator = abs((x3 - x2) * (y2 - y1) - (x2 - x1) * (y3 - y2));
    
    % Compute the denominator of the formula
    denominator = sqrt((x3 - x2)^2 + (y3 - y2)^2);
    
    % Compute the distance
    d = numerator / denominator;
end



function T = rigidTransformMatrix(A, B, Ap, Bp)
    % Ensure points are column vectors
    A = A(:); B = B(:);
    Ap = Ap(:); Bp = Bp(:);
    
    % Translate so that point A is at the origin
    T1 = eye(4);
    T1(1:3, 4) = -A;
    
    % Compute the rotation matrix
    v = B - A;
    vp = Bp - Ap;
    
    v = v / norm(v);  % Normalize to unit length
    vp = vp / norm(vp); % Normalize to unit length
    
    axis = cross(v, vp); % Rotation axis
    angle = acos(dot(v, vp)); % Rotation angle
    
    % Compute the rotation matrix using Rodrigues' formula
    K = [0 -axis(3) axis(2); axis(3) 0 -axis(1); -axis(2) axis(1) 0];
    R = eye(3) + sin(angle) * K + (1 - cos(angle)) * K^2;
    
    T2 = eye(4);
    T2(1:3, 1:3) = R;
    
    % Translate so that the origin goes to point Ap
    T3 = eye(4);
    T3(1:3, 4) = Ap;
    
    % Compute the final transformation matrix
    T = T3 * T2 * T1;
end

function T = rigidTransformFrom2Points(sourcePoints, targetPoints)
    % Ensure points are in homogeneous coordinates
    p1 = [sourcePoints(1, :), 1]';
    p2 = [sourcePoints(2, :), 1]';
    q1 = [targetPoints(1, :), 1]';
    q2 = [targetPoints(2, :), 1]';
    
    % Compute direction vectors
    v1 = p2 - p1;
    v2 = q2 - q1;
    
    % Normalize direction vectors
    v1 = v1/norm(v1(1:3));
    v2 = v2/norm(v2(1:3));
    
    % Compute rotation matrix
    axis = cross(v1(1:3), v2(1:3));
    angle = acos(dot(v1(1:3), v2(1:3)));
    k = axis/norm(axis);
    K = [0 -k(3) k(2); k(3) 0 -k(1); -k(2) k(1) 0];
    R = eye(3) + sin(angle) * K + (1 - cos(angle)) * K^2;
    
    % Compute translation vector
    t = q1(1:3) - R*p1(1:3);
    
    % Construct transformation matrix
    T = [R, t; 0 0 0 1];
end



function rot = rotMat(b,a,alpha)
% ROTMAT returns a rotation matrix that rotates unit vector b to a
%
%   rot = rotMat(b) returns a d x d rotation matrix that rotate
%   unit vector b to the north pole (0,0,...,0,1)
%
%   rot = rotMat(b,a ) returns a d x d rotation matrix that rotate
%   unit vector b to a
%
%   rot = rotMat(b,a,alpha) returns a d x d rotation matrix that rotate
%   unit vector b towards a by alpha (in radian)
%
%    See also .

% Last updated Nov 7, 2009
% Sungkyu Jung


[s1 s2]=size(b);
d = max(s1,s2);
b= b/norm(b);
if min(s1,s2) ~= 1 || nargin==0 , help rotMat, return, end  

if s1<=s2;    b = b'; end

if nargin == 1;
    a = [zeros(d-1,1); 1];
    alpha = acos(a'*b);
end

if nargin == 2;
    alpha = acos(a'*b);
end
if abs(a'*b - 1) < 1e-15; rot = eye(d); return, end
if abs(a'*b + 1) < 1e-15; rot = -eye(d); return, end

c = b - a * (a'*b); c = c / norm(c);
A = a*c' - c*a' ;

rot = eye(d) + sin(alpha)*A + (cos(alpha) - 1)*(a*a' +c*c');
end


function plot_pointcloud(all_points)

X = all_points(:,1);
Y = all_points(:,2);
Z = all_points(:,3);

ptCloud = pointCloud([X,Y,Z]);
pcshow(ptCloud)


end
