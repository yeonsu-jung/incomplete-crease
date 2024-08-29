function [X,Y,Z] = GetMesh(pts,xq,yq)

x = pts(:,1);
y = pts(:,2);
z = pts(:,3);

[X,Y] = meshgrid(xq,yq);
Z = griddata(x,y,z,X,Y);


end