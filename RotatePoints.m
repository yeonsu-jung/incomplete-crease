function rt = RotatePoints(centered, rotation_angle)

rotz = @(t) [cos(t) -sin(t) 0 ; sin(t) cos(t) 0 ; 0 0 1] ;
R = rotz(rotation_angle);
rt = zeros(size(centered));
for i = 1:size(centered,1)
    pt = centered(i,:);
    rt(i,:) = (R*pt')';
end


end