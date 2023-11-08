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