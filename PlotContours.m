function PlotContours(centered, r, min_r, max_r, num_R, delta_R)

R_list = linspace(min_r,max_r,num_R);

for i = 1:length(R_list)
    I_R = rwnorm(r - R_list(i)) < delta_R;    
    tmp = find(I_R);
    plot3v(centered(tmp(1:50:end),:),'.');hold on;
end

end