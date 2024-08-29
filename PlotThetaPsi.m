function PlotThetaPsi(az,el,r, min_r, max_r, num_R, delta_R)

R_list = linspace(min_r,max_r,num_R);

for i = 1:num_R
    I_R = rwnorm(r - R_list(i)) < delta_R;
    tmp = find(I_R);
    
    plot(az(tmp),el(tmp),'.');hold on;
end

end