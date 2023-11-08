function sample_azimuthal_variation(centered_points,phi_list,delta_phi)

[az,el,r]= cart2sph(centered_points(:,1),centered_points(:,2),centered_points(:,3));

num_R = 20;
R_list = linspace(350,850,num_R);
% az_list = linspace();
az_range = pi/4;


delta_R = 1;

for i = 1:num_R
    R = R_list(i);
    
    I_R = rwnorm(r- R) < delta_R;
    I_az = rwnorm(az2) < az_range/2;

end

end