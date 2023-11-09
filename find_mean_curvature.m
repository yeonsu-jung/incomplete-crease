dir_return = dir('results/Half*.mat');
num_datasets = numel(dir_return);

close all;
lgd_text = {};

r_list = linspace(5,700,20);

for i = [2,1,3,4]%1:num_datasets
    load(fullfile(dir_return(i).folder,dir_return(i).name));
    
    % sampling
    tmp = data_output.rotated;
    tmp(:,3) = -tmp(:,3);

    [az,el,r] = cart2sph(tmp(:,1),tmp(:,2),tmp(:,3));
    tmp = tmp(r < 700,:);
    [az,el,r] = cart2sph(tmp(:,1),tmp(:,2),tmp(:,3));

    I = (r > 100) & (r < 110);

    close all;plot_pointcloud(tmp(I,:));
    
    ;
end
%%
close all
plot(xx_out,yy_out,'o-');

%%
