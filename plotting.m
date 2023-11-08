dir_return = dir('results/Half*.mat');
num_datasets = numel(dir_return);

close all;
lgd_text = {};
for i = [2,1,3,4]%1:num_datasets
    load(fullfile(dir_return(i).folder,dir_return(i).name));
    dir_return(i).name
    xx = data_output.az_bin_all;

    xx(xx < 0) = xx(xx < 0) + 2*pi;
    if i == 2
        xx = -xx;
    end

    yy = mean(data_output.binned_data_all,1);
    [~,I_max] = max(yy);

    plot(xx - xx(I_max),-yy,'o');hold on;

    % data_output.p_left
    dpsidtheta = (abs(data_output.p_right(1)) + abs(data_output.p_right(1)) )/2*2;

    xx_out = xx - xx(I_max);
    yy_out = -yy';
    writematrix([xx_out,yy_out],sprintf('results/data_half_crease_%.4f.txt',dpsidtheta));
    
    lgd_text{end+1} = num2str(dpsidtheta);
    ;
end

legend(lgd_text);
print(gcf,'results/half_crease_results.png','-dpng','-r600');
%%
dir_return = dir('results/Inverse*.mat');
num_datasets = numel(dir_return);

close all;
lgd_text = {};
for i = 1:num_datasets
    load(fullfile(dir_return(i).folder,dir_return(i).name));
    dir_return(i).name
    xx = data_output.az_bin_all;

    xx(xx < 0) = xx(xx < 0) + 2*pi;
    if i == 2
        xx = -xx;
    end

    yy = mean(data_output.binned_data_all,1);
    [~,I_min] = max(yy);

    plot(xx - xx(I_min),yy,'o');hold on;

    % data_output.p_left
    dpsidtheta = (abs(data_output.p_right(1)) + abs(data_output.p_right(1)) )/2*2;

    xx_out = xx - xx(I_min);
    yy_out = yy';
    writematrix([xx_out,yy_out],sprintf('results/data_inverse_half_crease_%.4f.txt',dpsidtheta));
    
    lgd_text{end+1} = num2str(dpsidtheta);
    ;
end

legend(lgd_text);
print(gcf,'results/inverse_half_crease_results.png','-dpng','-r600');
%%
addpath(genpath('../entanglement/functions'))
dir_return = dir('results/Half*.mat');
num_datasets = numel(dir_return);

close all;
lgd_text = {};
for i = [2,1,3,4]%1:num_datasets
    load(fullfile(dir_return(i).folder,dir_return(i).name));

    [az,el,r] = cart2sph(data_output.rotated(:,1),data_output.rotated(:,2),data_output.rotated(:,3));
    I = r < 700;

    tmp = data_output.rotated(I,:);
    tmp(:,3) = -tmp(:,3);
    close all;
    hFig = set_figure(6,6);
    h = plot_pointcloud(tmp);
    
    view([45,45])
    colormap(viridis);

    hFig.Color = [1,1,1];
    h.Color = [1,1,1];
    caxis([0,200])
    grid off
    axis off
    % ylabel(cbar,'Height, $h$(db)');
    print(hFig,sprintf('results/half_crease_image_%d.png',i),'-dpng','-r600');
    ;
end
cbar = colorbar;    

print(hFig,sprintf('results/half_crease_image_colorbar.png'),'-dpng','-r600');
%%
addpath(genpath('../entanglement/functions'))
dir_return = dir('results/Inverse*.mat');
num_datasets = numel(dir_return);

close all;
lgd_text = {};
for i = 1:num_datasets
    load(fullfile(dir_return(i).folder,dir_return(i).name));

    [az,el,r] = cart2sph(data_output.rotated(:,1),data_output.rotated(:,2),data_output.rotated(:,3));
    I = r < 700;

    tmp = data_output.rotated(I,:);
    % tmp(:,3) = -tmp(:,3);
    close all;
    hFig = set_figure(6,6);
    h = plot_pointcloud(tmp);
    
    
    colormap(viridis);    
    view([45,45])
    hFig.Color = [1,1,1];
    h.Color = [1,1,1];
    caxis([-350,100])
    grid off
    axis off
    % ylabel(cbar,'Height, $h$(db)');
    print(hFig,sprintf('results/inverse_half_crease_image_%d.png',i),'-dpng','-r600');
    ;
end

cbar = colorbar;
print(hFig,sprintf('results/inverse_half_crease_image_cbar.png'),'-dpng','-r600');