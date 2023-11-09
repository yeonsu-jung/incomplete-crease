run('../entanglement/scripts/setup.m')
%%
dir_return = dir('results/data_half_*.txt')

close all;
lgd_text = {};
cmap = viridis(4);
k = 1;
set_figure(9,5);
for i = [2,1,3,4]%1:num_datasets
    theta_psi = readmatrix(fullfile(dir_return(i).folder,dir_return(i).name));
    az = theta_psi(:,1);
    el = theta_psi(:,2);

    [~,filename] = fileparts(dir_return(i).name);

    tmp = split(filename,'_');
    tmp = tmp(end);    
    lgd_text{end+1} = tmp{1};

    % crease_angle = str2num(tmp{1})
    
    plot(az,el,'o','color',cmap(k,:));hold on;    
    k = k + 1;
    
end
lgd = legend(lgd_text);
lgd.Location = 'bestoutside';
xlabel('$\theta$');
ylabel('$\psi$')
print(gcf,'results/half_crease.png','-dpng','-r600');

%%
dir_return = dir('results/data_inverse_*.txt')
num_datasets = numel(dir_return);
close all;
lgd_text = {};
cmap = viridis(3);
k = 1;

set_figure(9,5);
for i = 1:num_datasets
    theta_psi = readmatrix(fullfile(dir_return(i).folder,dir_return(i).name));
    az = theta_psi(:,1);
    az = az - min(az);
    el = theta_psi(:,2);

    [~,filename] = fileparts(dir_return(i).name);

    tmp = split(filename,'_');
    tmp = tmp(end);    
    lgd_text{end+1} = tmp{1};

    % crease_angle = str2num(tmp{1})
    
    plot(az,el,'o','color',cmap(k,:));hold on;    
    k = k + 1;
    
end
lgd = legend(lgd_text);
lgd.Location = 'bestoutside';
xlabel('$\theta$');
ylabel('$\psi$')
print(gcf,'results/inverse_half_crease.png','-dpng','-r600');