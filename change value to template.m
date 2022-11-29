clc;
clear vars;
close all;

% 将nii文件读入内存
vol = spm_vol('Dosenbach_Science_160ROIs_Radius5_Mask.nii');
% 读取nii文件中的矩阵
A = spm_read_vols(vol); %原始的nii文件 61*73*61
B = zeros(61,73,61);   %新的空矩阵 61*73*61
% t统计量 160*1 
C = xlsread('E:\Fudan_Luoqiang_MDDProject\848MDD_794NC\文章作图\Sex difference\Node\Dosen\Non_Global_Dosen160_NC_Sex_Node_Female_SampEnValue.xlsx');
%C = roundn(C, -4);

%数据替换
for i = 1:1:160
    B(A == i) = C(i)
end
% 保存
% matlab里面只需要修改vol的fname这个属性，然后将修改完的data和vol一起写回磁盘就行
% 如果不改名字，它会覆盖原来的nifti文件
vol.fname = 'NG_Dosen_NC_FemaleMean.nii';
vol.dt = [16,0];
spm_write_vol(vol, B);
disp('Done!');