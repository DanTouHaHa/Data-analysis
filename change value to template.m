clc;
clear vars;
close all;

% ��nii�ļ������ڴ�
vol = spm_vol('Dosenbach_Science_160ROIs_Radius5_Mask.nii');
% ��ȡnii�ļ��еľ���
A = spm_read_vols(vol); %ԭʼ��nii�ļ� 61*73*61
B = zeros(61,73,61);   %�µĿվ��� 61*73*61
% tͳ���� 160*1 
C = xlsread('E:\Fudan_Luoqiang_MDDProject\848MDD_794NC\������ͼ\Sex difference\Node\Dosen\Non_Global_Dosen160_NC_Sex_Node_Female_SampEnValue.xlsx');
%C = roundn(C, -4);

%�����滻
for i = 1:1:160
    B(A == i) = C(i)
end
% ����
% matlab����ֻ��Ҫ�޸�vol��fname������ԣ�Ȼ���޸����data��volһ��д�ش��̾���
% ����������֣����Ḳ��ԭ����nifti�ļ�
vol.fname = 'NG_Dosen_NC_FemaleMean.nii';
vol.dt = [16,0];
spm_write_vol(vol, B);
disp('Done!');