%% step4_3D_reconstruction.m
% 步骤 4：二维粗配准结果的三维回代与 ICP 局部精配准 (全自动流水线)
% 目标：
% 1) 加载 Step 1 生成的独立点云数据
% 2) 加载 Step 3 算出的高精度二维 FFT 粗配准参数
% 3) 估算重叠区域高度差 \Delta h，完成三维初始位姿回代
% 4) 执行点到面 ICP 精配准 (包含局部重叠率优化)
% 5) 还原至全局坐标系对比绝对真值，输出最终面形拼接 RMSE，并极速绘图

clear; clc; close all;

%% 1. 加载数据 (Step 1 点云 & Step 3 FFT 结果)
data_step1 = 'phs_freeform_fit_data_generalized.mat';
if ~isfile(data_step1)
    error('找不到 %s，请先运行 Step 1 生成数据。', data_step1);
end
S = load(data_step1);

data_step3 = 'fft_registration_result.mat';
if ~isfile(data_step3)
    error('找不到 %s，请先运行 Step 3。', data_step3);
end
FFT_Res = load(data_step3);

pc1_global = S.point_cloud_sub1;
pc2_global = S.point_cloud_sub2;

%% 2. 模拟盲拼接：剥离全局坐标，转换到独立的传感器局部坐标系
% 真实仪器测量时，点云往往以传感器光轴为中心
c1 = [S.surface_params.center1, mean(pc1_global(:,3))];
c2 = [S.surface_params.center2, mean(pc2_global(:,3))];

pc1_local = pc1_global - c1;
pc2_local = pc2_global - c2;

%% 3. 注入 Step 3 的二维 FFT 粗配准结果进行三维回代
best_tx = FFT_Res.best_tx;
best_ty = FFT_Res.best_ty;
best_theta = FFT_Res.best_theta;

disp('================ 1. 三维粗配准初值构造 ================');
fprintf('自动加载 FFT 粗配准参数: tx = %.4f mm, ty = %.4f mm, theta = %.2f°\n', best_tx, best_ty, best_theta);

% 3.1 应用 XY 平面的旋转和平移
theta_rad = deg2rad(best_theta);
R_2D = [cos(theta_rad), -sin(theta_rad);
        sin(theta_rad),  cos(theta_rad)];
    
pc1_coarse = zeros(size(pc1_local));
pc1_coarse(:, 1:2) = (R_2D * pc1_local(:, 1:2)')' + [best_tx, best_ty];
pc1_coarse(:, 3)   = pc1_local(:, 3); 

% 3.2 估算重叠区域的平均高度差 \Delta h
mdl2 = KDTreeSearcher(pc2_local(:, 1:2));
[idx, D] = knnsearch(mdl2, pc1_coarse(:, 1:2));

dx = abs(S.x_vec(2) - S.x_vec(1));
overlap_mask = D < (2 * dx);

if sum(overlap_mask) > 10
    dz_overlap = pc2_local(idx(overlap_mask), 3) - pc1_coarse(overlap_mask, 3);
    delta_h = mean(dz_overlap);
else
    delta_h = 0;
end

fprintf('估算重叠区高度差 \\Delta h : %.6f mm\n', delta_h);

% 补偿高度差，完成完整的三维初值回代
pc1_coarse(:, 3) = pc1_coarse(:, 3) + delta_h;

%% 4. 点到面 ICP 高精度精配准
disp('================ 2. ICP 三维精配准 ================');

ptCloudTarget = pointCloud(pc2_local);
ptCloudSource = pointCloud(pc1_coarse);

% 必须设置 InlierRatio 保护机制！允许只有部分点(如 25%)参与匹配，防止算法向非重叠区过度拉扯
overlap_ratio = 0.25; 

tic;
[tform, ptCloudAligned, rmse_icp] = pcregistericp(ptCloudSource, ptCloudTarget, ...
    'Metric', 'pointToPlane', ...
    'Extrapolate', true, ...
    'InlierRatio', overlap_ratio, ... 
    'Tolerance', [1e-6, 1e-6], ...
    'MaxIterations', 50);
toc;

pc1_fine = ptCloudAligned.Location;

fprintf('ICP 内部局部收敛 RMS 误差: %.6e mm\n', rmse_icp);

% 提取 ICP 进一步消除的残余错位
R_final = tform.R;
t_final = tform.Translation;
disp('ICP 补充精调姿态 (相对于粗配准):');
disp('  微量旋转矩阵 R:'); disp(R_final);
disp('  微量平移向量 t:'); disp(t_final);

%% 5. 还原至全局坐标系，并评估终极拼接面形误差
disp('================ 3. 全局面形拼接误差终极评估 ================');
% 将组装好的点云平移回 Sub2 最初的全局设计位置进行基准验算
pc2_final_global = pc2_local + c2;
pc1_final_global = pc1_fine + c2;

% 在全局真值曲面 (Z_true) 上进行插值，计算绝对残差
Z_true_eval_1 = interp2(S.X, S.Y, S.Z_true, pc1_final_global(:,1), pc1_final_global(:,2), 'spline');
Z_true_eval_2 = interp2(S.X, S.Y, S.Z_true, pc2_final_global(:,1), pc2_final_global(:,2), 'spline');

err_1 = pc1_final_global(:,3) - Z_true_eval_1;
err_2 = pc2_final_global(:,3) - Z_true_eval_2;

rms_err_1 = sqrt(mean(err_1.^2, 'omitnan'));
rms_err_2 = sqrt(mean(err_2.^2, 'omitnan'));

% 总体拼接 RMSE
err_total = [err_1; err_2];
rms_err_total = sqrt(mean(err_total.^2, 'omitnan'));

fprintf('子孔径 1 (动态配准) 面形 RMSE : %.6e mm\n', rms_err_1);
fprintf('子孔径 2 (固定基准) 面形 RMSE : %.6e mm\n', rms_err_2);
fprintf('==================================================\n');
fprintf('全口径拼接整体终极 RMSE    : %.6e mm\n', rms_err_total);
fprintf('==================================================\n');

%% 6. 极速可视化 (白底、学术期刊标准)
fig = figure('Position', [50, 100, 1600, 400], 'Name', 'Final 3D Stitching Pipeline', 'Color', 'w');  

% 1) 盲状态：局部坐标系下的初始点云 (原点堆叠)
subplot(1, 4, 1);
plot3(pc1_local(:,1), pc1_local(:,2), pc1_local(:,3), '.r', 'MarkerSize', 4); hold on;
plot3(pc2_local(:,1), pc2_local(:,2), pc2_local(:,3), '.b', 'MarkerSize', 4);
title('1. Sensor Local Frames (Blind State)');
xlabel('X'); ylabel('Y'); zlabel('Z');
legend('Sub1 (Source)', 'Sub2 (Target)', 'Location', 'southoutside');
view(-30, 30); grid on; axis equal;

% 2) 粗配准：注入 FFT 结果
subplot(1, 4, 2);
plot3(pc1_coarse(:,1), pc1_coarse(:,2), pc1_coarse(:,3), '.r', 'MarkerSize', 4); hold on;
plot3(pc2_local(:,1), pc2_local(:,2), pc2_local(:,3), '.b', 'MarkerSize', 4);
title('2. After FFT Coarse Registration');
xlabel('X'); ylabel('Y'); zlabel('Z');
view(-30, 30); grid on; axis equal;

% 3) 精配准：ICP 结果
subplot(1, 4, 3);
plot3(pc1_fine(:,1), pc1_fine(:,2), pc1_fine(:,3), '.r', 'MarkerSize', 4); hold on;
plot3(pc2_local(:,1), pc2_local(:,2), pc2_local(:,3), '.b', 'MarkerSize', 4);
title('3. After ICP Fine Registration');
xlabel('X'); ylabel('Y'); zlabel('Z');
view(-30, 30); grid on; axis equal;

% 4) 最终全局重建面形残差图
subplot(1, 4, 4);
scatter3(pc1_final_global(:,1), pc1_final_global(:,2), err_1, 15, err_1, 'filled'); hold on;
scatter3(pc2_final_global(:,1), pc2_final_global(:,2), err_2, 15, err_2, 'filled');
title(sprintf('Final Global Stitching Error\nOverall RMS: %.2e mm', rms_err_total));
xlabel('Global X (mm)'); ylabel('Global Y (mm)'); zlabel('Error (mm)');
colormap(gca, jet); 
colorbar;
view(2); grid on; axis equal; 
axis([min(S.X(:)) max(S.X(:)) min(S.Y(:)) max(S.Y(:))]);