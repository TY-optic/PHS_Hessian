%% step3b_fft_registration.m
% 基于 FFT 加速的 Hessian 张量场粗配准求解 (严格硬阈值版)
% 目标：
% 1) 遍历离散的候选角度 (theta)
% 2) 在每个角度下，利用 FFT 计算张量匹配目标函数 J 的全平面平移误差
% 3) 提取全局最小误差对应的 (tx, ty, theta)

clear; clc; close all;

%% 1. 加载数据
data_name = 'hessian_validation_from_generalized_fit.mat';
if ~isfile(data_name)
    error('找不到 %s，请先运行 Step 2。', data_name);
end
S = load(data_name);

% 提取网格分辨率
dx = abs(S.sub1_grid.x_vec(2) - S.sub1_grid.x_vec(1));
dy = abs(S.sub1_grid.y_vec(2) - S.sub1_grid.y_vec(1));

%% 2. 准备配准特征场与置信度权重
% 提取各向异性强度 eta
eta1 = S.eta_fit_1; eta1(~S.sub1_mask) = 0;
eta2 = S.eta_fit_2; eta2(~S.sub2_mask) = 0;

% 边界置信度衰减 (防止外推边缘震荡引发的伪匹配)
delta_pixels = 8; 
beta1 = min(1, bwdist(~S.sub1_mask) / delta_pixels);
beta2 = min(1, bwdist(~S.sub2_mask) / delta_pixels);

% 综合掩膜权重 W = Mask * Beta
W1 = double(S.sub1_mask) .* beta1;
W2 = double(S.sub2_mask) .* beta2;

% 设置常数底噪 c0，防止完全各向同性区域分母为0
c0 = mean(eta1(S.sub1_mask)) * mean(eta2(S.sub2_mask)); 
if c0 == 0, c0 = 1e-6; end

% Target (Sub2) Hessian 置零保护 (防止 NaN 污染 FFT)
H2xx = S.Hxx_fit_2; H2xx(~S.sub2_mask) = 0;
H2xy = S.Hxy_fit_2; H2xy(~S.sub2_mask) = 0;
H2yy = S.Hyy_fit_2; H2yy(~S.sub2_mask) = 0;
H2_norm2 = H2xx.^2 + 2*H2xy.^2 + H2yy.^2;

% Source (Sub1) Hessian 置零保护
H1xx_val = S.Hxx_fit_1; H1xx_val(~S.sub1_mask) = 0;
H1xy_val = S.Hxy_fit_1; H1xy_val(~S.sub1_mask) = 0;
H1yy_val = S.Hyy_fit_1; H1yy_val(~S.sub1_mask) = 0;

%% 3. FFT Padding 设置 (防止循环卷积的混叠)
[Ny1, Nx1] = size(W1);
[Ny2, Nx2] = size(W2);
Npad_y = 2^nextpow2(Ny1 + Ny2 - 1);
Npad_x = 2^nextpow2(Nx1 + Nx2 - 1);

%% 4. 预计算 Target (Sub2) 的 FFT
FW2             = fft2(W2, Npad_y, Npad_x);
FW2_eta2        = fft2(W2 .* eta2, Npad_y, Npad_x);
FW2_H2norm      = fft2(W2 .* H2_norm2, Npad_y, Npad_x);
FW2_eta2_H2norm = fft2(W2 .* eta2 .* H2_norm2, Npad_y, Npad_x);
FW2_Hxx         = fft2(W2 .* H2xx, Npad_y, Npad_x);
FW2_Hxy         = fft2(W2 .* H2xy, Npad_y, Npad_x);
FW2_Hyy         = fft2(W2 .* H2yy, Npad_y, Npad_x);
FW2_eta2_Hxx    = fft2(W2 .* eta2 .* H2xx, Npad_y, Npad_x);
FW2_eta2_Hxy    = fft2(W2 .* eta2 .* H2xy, Npad_y, Npad_x);
FW2_eta2_Hyy    = fft2(W2 .* eta2 .* H2yy, Npad_y, Npad_x);

%% 5. 离散角度搜索 & FFT 互相关
angles_deg = linspace(-2, 2, 9); 
min_J_global = inf;
best_idx_x = 0; best_idx_y = 0; best_theta = 0;

% 局部网格坐标 (用于旋转插值)
X1 = S.sub1_grid.X;
Y1 = S.sub1_grid.Y;
xc1 = S.surface_params.center1(1);
yc1 = S.surface_params.center1(2);
X1_c = X1 - xc1;
Y1_c = Y1 - yc1;

disp('开始基于 FFT 的张量匹配搜索...');
tic;
for i = 1:length(angles_deg)
    theta = angles_deg(i);
    
    % 1) 坐标旋转映射 (逆向)
    Xq = X1_c * cosd(theta) + Y1_c * sind(theta);
    Yq = -X1_c * sind(theta) + Y1_c * cosd(theta);
    
    % 2) 标量场与张量场插值
    W1_rot   = interp2(X1_c, Y1_c, W1, Xq, Yq, 'linear', 0);
    eta1_rot = interp2(X1_c, Y1_c, eta1, Xq, Yq, 'linear', 0);
    H1xx_in  = interp2(X1_c, Y1_c, H1xx_val, Xq, Yq, 'linear', 0);
    H1xy_in  = interp2(X1_c, Y1_c, H1xy_val, Xq, Yq, 'linear', 0);
    H1yy_in  = interp2(X1_c, Y1_c, H1yy_val, Xq, Yq, 'linear', 0);
    
    % 3) Hessian 张量旋转数学公式
    H1xx_rot = H1xx_in * cosd(theta)^2 - 2 * H1xy_in * sind(theta)*cosd(theta) + H1yy_in * sind(theta)^2;
    H1xy_rot = H1xx_in * sind(theta)*cosd(theta) + H1xy_in * (cosd(theta)^2 - sind(theta)^2) - H1yy_in * sind(theta)*cosd(theta);
    H1yy_rot = H1xx_in * sind(theta)^2 + 2 * H1xy_in * sind(theta)*cosd(theta) + H1yy_in * cosd(theta)^2;
    H1_norm2 = H1xx_rot.^2 + 2*H1xy_rot.^2 + H1yy_rot.^2;
    
    % 4) Source (Sub1) FFT
    FW1             = fft2(W1_rot, Npad_y, Npad_x);
    FW1_eta1        = fft2(W1_rot .* eta1_rot, Npad_y, Npad_x);
    FW1_H1norm      = fft2(W1_rot .* H1_norm2, Npad_y, Npad_x);
    FW1_eta1_H1norm = fft2(W1_rot .* eta1_rot .* H1_norm2, Npad_y, Npad_x);
    FW1_Hxx         = fft2(W1_rot .* H1xx_rot, Npad_y, Npad_x);
    FW1_Hxy         = fft2(W1_rot .* H1xy_rot, Npad_y, Npad_x);
    FW1_Hyy         = fft2(W1_rot .* H1yy_rot, Npad_y, Npad_x);
    FW1_eta1_Hxx    = fft2(W1_rot .* eta1_rot .* H1xx_rot, Npad_y, Npad_x);
    FW1_eta1_Hxy    = fft2(W1_rot .* eta1_rot .* H1xy_rot, Npad_y, Npad_x);
    FW1_eta1_Hyy    = fft2(W1_rot .* eta1_rot .* H1yy_rot, Npad_y, Npad_x);
    
    % 5) 频域乘法合成目标函数分子与分母
    A_ov_freq = FW2 .* conj(FW1);
    A_ov = fftshift(real(ifft2(A_ov_freq))); % 重叠面积积分
    
    Aq_freq = c0 * A_ov_freq + (FW2_eta2 .* conj(FW1_eta1));
    Aq = fftshift(real(ifft2(Aq_freq))); % 加权面积积分
    
    FT1 = c0 * (FW2_H2norm .* conj(FW1)) + (FW2_eta2_H2norm .* conj(FW1_eta1));
    FT2 = c0 * (FW2 .* conj(FW1_H1norm)) + (FW2_eta2 .* conj(FW1_eta1_H1norm));
    FT3_xx = c0 * (FW2_Hxx .* conj(FW1_Hxx)) + (FW2_eta2_Hxx .* conj(FW1_eta1_Hxx));
    FT3_xy = c0 * (FW2_Hxy .* conj(FW1_Hxy)) + (FW2_eta2_Hxy .* conj(FW1_eta1_Hxy));
    FT3_yy = c0 * (FW2_Hyy .* conj(FW1_Hyy)) + (FW2_eta2_Hyy .* conj(FW1_eta1_Hyy));
    FT3 = -2 * (FT3_xx + 2 * FT3_xy + FT3_yy); 
    
    Num = fftshift(real(ifft2(FT1 + FT2 + FT3))); % 张量误差积分
    
    % 6) 计算最终目标函数 J
    J_match = Num ./ (Aq + 1e-12);
    
    % 面积约束：强制要求重叠权重至少达到子孔径 2 总权重的 30%
    min_area = 0.30 * sum(W2(:)); 
    valid_mask = A_ov > min_area;
    
    % 移除坑人的指数面积引力项，直接使用硬阈值！
    J_total = J_match; 
    J_total(~valid_mask) = inf;
    
    % 7) 寻找当前角度最小误差
    [val, idx] = min(J_total(:));
    
    if val < min_J_global
        min_J_global = val;
        best_theta = theta;
        [best_idx_y, best_idx_x] = ind2sub(size(J_total), idx);
        best_J_map = J_total;
    end
end
toc;
disp('FFT 匹配搜索完毕！');

%% 6. 将 FFT 像素索引转换为物理平移
cy = floor(Npad_y/2) + 1;
cx = floor(Npad_x/2) + 1;

pixel_shift_x = best_idx_x - cx;
pixel_shift_y = best_idx_y - cy;

% 提取网格的物理起点
start_x1 = S.sub1_grid.x_vec(1); start_y1 = S.sub1_grid.y_vec(1);
start_x2 = S.sub2_grid.x_vec(1); start_y2 = S.sub2_grid.y_vec(1);

% T_grid 是全局物理坐标系下的对齐误差（理想值为 0）
T_grid_x = (start_x2 - start_x1) + pixel_shift_x * dx;
T_grid_y = (start_y2 - start_y1) + pixel_shift_y * dy;

% 真值：将 Sub1 的局部原点平移多少才能与 Sub2 重合？
true_shift_x = S.surface_params.center1(1) - S.surface_params.center2(1); % -25 - 25 = -50 mm
true_shift_y = S.surface_params.center1(2) - S.surface_params.center2(2); % 0 mm

% calc_t 是通过 FFT 计算出的局部相对位姿平移量
calc_tx = true_shift_x + T_grid_x;
calc_ty = true_shift_y + T_grid_y;

disp('================ 粗配准结果分析 ================');
fprintf('理论真值局部相对位姿 : tx = %8.4f mm, ty = %8.4f mm, theta = %5.2f°\n', true_shift_x, true_shift_y, 0);
fprintf('FFT 解算的局部相对位姿: tx = %8.4f mm, ty = %8.4f mm, theta = %5.2f°\n', calc_tx, calc_ty, best_theta);
fprintf('平移绝对定位误差     : %.4f mm (即 FFT 全局匹配残差)\n', sqrt(T_grid_x^2 + T_grid_y^2));

%% 7. 绘制误差曲面
figure('Position', [300, 200, 600, 500], 'Name', 'FFT Cost Surface');
crop_rad = 60;
J_crop = best_J_map(cy-crop_rad : cy+crop_rad, cx-crop_rad : cx+crop_rad);

% 将坐标轴映射为局部相对平移
x_axis = true_shift_x + (start_x2 - start_x1) + ((-crop_rad:crop_rad) * dx);
y_axis = true_shift_y + (start_y2 - start_y1) + ((-crop_rad:crop_rad) * dy);

imagesc(x_axis, y_axis, log10(J_crop)); 
set(gca, 'YDir', 'normal'); hold on;
plot(calc_tx, calc_ty, 'rp', 'MarkerSize', 15, 'MarkerFaceColor', 'r');
title(sprintf('Cost Surface (Log10) at \\theta = %.2f^\\circ', best_theta));
xlabel('Local Relative t_x (mm)'); ylabel('Local Relative t_y (mm)');
axis image; colormap parula; colorbar;


%% 8. 保存粗配准结果供 Step 4 使用
% 注意：此处保存的 calc_tx, calc_ty 是物理相对平移量
best_tx = calc_tx;
best_ty = calc_ty;

save_name_fft = 'fft_registration_result.mat';
save(save_name_fft, 'best_tx', 'best_ty', 'best_theta', 'min_J_global');
disp(['已保存 FFT 粗配准结果至：', save_name_fft]);