%% run_phs_freeform_demo_generalized.m
% 自由曲面子孔径提取 + 真实支撑域 mask + 三次多项式去趋势 + PHS 残差拟合
% 当前版本：
% 1) 真值面型 = 三次多项式主形貌 + 局部高斯凸起/凹陷 + 小幅中频波纹
% 2) 用于验证：Poly-only 无法完全表达，Poly + PHS 应体现增益
% 3) 输出 MAT 文件，供下一步 Hessian 场计算使用

clear; clc; close all;
rng(1);

%% 1. 原始面型参数设置
surface_params = struct();

% 矩形全口径尺寸（mm）
surface_params.Lx = 100;
surface_params.Ly = 60;

% 采样分辨率
surface_params.num_points_X = 250;
surface_params.num_points_Y = 150;

% 圆形子孔径参数
surface_params.center1 = [-25, 0];
surface_params.center2 = [ 25, 0];
surface_params.R = 40;

% 三次多项式主形貌
surface_params.c1 = 5e-5;
surface_params.c2 = 2e-5;
surface_params.c3 = -2e-5;
surface_params.c4 = 8e-5;
surface_params.c5 = 5e-3;
surface_params.c6 = 8e-3;

% 局部高斯自由曲面残差（单位 mm）
surface_params.bump1_amp = 0.030;
surface_params.bump1_x0  = -18;
surface_params.bump1_y0  =  10;
surface_params.bump1_sx  =  10;
surface_params.bump1_sy  =   7;

surface_params.bump2_amp = -0.022;
surface_params.bump2_x0  =  22;
surface_params.bump2_y0  = -12;
surface_params.bump2_sx  =   9;
surface_params.bump2_sy  =   8;

surface_params.bump3_amp = 0.015;
surface_params.bump3_x0  =   5;
surface_params.bump3_y0  =  18;
surface_params.bump3_sx  =   6;
surface_params.bump3_sy  =   6;

% 中频波纹项（单位 mm）
surface_params.ripple_amp = 0.004;
surface_params.ripple_Lx  = 28;
surface_params.ripple_Ly  = 22;
surface_params.ripple_phi = pi/7;

% 余弦调制项（单位 mm）
surface_params.cos_amp = 0.003;
surface_params.cos_Lx  = 40;
surface_params.cos_Ly  = 32;

% 噪声参数（单位 mm）
surface_params.add_noise = true;
surface_params.noise_std = 5e-6;

%% 2. 拟合参数
fit_opts = struct();
fit_opts.normalize_poly = true;
fit_opts.normalize_phs = true;
%fit_opts.kernel = 'r6logr';
fit_opts.kernel = 'r4logr';
fit_opts.lambda = 1e-10;
fit_opts.max_centers = 1500;
fit_opts.skip_phs_if_small = true;
fit_opts.residual_tol = 1e-12;

%% 3. 生成全口径规则网格与面型
x_vec = linspace(-surface_params.Lx/2, surface_params.Lx/2, surface_params.num_points_X);
y_vec = linspace(-surface_params.Ly/2, surface_params.Ly/2, surface_params.num_points_Y);
[X, Y] = meshgrid(x_vec, y_vec);

[Z_true, Z_cubic_true, Z_residual_true] = surface_freeform_general(X, Y, surface_params);
Z_meas = Z_true;

if surface_params.add_noise
    Z_meas = Z_meas + surface_params.noise_std .* randn(size(Z_meas));
end

point_cloud_full = [X(:), Y(:), Z_meas(:)];

%% 4. 提取两个圆形子孔径点云
[point_cloud_sub1, idx_sub1] = extract_subaperture(point_cloud_full, surface_params.center1, surface_params.R);
[point_cloud_sub2, idx_sub2] = extract_subaperture(point_cloud_full, surface_params.center2, surface_params.R);

mask_full_sub1 = reshape(idx_sub1, size(X));
mask_full_sub2 = reshape(idx_sub2, size(X));

[sub1_grid, sub1_mask] = build_support_grid_from_full_grid(X, Y, Z_true, Z_meas, Z_cubic_true, Z_residual_true, mask_full_sub1);
[sub2_grid, sub2_mask] = build_support_grid_from_full_grid(X, Y, Z_true, Z_meas, Z_cubic_true, Z_residual_true, mask_full_sub2);

%% 5. 子孔径 1：三次多项式 + PHS 残差拟合
disp('开始对子孔径 1 进行拟合...');
model_sub1 = fit_poly_plus_phs(point_cloud_sub1(:,1), point_cloud_sub1(:,2), point_cloud_sub1(:,3), fit_opts);

%% 6. 子孔径 2：三次多项式 + PHS 残差拟合
disp('开始对子孔径 2 进行拟合...');
model_sub2 = fit_poly_plus_phs(point_cloud_sub2(:,1), point_cloud_sub2(:,2), point_cloud_sub2(:,3), fit_opts);

%% 7. 在原始采样点上评价拟合结果
z_fit_sub1_pts  = eval_poly_plus_phs(model_sub1, point_cloud_sub1(:,1), point_cloud_sub1(:,2));
z_fit_sub2_pts  = eval_poly_plus_phs(model_sub2, point_cloud_sub2(:,1), point_cloud_sub2(:,2));
z_poly_sub1_pts = eval_poly_model(model_sub1.polyModel, point_cloud_sub1(:,1), point_cloud_sub1(:,2));
z_poly_sub2_pts = eval_poly_model(model_sub2.polyModel, point_cloud_sub2(:,1), point_cloud_sub2(:,2));

[z_true_sub1_pts, z_cubic_sub1_pts, z_res_true_sub1_pts] = surface_freeform_general(point_cloud_sub1(:,1), point_cloud_sub1(:,2), surface_params);
[z_true_sub2_pts, z_cubic_sub2_pts, z_res_true_sub2_pts] = surface_freeform_general(point_cloud_sub2(:,1), point_cloud_sub2(:,2), surface_params);

res_sub1_meas = z_fit_sub1_pts - point_cloud_sub1(:,3);
res_sub2_meas = z_fit_sub2_pts - point_cloud_sub2(:,3);

res_sub1_true = z_fit_sub1_pts - z_true_sub1_pts;
res_sub2_true = z_fit_sub2_pts - z_true_sub2_pts;

res_sub1_poly_true = z_poly_sub1_pts - z_true_sub1_pts;
res_sub2_poly_true = z_poly_sub2_pts - z_true_sub2_pts;

metrics_sub1_meas = calc_metrics(res_sub1_meas);
metrics_sub2_meas = calc_metrics(res_sub2_meas);
metrics_sub1_true = calc_metrics(res_sub1_true);
metrics_sub2_true = calc_metrics(res_sub2_true);
metrics_sub1_poly_true = calc_metrics(res_sub1_poly_true);
metrics_sub2_poly_true = calc_metrics(res_sub2_poly_true);

disp('--- 点云采样点上的误差 ---');
disp(['Sub1 相对测量值 RMSE = ', num2str(metrics_sub1_meas.rmse, '%.6e')]);
disp(['Sub2 相对测量值 RMSE = ', num2str(metrics_sub2_meas.rmse, '%.6e')]);
disp(['Sub1 相对真值   RMSE = ', num2str(metrics_sub1_true.rmse, '%.6e')]);
disp(['Sub2 相对真值   RMSE = ', num2str(metrics_sub2_true.rmse, '%.6e')]);
disp(['Sub1 Poly-only 真值 RMSE = ', num2str(metrics_sub1_poly_true.rmse, '%.6e')]);
disp(['Sub2 Poly-only 真值 RMSE = ', num2str(metrics_sub2_poly_true.rmse, '%.6e')]);

%% 8. 在真实采样支撑域局部网格上评价拟合结果
Z1_fit_grid  = eval_poly_plus_phs(model_sub1, sub1_grid.X, sub1_grid.Y);
Z2_fit_grid  = eval_poly_plus_phs(model_sub2, sub2_grid.X, sub2_grid.Y);

Z1_poly_grid = eval_poly_model(model_sub1.polyModel, sub1_grid.X, sub1_grid.Y);
Z2_poly_grid = eval_poly_model(model_sub2.polyModel, sub2_grid.X, sub2_grid.Y);

Z1_true_grid = sub1_grid.Z_true;
Z2_true_grid = sub2_grid.Z_true;

Z1_fit_grid(~sub1_mask)  = nan;
Z2_fit_grid(~sub2_mask)  = nan;
Z1_poly_grid(~sub1_mask) = nan;
Z2_poly_grid(~sub2_mask) = nan;
Z1_true_grid(~sub1_mask) = nan;
Z2_true_grid(~sub2_mask) = nan;

Res1_grid = Z1_fit_grid - Z1_true_grid;
Res2_grid = Z2_fit_grid - Z2_true_grid;

Res1_poly_grid = Z1_poly_grid - Z1_true_grid;
Res2_poly_grid = Z2_poly_grid - Z2_true_grid;

metrics_sub1_grid = calc_metrics(Res1_grid(sub1_mask));
metrics_sub2_grid = calc_metrics(Res2_grid(sub2_mask));
metrics_sub1_poly = calc_metrics(Res1_poly_grid(sub1_mask));
metrics_sub2_poly = calc_metrics(Res2_poly_grid(sub2_mask));

disp('--- 局部规则网格（真实采样支撑域内）的误差 ---');
disp(['Sub1 Final Grid RMSE = ', num2str(metrics_sub1_grid.rmse, '%.6e')]);
disp(['Sub2 Final Grid RMSE = ', num2str(metrics_sub2_grid.rmse, '%.6e')]);
disp(['Sub1 Poly-only Grid RMSE = ', num2str(metrics_sub1_poly.rmse, '%.6e')]);
disp(['Sub2 Poly-only Grid RMSE = ', num2str(metrics_sub2_poly.rmse, '%.6e')]);

%% 9. 残差面真实值与拟合值
% PHS 真正的拟合目标是：总真值面形 减去 多项式拟合出的面形
Z1_res_true_grid = Z1_true_grid - Z1_poly_grid;
Z2_res_true_grid = Z2_true_grid - Z2_poly_grid;

%Z1_res_true_grid = sub1_grid.Z_residual_true;
%Z2_res_true_grid = sub2_grid.Z_residual_true;

Z1_res_fit_grid = eval_phs_residual_model(model_sub1.phsModel, sub1_grid.X, sub1_grid.Y);
Z2_res_fit_grid = eval_phs_residual_model(model_sub2.phsModel, sub2_grid.X, sub2_grid.Y);

Z1_res_true_grid(~sub1_mask) = nan;
Z2_res_true_grid(~sub2_mask) = nan;
Z1_res_fit_grid(~sub1_mask)  = nan;
Z2_res_fit_grid(~sub2_mask)  = nan;

Res1_residual_grid = Z1_res_fit_grid - Z1_res_true_grid;
Res2_residual_grid = Z2_res_fit_grid - Z2_res_true_grid;

metrics_sub1_residual = calc_metrics(Res1_residual_grid(sub1_mask));
metrics_sub2_residual = calc_metrics(Res2_residual_grid(sub2_mask));

disp('--- PHS 对真实残差面的拟合误差 ---');
disp(['Sub1 Residual-surface RMSE = ', num2str(metrics_sub1_residual.rmse, '%.6e')]);
disp(['Sub2 Residual-surface RMSE = ', num2str(metrics_sub2_residual.rmse, '%.6e')]);

%% 10. 可视化
zmin = min(Z_meas(:));
zmax = max(Z_meas(:));

figure('Position', [40, 40, 1800, 980], 'Name', 'Generalized Freeform: Poly-only vs Poly+PHS');

subplot(3,4,1);
surf(X, Y, Z_true, 'EdgeColor', 'none');
title('Global True Surface');
xlabel('X'); ylabel('Y'); zlabel('Z');
axis tight; axis equal; grid on; view([-35, 35]); colormap jet; colorbar;

subplot(3,4,2);
surf(X, Y, Z_cubic_true, 'EdgeColor', 'none');
title('Global Cubic Component');
xlabel('X'); ylabel('Y'); zlabel('Z');
axis tight; axis equal; grid on; view([-35, 35]); colormap jet; colorbar;

% subplot(3,4,3);
% surf(X, Y, Z_residual_true, 'EdgeColor', 'none');
% title('Global Non-polynomial Residual');
% xlabel('X'); ylabel('Y'); zlabel('Z');
% axis tight; axis equal; grid on; view([-35, 35]); colormap jet; colorbar;

subplot(3,4,3);
imagesc(x_vec, y_vec, double(mask_full_sub1));
set(gca, 'YDir', 'normal');
title('Sub-aperture 1 Mask');
xlabel('X'); ylabel('Y');
axis image; colorbar;

subplot(3,4,4);
imagesc(x_vec, y_vec, double(mask_full_sub2));
set(gca, 'YDir', 'normal');
title('Sub-aperture 2 Mask');
xlabel('X'); ylabel('Y');
axis image; colorbar;

subplot(3,4,5);
imagesc(sub1_grid.x_vec, sub1_grid.y_vec, Res1_poly_grid);
set(gca, 'YDir', 'normal');
title(sprintf('Sub1 Poly-only Error, RMSE=%.3e', metrics_sub1_poly.rmse));
xlabel('X'); ylabel('Y');
axis image; colormap jet; colorbar;


subplot(3,4,6);
imagesc(sub1_grid.x_vec, sub1_grid.y_vec, Z1_res_true_grid);
set(gca, 'YDir', 'normal');
title('Sub1 True Residual Surface');
xlabel('X'); ylabel('Y');
axis image; colormap jet; colorbar;

subplot(3,4,7);
imagesc(sub1_grid.x_vec, sub1_grid.y_vec, Z1_res_fit_grid);
set(gca, 'YDir', 'normal');
title('Sub1 PHS Residual Fit');
xlabel('X'); ylabel('Y');
axis image; colormap jet; colorbar;

subplot(3,4,8);
imagesc(sub1_grid.x_vec, sub1_grid.y_vec, Res1_grid);
set(gca, 'YDir', 'normal');
title(sprintf('Sub1 Final Error, RMSE=%.3e', metrics_sub1_grid.rmse));
xlabel('X'); ylabel('Y');
axis image; colormap jet; colorbar;

% subplot(3,4,9);
% imagesc(sub1_grid.x_vec, sub1_grid.y_vec, Res1_residual_grid);
% set(gca, 'YDir', 'normal');
% title(sprintf('Sub1 Residual-fit Error, RMSE=%.3e', metrics_sub1_residual.rmse));
% xlabel('X'); ylabel('Y');
% axis image; colormap jet; colorbar;

subplot(3,4,9);
imagesc(sub2_grid.x_vec, sub2_grid.y_vec, Res2_poly_grid);
set(gca, 'YDir', 'normal');
title(sprintf('Sub2 Poly-only Error, RMSE=%.3e', metrics_sub2_poly.rmse));
xlabel('X'); ylabel('Y');
axis image; colormap jet; colorbar;


subplot(3,4,10);
imagesc(sub2_grid.x_vec, sub2_grid.y_vec, Z2_res_true_grid);
set(gca, 'YDir', 'normal');
title('Sub2 True Residual Surface');
xlabel('X'); ylabel('Y');
axis image; colormap jet; colorbar;

subplot(3,4,11);
imagesc(sub2_grid.x_vec, sub2_grid.y_vec, Z2_res_fit_grid);
set(gca, 'YDir', 'normal');
title('Sub2 PHS Residual Fit');
xlabel('X'); ylabel('Y');
axis image; colormap jet; colorbar;


subplot(3,4,12);
imagesc(sub2_grid.x_vec, sub2_grid.y_vec, Res2_grid);
set(gca, 'YDir', 'normal');
title(sprintf('Sub2 Final Error, RMSE=%.3e', metrics_sub2_grid.rmse));
xlabel('X'); ylabel('Y');
axis image; colormap jet; colorbar;

% subplot(3,5,15);
% imagesc(sub2_grid.x_vec, sub2_grid.y_vec, Res2_residual_grid);
% set(gca, 'YDir', 'normal');
% title(sprintf('Sub2 Residual-fit Error, RMSE=%.3e', metrics_sub2_residual.rmse));
% xlabel('X'); ylabel('Y');
% axis image; colormap jet; colorbar;

figure('Position', [90, 90, 1400, 450], 'Name', 'Residual Histograms');

subplot(1,4,1);
histogram(Res1_poly_grid(sub1_mask), 80);
title('Sub1 Poly-only Error');
xlabel('Residual'); ylabel('Count'); grid on;

subplot(1,4,2);
histogram(Res1_grid(sub1_mask), 80);
title('Sub1 Final Error');
xlabel('Residual'); ylabel('Count'); grid on;

subplot(1,4,3);
histogram(Res2_poly_grid(sub2_mask), 80);
title('Sub2 Poly-only Error');
xlabel('Residual'); ylabel('Count'); grid on;

subplot(1,4,4);
histogram(Res2_grid(sub2_mask), 80);
title('Sub2 Final Error');
xlabel('Residual'); ylabel('Count'); grid on;

figure('Position', [120, 120, 1200, 500], 'Name', '3D Global Surface with Sampling Boundaries');
surf(X, Y, Z_true, 'EdgeColor', 'none'); hold on;
plot_mask_boundary(X, Y, mask_full_sub1, zmax, 'w', 2.0);
plot_mask_boundary(X, Y, mask_full_sub2, zmax, 'k', 2.0);
title('Global Surface with Two Sub-apertures');
xlabel('X'); ylabel('Y'); zlabel('Z');
axis tight; axis equal; grid on; view([-35, 35]); colormap jet; colorbar;

%% 11. 保存 MAT 文件
save_name = 'phs_freeform_fit_data_generalized.mat';

save(save_name, ...
    'surface_params', ...
    'fit_opts', ...
    'x_vec', 'y_vec', 'X', 'Y', ...
    'Z_true', 'Z_meas', 'Z_cubic_true', 'Z_residual_true', ...
    'point_cloud_full', ...
    'idx_sub1', 'idx_sub2', ...
    'mask_full_sub1', 'mask_full_sub2', ...
    'point_cloud_sub1', 'point_cloud_sub2', ...
    'sub1_grid', 'sub2_grid', ...
    'sub1_mask', 'sub2_mask', ...
    'model_sub1', 'model_sub2', ...
    'z_fit_sub1_pts', 'z_fit_sub2_pts', ...
    'z_poly_sub1_pts', 'z_poly_sub2_pts', ...
    'res_sub1_meas', 'res_sub2_meas', ...
    'res_sub1_true', 'res_sub2_true', ...
    'res_sub1_poly_true', 'res_sub2_poly_true', ...
    'metrics_sub1_meas', 'metrics_sub2_meas', ...
    'metrics_sub1_true', 'metrics_sub2_true', ...
    'metrics_sub1_poly_true', 'metrics_sub2_poly_true', ...
    'Z1_fit_grid', 'Z2_fit_grid', ...
    'Z1_poly_grid', 'Z2_poly_grid', ...
    'Z1_true_grid', 'Z2_true_grid', ...
    'Res1_grid', 'Res2_grid', ...
    'Res1_poly_grid', 'Res2_poly_grid', ...
    'metrics_sub1_grid', 'metrics_sub2_grid', ...
    'metrics_sub1_poly', 'metrics_sub2_poly', ...
    'Z1_res_true_grid', 'Z2_res_true_grid', ...
    'Z1_res_fit_grid', 'Z2_res_fit_grid', ...
    'Res1_residual_grid', 'Res2_residual_grid', ...
    'metrics_sub1_residual', 'metrics_sub2_residual', ...
    '-v7.3');

disp(['已保存 MAT 文件：', save_name]);

%% ========================= 局部函数 =========================

function [Z, Z_cubic, Z_residual] = surface_freeform_general(X, Y, p)
    Z_cubic = p.c1 .* X.^3 ...
            + p.c2 .* (X.^2) .* Y ...
            + p.c3 .* X .* (Y.^2) ...
            + p.c4 .* Y.^3 ...
            + p.c5 .* X.^2 ...
            + p.c6 .* Y.^2;

    G1 = p.bump1_amp .* exp( ...
        -((X - p.bump1_x0).^2 ./ (2 * p.bump1_sx^2) + ...
          (Y - p.bump1_y0).^2 ./ (2 * p.bump1_sy^2)));

    G2 = p.bump2_amp .* exp( ...
        -((X - p.bump2_x0).^2 ./ (2 * p.bump2_sx^2) + ...
          (Y - p.bump2_y0).^2 ./ (2 * p.bump2_sy^2)));

    G3 = p.bump3_amp .* exp( ...
        -((X - p.bump3_x0).^2 ./ (2 * p.bump3_sx^2) + ...
          (Y - p.bump3_y0).^2 ./ (2 * p.bump3_sy^2)));

    R1 = p.ripple_amp .* sin(2*pi*X./p.ripple_Lx + p.ripple_phi) .* cos(2*pi*Y./p.ripple_Ly);
    R2 = p.cos_amp    .* cos(2*pi*(X + 0.4*Y)./p.cos_Lx) .* cos(2*pi*(Y - 0.2*X)./p.cos_Ly);

    Z_residual = G1 + G2 + G3 + R1 + R2;
    Z = Z_cubic + Z_residual;
end

function [pc_sub, idx_sub] = extract_subaperture(point_cloud_full, center_xy, R)
    dist_sq = (point_cloud_full(:,1) - center_xy(1)).^2 + (point_cloud_full(:,2) - center_xy(2)).^2;
    idx_sub = dist_sq <= R^2;
    pc_sub = point_cloud_full(idx_sub, :);
end

function [grid_struct, local_mask] = build_support_grid_from_full_grid(X, Y, Z_true, Z_meas, Z_cubic_true, Z_residual_true, mask_full)
    [row_idx, col_idx] = find(mask_full);

    rmin = min(row_idx);
    rmax = max(row_idx);
    cmin = min(col_idx);
    cmax = max(col_idx);

    rows = rmin:rmax;
    cols = cmin:cmax;

    grid_struct = struct();
    grid_struct.X = X(rows, cols);
    grid_struct.Y = Y(rows, cols);
    grid_struct.Z_true = Z_true(rows, cols);
    grid_struct.Z_meas = Z_meas(rows, cols);
    grid_struct.Z_cubic_true = Z_cubic_true(rows, cols);
    grid_struct.Z_residual_true = Z_residual_true(rows, cols);
    grid_struct.x_vec = grid_struct.X(1, :);
    grid_struct.y_vec = grid_struct.Y(:, 1);
    grid_struct.row_range = rows;
    grid_struct.col_range = cols;

    local_mask = mask_full(rows, cols);
end

function metrics = calc_metrics(residual_vec)
    residual_vec = residual_vec(:);
    residual_vec = residual_vec(isfinite(residual_vec));

    metrics = struct();
    metrics.rmse = sqrt(mean(residual_vec.^2));
    metrics.max_abs = max(abs(residual_vec));
    metrics.mean_abs = mean(abs(residual_vec));
    metrics.std = std(residual_vec);
end

function plot_mask_boundary(X, Y, mask_full, z_level, line_color, line_width)
    C = contourc(X(1,:), Y(:,1), double(mask_full), [0.5 0.5]);

    idx = 1;
    while idx < size(C, 2)
        npt = C(2, idx);
        xs = C(1, idx+1:idx+npt);
        ys = C(2, idx+1:idx+npt);
        zs = z_level .* ones(size(xs));
        plot3(xs, ys, zs, '-', 'Color', line_color, 'LineWidth', line_width);
        idx = idx + npt + 1;
    end
end

function model = fit_poly_plus_phs(x, y, z, opts)
    x = x(:);
    y = y(:);
    z = z(:);

    valid = isfinite(x) & isfinite(y) & isfinite(z);
    x = x(valid);
    y = y(valid);
    z = z(valid);

    polyModel = fit_poly3_surface(x, y, z, opts.normalize_poly);
    z_poly = eval_poly_model(polyModel, x, y);

    residual = z - z_poly;
    residual_rmse = sqrt(mean(residual.^2));

    if opts.skip_phs_if_small && residual_rmse < opts.residual_tol
        phsModel = make_empty_phs_model(x, y, residual, opts);
        phsModel.skipped = true;
        phsModel.residual_rmse_before = residual_rmse;
    else
        phsModel = fit_phs_residual(x, y, residual, opts);
        phsModel.skipped = false;
        phsModel.residual_rmse_before = residual_rmse;
    end

    model = struct();
    model.polyModel = polyModel;
    model.phsModel = phsModel;
end

function polyModel = fit_poly3_surface(x, y, z, normalize_flag)
    x = x(:);
    y = y(:);
    z = z(:);

    if normalize_flag
        xc = mean(x);
        yc = mean(y);
        sx = std(x); if sx < eps, sx = 1; end
        sy = std(y); if sy < eps, sy = 1; end
        xn = (x - xc) ./ sx;
        yn = (y - yc) ./ sy;
    else
        xc = 0; yc = 0;
        sx = 1; sy = 1;
        xn = x;
        yn = y;
    end

    P = poly_basis_cubic(xn, yn);
    coef = P \ z;

    polyModel = struct();
    polyModel.normalize = normalize_flag;
    polyModel.center_xy = [xc, yc];
    polyModel.scale_xy = [sx, sy];
    polyModel.coef = coef;
end

function zq = eval_poly_model(polyModel, xq, yq)
    sz = size(xq);
    xq = xq(:);
    yq = yq(:);

    if polyModel.normalize
        xqn = (xq - polyModel.center_xy(1)) ./ polyModel.scale_xy(1);
        yqn = (yq - polyModel.center_xy(2)) ./ polyModel.scale_xy(2);
    else
        xqn = xq;
        yqn = yq;
    end

    Pq = poly_basis_cubic(xqn, yqn);
    zq = Pq * polyModel.coef;
    zq = reshape(zq, sz);
end

function phsModel = fit_phs_residual(x, y, residual, opts)
    x = x(:);
    y = y(:);
    residual = residual(:);

    if opts.normalize_phs
        xc = mean(x);
        yc = mean(y);
        sx = std(x); if sx < eps, sx = 1; end
        sy = std(y); if sy < eps, sy = 1; end
        xn = (x - xc) ./ sx;
        yn = (y - yc) ./ sy;
    else
        xc = 0; yc = 0;
        sx = 1; sy = 1;
        xn = x;
        yn = y;
    end

    center_idx = select_uniform_centers(xn, yn, opts.max_centers);
    xcens = xn(center_idx);
    ycens = yn(center_idx);

    Phi = phs_kernel_matrix(xn, yn, xcens, ycens, opts.kernel);
    M = numel(center_idx);

    A_aug = [Phi; sqrt(opts.lambda) * eye(M)];
    b_aug = [residual; zeros(M, 1)];

    omega = A_aug \ b_aug;

    phsModel = struct();
    phsModel.kernel = opts.kernel;
    phsModel.lambda = opts.lambda;
    phsModel.normalize = opts.normalize_phs;
    phsModel.center_xy = [xc, yc];
    phsModel.scale_xy = [sx, sy];
    phsModel.center_idx = center_idx;
    phsModel.centers_n = [xcens, ycens];
    phsModel.omega = omega;
    phsModel.num_centers = M;
end

function phsModel = make_empty_phs_model(x, y, residual, opts)
    x = x(:);
    y = y(:);
    residual = residual(:); 

    if opts.normalize_phs
        xc = mean(x);
        yc = mean(y);
        sx = std(x); if sx < eps, sx = 1; end
        sy = std(y); if sy < eps, sy = 1; end
    else
        xc = 0; yc = 0;
        sx = 1; sy = 1;
    end

    phsModel = struct();
    phsModel.kernel = opts.kernel;
    phsModel.lambda = opts.lambda;
    phsModel.normalize = opts.normalize_phs;
    phsModel.center_xy = [xc, yc];
    phsModel.scale_xy = [sx, sy];
    phsModel.center_idx = [];
    phsModel.centers_n = zeros(0,2);
    phsModel.omega = zeros(0,1);
    phsModel.num_centers = 0;
    phsModel.skipped = true;
end

function zq = eval_poly_plus_phs(model, xq, yq)
    z_poly = eval_poly_model(model.polyModel, xq, yq);
    z_phs = eval_phs_residual_model(model.phsModel, xq, yq);
    zq = z_poly + z_phs;
end

function zq = eval_phs_residual_model(phsModel, xq, yq)
    sz = size(xq);
    xq = xq(:);
    yq = yq(:);

    if isempty(phsModel.omega)
        zq = zeros(size(xq));
        zq = reshape(zq, sz);
        return;
    end

    if phsModel.normalize
        xqn = (xq - phsModel.center_xy(1)) ./ phsModel.scale_xy(1);
        yqn = (yq - phsModel.center_xy(2)) ./ phsModel.scale_xy(2);
    else
        xqn = xq;
        yqn = yq;
    end

    Phi_q = phs_kernel_matrix(xqn, yqn, phsModel.centers_n(:,1), phsModel.centers_n(:,2), phsModel.kernel);
    zq = Phi_q * phsModel.omega;
    zq = reshape(zq, sz);
end

function K = phs_kernel_matrix(x1, y1, x2, y2, kernel_name)
    x1 = x1(:);
    y1 = y1(:);
    x2 = x2(:).';
    y2 = y2(:).';

    dx = x1 - x2;
    dy = y1 - y2;
    r2 = dx.^2 + dy.^2;

    K = zeros(size(r2));

    switch lower(kernel_name)
        case 'r4logr'
            mask = r2 > 0;
            K(mask) = 0.5 .* (r2(mask).^2) .* log(r2(mask));
        case 'r6logr'
            mask = r2 > 0;
            K(mask) = 0.5 .* (r2(mask).^3) .* log(r2(mask));
        otherwise
            error('未实现的核函数类型。');
    end
end

function P = poly_basis_cubic(x, y)
    x = x(:);
    y = y(:);

    P = [ ...
        ones(size(x)), ...
        x, ...
        y, ...
        x.^2, ...
        x .* y, ...
        y.^2, ...
        x.^3, ...
        x.^2 .* y, ...
        x .* y.^2, ...
        y.^3];
end

function idx = select_uniform_centers(x, y, max_centers)
    N = numel(x);
    if N <= max_centers
        idx = (1:N).';
        return;
    end

    xrange = max(x) - min(x);
    yrange = max(y) - min(y);

    if xrange < eps
        xrange = 1;
    end
    if yrange < eps
        yrange = 1;
    end

    nx = max(8, round(sqrt(max_centers * xrange / yrange)));
    ny = max(8, round(max_centers / nx));

    x_edges = linspace(min(x), max(x), nx + 1);
    y_edges = linspace(min(y), max(y), ny + 1);
    x_edges(end) = x_edges(end) + eps;
    y_edges(end) = y_edges(end) + eps;

    [~,~,bx] = histcounts(x, x_edges);
    [~,~,by] = histcounts(y, y_edges);

    valid = bx > 0 & by > 0;
    lin_id = sub2ind([ny, nx], by(valid), bx(valid));
    raw_id = find(valid);

    [~, ia] = unique(lin_id, 'stable');
    idx = raw_id(ia);

    if numel(idx) > max_centers
        keep = round(linspace(1, numel(idx), max_centers));
        idx = idx(keep);
    end

    idx = idx(:);
end