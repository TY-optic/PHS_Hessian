%% step5_local_smooth_error_evaluation.m
% Step 5:
% 1) 读取 Step1 与 Step4 的结果
% 2) 在统一 global 网格上分别重建 sub1 / sub2_final 的离散面
% 3) 用边界距离权重进行重叠区融合，得到离散融合面 Z_fused_grid
% 4) 直接在有效拼接区域上计算“全局误差结果”
% 5) 使用局部 MLS 二次光滑函数对离散融合面做局部平滑评估
% 6) 输出离散融合误差、局部平滑误差、重叠区一致性误差
%
% 说明：
% - 主精度指标：离散融合面误差（不引入额外全局模型逼近误差）
% - 辅助平滑指标：局部 MLS 二次光滑误差（用于可视化与局部连续评估）
% - 不再强制拟合全局单一连续曲面
%
% 如果 Step4 保存文件名不同，请修改 data_step4_candidates 的候选文件名。

clear; clc; close all;

%% ========================= 1. 读取数据 =========================
data_step1 = 'phs_freeform_fit_data_generalized.mat';
data_step4_candidates = { ...
    'step4_registration_only_results_generalized.mat', ...
    'step4_3D_reconstruction_results_generalized.mat' ...
    };

if ~isfile(data_step1)
    error('找不到 %s，请先运行 Step1。', data_step1);
end

data_step4 = '';
for k = 1:numel(data_step4_candidates)
    if isfile(data_step4_candidates{k})
        data_step4 = data_step4_candidates{k};
        break;
    end
end
if isempty(data_step4)
    error('找不到 Step4 结果文件，请确认 Step4 已运行并保存。');
end

S1 = load(data_step1);
S4 = load(data_step4);

required_s1 = { ...
    'X', 'Y', 'Z_true', 'x_vec', 'y_vec', ...
    'surface_params' ...
    };

for k = 1:numel(required_s1)
    if ~isfield(S1, required_s1{k})
        error('Step1 MAT 文件缺少变量：%s', required_s1{k});
    end
end

required_s4_primary = {'pc1_global', 'pc2_final'};
for k = 1:numel(required_s4_primary)
    if ~isfield(S4, required_s4_primary{k})
        error('Step4 MAT 文件缺少变量：%s', required_s4_primary{k});
    end
end

Xg = S1.X;
Yg = S1.Y;
Z_true = S1.Z_true;
x_vec = S1.x_vec;
y_vec = S1.y_vec;

pc1_global = S4.pc1_global;
pc2_final  = S4.pc2_final;

dx = abs(x_vec(2) - x_vec(1));
dy = abs(y_vec(2) - y_vec(1));

fprintf('================ Step5: local smooth error evaluation =================\n');
fprintf('Using Step4 file: %s\n', data_step4);

%% ========================= 2. 参数设置 =========================
opt = struct();

% 全局投影到规则网格
opt.boundary_shrink = 0.95;
opt.interp_method = 'natural';
opt.extrap_method = 'none';

% 融合权重
opt.boundary_soft_width_pix = 8;

% 是否对重叠区先做低阶一致性校正
opt.use_overlap_planar_correction = false;
opt.use_overlap_quadratic_correction = false;   % 若为 true，则忽略 planar 开关
opt.min_overlap_corr_points = 500;
opt.max_corr_order = 2;                         % 1=平面, 2=二次

% 局部平滑评估器：MLS 二次
opt.smooth.enable = true;
opt.smooth.radius_pix = 5;                      % 邻域半径（像素）
opt.smooth.sigma_pix = 2.5;                     % 高斯权尺度（像素）
opt.smooth.min_points = 18;
opt.smooth.ridge = 1e-10;
opt.smooth.use_confidence_weight = true;
opt.smooth.progress_step = 2000;

% 是否仅在有效拼接区域上计算误差
opt.evaluate_only_on_union_support = true;

%% ========================= 3. 将两个子孔径投影到统一 global 网格 =========================
[mask1_grid, Z1_global_grid] = project_point_cloud_to_global_grid( ...
    pc1_global, Xg, Yg, opt.boundary_shrink, opt.interp_method, opt.extrap_method);

[mask2_grid, Z2_global_grid] = project_point_cloud_to_global_grid( ...
    pc2_final, Xg, Yg, opt.boundary_shrink, opt.interp_method, opt.extrap_method);

valid1 = mask1_grid & isfinite(Z1_global_grid);
valid2 = mask2_grid & isfinite(Z2_global_grid);

fprintf('Sub1 projected support points = %d\n', nnz(valid1));
fprintf('Sub2 projected support points = %d\n', nnz(valid2));

%% ========================= 4. 可选：重叠区低阶一致性校正 =========================
corr_model = struct('type','none','coef',[]);
Z2_global_grid_corr = Z2_global_grid;

mask_overlap_raw = valid1 & valid2;
if nnz(mask_overlap_raw) >= opt.min_overlap_corr_points
    if opt.use_overlap_quadratic_correction
        corr_order = 2;
    elseif opt.use_overlap_planar_correction
        corr_order = 1;
    else
        corr_order = 0;
    end

    if corr_order > 0
        x_ov = Xg(mask_overlap_raw);
        y_ov = Yg(mask_overlap_raw);
        dz_ov = Z1_global_grid(mask_overlap_raw) - Z2_global_grid(mask_overlap_raw);

        coef = fit_low_order_delta_surface(x_ov, y_ov, dz_ov, corr_order);
        dZ_corr = eval_low_order_delta_surface(coef, Xg, Yg, corr_order);

        Z2_global_grid_corr(valid2) = Z2_global_grid(valid2) + dZ_corr(valid2);

        if corr_order == 1
            corr_model.type = 'planar';
        else
            corr_model.type = 'quadratic';
        end
        corr_model.coef = coef;

        fprintf('Applied overlap low-order correction: %s\n', corr_model.type);
    end
end

valid2_corr = mask2_grid & isfinite(Z2_global_grid_corr);
mask_overlap = valid1 & valid2_corr;
mask_union   = valid1 | valid2_corr;

fprintf('Overlap support points = %d\n', nnz(mask_overlap));
fprintf('Union support points   = %d\n', nnz(mask_union));

%% ========================= 5. 构造边界权重并进行离散融合 =========================
Wbd1 = build_boundary_weight(valid1, opt.boundary_soft_width_pix);
Wbd2 = build_boundary_weight(valid2_corr, opt.boundary_soft_width_pix);

W1 = Wbd1;
W2 = Wbd2;

W1(~valid1) = 0;
W2(~valid2_corr) = 0;

Z_fused_grid = nan(size(Xg));
Wsum = W1 + W2;

only1 = valid1 & ~valid2_corr;
only2 = valid2_corr & ~valid1;
both  = mask_overlap;

Z_fused_grid(only1) = Z1_global_grid(only1);
Z_fused_grid(only2) = Z2_global_grid_corr(only2);

idx_both = both & (Wsum > 0);
Z_fused_grid(idx_both) = ...
    (W1(idx_both) .* Z1_global_grid(idx_both) + W2(idx_both) .* Z2_global_grid_corr(idx_both)) ./ Wsum(idx_both);

idx_both_zero = both & ~(Wsum > 0);
Z_fused_grid(idx_both_zero) = 0.5 * (Z1_global_grid(idx_both_zero) + Z2_global_grid_corr(idx_both_zero));

valid_fused = mask_union & isfinite(Z_fused_grid);

%% ========================= 6. 直接输出“全局误差结果” =========================
err1_grid = Z1_global_grid - Z_true;
err2_grid = Z2_global_grid_corr - Z_true;
err_fused_grid = Z_fused_grid - Z_true;

err1_grid(~valid1) = nan;
err2_grid(~valid2_corr) = nan;
err_fused_grid(~valid_fused) = nan;

metrics_sub1 = calc_field_metrics(err1_grid, valid1);
metrics_sub2 = calc_field_metrics(err2_grid, valid2_corr);
metrics_fused = calc_field_metrics(err_fused_grid, valid_fused);

fprintf('Sub1 projected surface RMSE = %.6e mm\n', metrics_sub1.rmse);
fprintf('Sub2 projected surface RMSE = %.6e mm\n', metrics_sub2.rmse);
fprintf('Discrete fused surface RMSE = %.6e mm\n', metrics_fused.rmse);

%% ========================= 7. 重叠区一致性误差诊断 =========================
overlap_diff_before = Z1_global_grid - Z2_global_grid;
overlap_diff_after  = Z1_global_grid - Z2_global_grid_corr;
overlap_diff_fused  = Z1_global_grid - Z_fused_grid;

overlap_diff_before(~mask_overlap_raw) = nan;
overlap_diff_after(~mask_overlap) = nan;
overlap_diff_fused(~mask_overlap) = nan;

metrics_overlap_before = calc_field_metrics(overlap_diff_before, mask_overlap_raw);
metrics_overlap_after  = calc_field_metrics(overlap_diff_after,  mask_overlap);
metrics_overlap_fused  = calc_field_metrics(overlap_diff_fused,  mask_overlap);

fprintf('Overlap difference before correction RMSE = %.6e mm\n', metrics_overlap_before.rmse);
fprintf('Overlap difference after  correction RMSE = %.6e mm\n', metrics_overlap_after.rmse);
fprintf('Overlap difference fused-to-sub1   RMSE   = %.6e mm\n', metrics_overlap_fused.rmse);

%% ========================= 8. 局部 MLS 二次光滑评估 =========================
Z_local_smooth_grid = nan(size(Xg));
err_local_smooth = nan(size(Xg));
metrics_local_smooth = struct('rmse', nan, 'mae', nan, 'pv', nan, 'mean', nan, 'std', nan);

if opt.smooth.enable
    fprintf('---------------- Running local MLS smooth evaluation ----------------\n');

    conf_map = W1 + W2;
    conf_map = conf_map ./ max(conf_map(:) + eps);

    Z_local_smooth_grid = local_mls_quadratic_smooth( ...
        Xg, Yg, Z_fused_grid, valid_fused, conf_map, dx, dy, opt.smooth);

    valid_smooth = valid_fused & isfinite(Z_local_smooth_grid);
    err_local_smooth = Z_local_smooth_grid - Z_true;
    err_local_smooth(~valid_smooth) = nan;

    metrics_local_smooth = calc_field_metrics(err_local_smooth, valid_smooth);
    fprintf('Local smooth surface RMSE = %.6e mm\n', metrics_local_smooth.rmse);
end

%% ========================= 9. 可视化 =========================
figure('Position', [30, 40, 1750, 980], 'Name', 'Step5 Local Smooth Error Evaluation', 'Color', 'w');

subplot(2,3,1);
imagesc(x_vec, y_vec, Z1_global_grid);
set(gca, 'YDir', 'normal'); axis image; colorbar;
title(sprintf('1. Sub1 projected surface\nRMSE = %.2e mm', metrics_sub1.rmse));
xlabel('Global X (mm)'); ylabel('Global Y (mm)');

subplot(2,3,2);
imagesc(x_vec, y_vec, Z2_global_grid_corr);
set(gca, 'YDir', 'normal'); axis image; colorbar;
title(sprintf('2. Sub2 final projected surface\nRMSE = %.2e mm', metrics_sub2.rmse));
xlabel('Global X (mm)'); ylabel('Global Y (mm)');

subplot(2,3,3);
imagesc(x_vec, y_vec, Z_fused_grid);
set(gca, 'YDir', 'normal'); axis image; colorbar;
title(sprintf('3. Discrete fused surface\nRMSE = %.2e mm', metrics_fused.rmse));
xlabel('Global X (mm)'); ylabel('Global Y (mm)');

subplot(2,3,4);
imagesc(x_vec, y_vec, err_fused_grid);
set(gca, 'YDir', 'normal'); axis image; colorbar;
title(sprintf('4. Discrete fusion error\nRMSE = %.2e mm', metrics_fused.rmse));
xlabel('Global X (mm)'); ylabel('Global Y (mm)');

subplot(2,3,5);
if opt.smooth.enable
    imagesc(x_vec, y_vec, err_local_smooth);
    set(gca, 'YDir', 'normal'); axis image; colorbar;
    title(sprintf('5. Local smooth error\nRMSE = %.2e mm', metrics_local_smooth.rmse));
    xlabel('Global X (mm)'); ylabel('Global Y (mm)');
else
    text(0.5, 0.5, 'Local smooth disabled', 'HorizontalAlignment', 'center');
    axis off;
end

subplot(2,3,6);
err_hist = err_fused_grid(valid_fused);
err_hist = err_hist(isfinite(err_hist));
if numel(err_hist) > 20
    histogram(err_hist, 60, 'Normalization', 'pdf');
    xlabel('Residual (mm)');
    ylabel('PDF');
    title('6. Fused discrete residual histogram');
    grid on;
else
    text(0.5, 0.5, 'Insufficient valid residuals', 'HorizontalAlignment', 'center');
    axis off;
end

%% ========================= 10. 重叠区诊断图 =========================
figure('Position', [80, 80, 1500, 520], 'Name', 'Overlap Diagnostics', 'Color', 'w');

subplot(1,4,1);
imagesc(x_vec, y_vec, double(mask_overlap));
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('1. Overlap mask');
xlabel('Global X (mm)'); ylabel('Global Y (mm)');

subplot(1,4,2);
imagesc(x_vec, y_vec, overlap_diff_before);
set(gca, 'YDir', 'normal'); axis image; colorbar;
title(sprintf('2. Z1 - Z2 before\nRMSE = %.2e mm', metrics_overlap_before.rmse));
xlabel('Global X (mm)'); ylabel('Global Y (mm)');

subplot(1,4,3);
imagesc(x_vec, y_vec, overlap_diff_after);
set(gca, 'YDir', 'normal'); axis image; colorbar;
title(sprintf('3. Z1 - Z2 after corr\nRMSE = %.2e mm', metrics_overlap_after.rmse));
xlabel('Global X (mm)'); ylabel('Global Y (mm)');

subplot(1,4,4);
W_ratio = nan(size(Xg));
W_ratio(mask_overlap) = W1(mask_overlap) ./ (W1(mask_overlap) + W2(mask_overlap) + eps);
imagesc(x_vec, y_vec, W_ratio);
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('4. Fusion weight ratio W1/(W1+W2)');
xlabel('Global X (mm)'); ylabel('Global Y (mm)');

%% ========================= 11. 输出主结论 =========================
fprintf('\n================ Final global error result =================\n');
fprintf('Primary metric (discrete fused surface):\n');
fprintf('  RMSE = %.6e mm\n', metrics_fused.rmse);
fprintf('  MAE  = %.6e mm\n', metrics_fused.mae);
fprintf('  PV   = %.6e mm\n', metrics_fused.pv);
fprintf('  STD  = %.6e mm\n', metrics_fused.std);

if opt.smooth.enable
    fprintf('Auxiliary metric (local smooth evaluator):\n');
    fprintf('  RMSE = %.6e mm\n', metrics_local_smooth.rmse);
    fprintf('  MAE  = %.6e mm\n', metrics_local_smooth.mae);
    fprintf('  PV   = %.6e mm\n', metrics_local_smooth.pv);
    fprintf('  STD  = %.6e mm\n', metrics_local_smooth.std);
end

%% ========================= 12. 保存结果 =========================
result_step5 = struct();

result_step5.metrics.sub1_projected = metrics_sub1;
result_step5.metrics.sub2_projected = metrics_sub2;
result_step5.metrics.fused_discrete = metrics_fused;
result_step5.metrics.local_smooth = metrics_local_smooth;

result_step5.metrics.overlap_before = metrics_overlap_before;
result_step5.metrics.overlap_after = metrics_overlap_after;
result_step5.metrics.overlap_fused = metrics_overlap_fused;

result_step5.support.sub1_count = nnz(valid1);
result_step5.support.sub2_count = nnz(valid2_corr);
result_step5.support.overlap_count = nnz(mask_overlap);
result_step5.support.union_count = nnz(valid_fused);

result_step5.correction = corr_model;
result_step5.options = opt;

save_name = 'step5_local_smooth_error_evaluation_results.mat';
save(save_name, ...
    'result_step5', ...
    'Z1_global_grid', 'Z2_global_grid', 'Z2_global_grid_corr', ...
    'mask1_grid', 'mask2_grid', ...
    'W1', 'W2', ...
    'mask_overlap', 'mask_union', ...
    'Z_fused_grid', 'err_fused_grid', ...
    'Z_local_smooth_grid', 'err_local_smooth', ...
    'overlap_diff_before', 'overlap_diff_after', 'overlap_diff_fused', ...
    '-v7.3');

disp(['Saved Step5 result file: ', save_name]);

%% ========================= 局部函数 =========================

function [mask_grid, Z_grid] = project_point_cloud_to_global_grid(pc, Xg, Yg, shrink_factor, interp_method, extrap_method)
x = pc(:,1);
y = pc(:,2);
z = pc(:,3);

k = boundary(x, y, shrink_factor);
bx = x(k);
by = y(k);

mask_grid = inpolygon(Xg, Yg, bx, by);

F = scatteredInterpolant(x, y, z, interp_method, extrap_method);
Z_grid = F(Xg, Yg);

mask_grid = mask_grid & isfinite(Z_grid);
Z_grid(~mask_grid) = nan;
end

function coef = fit_low_order_delta_surface(x, y, dz, order)
x = x(:); y = y(:); dz = dz(:);

x0 = mean(x);
y0 = mean(y);
xs = max(std(x), eps);
ys = max(std(y), eps);

xn = (x - x0) ./ xs;
yn = (y - y0) ./ ys;

switch order
    case 1
        P = [ones(size(xn)), xn, yn];
    case 2
        P = [ones(size(xn)), xn, yn, xn.^2, xn.*yn, yn.^2];
    otherwise
        error('仅支持 1 阶或 2 阶低阶校正。');
end

coef.beta = P \ dz;
coef.order = order;
coef.x0 = x0;
coef.y0 = y0;
coef.xs = xs;
coef.ys = ys;
end

function dZ = eval_low_order_delta_surface(coef, X, Y, order)
xn = (X - coef.x0) ./ coef.xs;
yn = (Y - coef.y0) ./ coef.ys;

switch order
    case 1
        P = [ones(numel(X),1), xn(:), yn(:)];
    case 2
        P = [ones(numel(X),1), xn(:), yn(:), xn(:).^2, xn(:).*yn(:), yn(:).^2];
    otherwise
        error('仅支持 1 阶或 2 阶低阶校正。');
end

dZ = reshape(P * coef.beta, size(X));
end

function Wbd = build_boundary_weight(mask, soft_width_pix)
mask = logical(mask);
dist_layers = boundary_distance_layers(mask);
Wbd = min(dist_layers / max(soft_width_pix, 1), 1);
Wbd(~mask) = 0;
end

function dist_layers = boundary_distance_layers(mask)
mask = logical(mask);
dist_layers = zeros(size(mask));

cur = mask;
layer = 0;
while any(cur(:))
    layer = layer + 1;
    inner = erode_binary_mask(cur, 1);
    boundary = cur & ~inner;
    dist_layers(boundary) = layer;
    cur = inner;
end
end

function mask_erode = erode_binary_mask(mask_in, n_step)
mask_erode = logical(mask_in);
for k = 1:n_step
    mask_erode = (conv2(double(mask_erode), ones(3), 'same') == 9) & mask_erode;
end
end

function Z_smooth = local_mls_quadratic_smooth(X, Y, Z, valid_mask, conf_map, dx, dy, smooth_opt)
[Ny, Nx] = size(Z);
Z_smooth = nan(Ny, Nx);

radius = smooth_opt.radius_pix;
sigma2 = (smooth_opt.sigma_pix)^2;
min_points = smooth_opt.min_points;
ridge = smooth_opt.ridge;
use_conf = smooth_opt.use_confidence_weight;
progress_step = smooth_opt.progress_step;

valid_idx = find(valid_mask);
num_valid = numel(valid_idx);

for k = 1:num_valid
    idx0 = valid_idx(k);
    [iy, ix] = ind2sub([Ny, Nx], idx0);

    r1 = max(1, iy - radius);
    r2 = min(Ny, iy + radius);
    c1 = max(1, ix - radius);
    c2 = min(Nx, ix + radius);

    Zloc = Z(r1:r2, c1:c2);
    Mloc = valid_mask(r1:r2, c1:c2) & isfinite(Zloc);

    if nnz(Mloc) < min_points
        Z_smooth(iy, ix) = Z(iy, ix);
        continue;
    end

    [YYloc, XXloc] = ndgrid(r1:r2, c1:c2);
    xloc = (XXloc - ix) * dx;
    yloc = (YYloc - iy) * dy;

    xv = xloc(Mloc);
    yv = yloc(Mloc);
    zv = Zloc(Mloc);

    dist2 = (xv / dx).^2 + (yv / dy).^2;
    w_spatial = exp(-0.5 * dist2 / sigma2);

    if use_conf
        Cloc = conf_map(r1:r2, c1:c2);
        w_conf = Cloc(Mloc);
        w = w_spatial .* max(w_conf, 0);
    else
        w = w_spatial;
    end

    P = [ones(size(xv)), xv, yv, xv.^2, xv.*yv, yv.^2];
    W = diag(w);

    beta = (P.' * W * P + ridge * eye(6)) \ (P.' * W * zv);

    % 中心点处 x=0, y=0，因此平滑值即 beta(1)
    Z_smooth(iy, ix) = beta(1);

    if mod(k, progress_step) == 0 || k == num_valid
        fprintf('Local MLS progress: %d / %d (%.1f%%)\n', ...
            k, num_valid, 100 * k / num_valid);
    end
end
end

function metrics = calc_field_metrics(err_field, mask)
v = err_field(mask);
v = v(isfinite(v));

if isempty(v)
    metrics = struct('rmse', nan, 'mae', nan, 'pv', nan, 'mean', nan, 'std', nan, 'count', 0);
    return;
end

metrics = struct();
metrics.rmse = sqrt(mean(v.^2));
metrics.mae = mean(abs(v));
metrics.pv = max(v) - min(v);
metrics.mean = mean(v);
metrics.std = std(v);
metrics.count = numel(v);
end