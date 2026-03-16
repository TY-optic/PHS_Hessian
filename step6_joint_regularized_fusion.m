%% step6_joint_regularized_fusion.m
% 非必要，，正则化改善不明显
% Step 5:
% 1) 读取 Step1 与 Step4 结果
% 2) 将 sub1 / sub2_final 投影到统一 global 规则网格
% 3) 构造基线离散融合面（加权平均）
% 4) 在统一 global 网格上求解“观测一致性 + 二阶正则化”的联合融合面
% 5) 输出 full-mask / inner-mask 误差统计
%
% 目标函数：
%   min_Z  sum_{Omega1} w1*(Z-Z1)^2 + sum_{Omega2} w2*(Z-Z2)^2
%        + lambda*( ||Dxx Z||^2 + 2||Dxy Z||^2 + ||Dyy Z||^2 )
%
% 说明：
% - 主结果可以直接采用联合正则化融合面，也保留基线离散融合面作对比
% - 默认提供 lambda 扫描；在数值仿真中可用真值选最优 lambda
% - 若在真实实验中使用，应关闭“按真值选 lambda”，转而手动指定 lambda

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

required_s1 = {'X','Y','Z_true','x_vec','y_vec','surface_params'};
required_s4 = {'pc1_global','pc2_final'};

for k = 1:numel(required_s1)
    if ~isfield(S1, required_s1{k})
        error('Step1 MAT 文件缺少变量：%s', required_s1{k});
    end
end
for k = 1:numel(required_s4)
    if ~isfield(S4, required_s4{k})
        error('Step4 MAT 文件缺少变量：%s', required_s4{k});
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

fprintf('================ Step5: joint regularized fusion =================\n');
fprintf('Using Step4 file: %s\n', data_step4);

%% ========================= 2. 参数设置 =========================
opt = struct();

% 投影到统一 global 规则网格
opt.boundary_shrink = 0.95;
opt.interp_method = 'natural';
opt.extrap_method = 'none';

% 可选：重叠区低阶一致性校正
opt.use_overlap_planar_correction = false;
opt.use_overlap_quadratic_correction = false;
opt.min_overlap_corr_points = 500;

% 数据权重
opt.boundary_soft_width_pix = 8;
opt.data_weight_floor = 0.05;   % 避免边界权重精确为 0 导致局部欠定

% 误差统计 inner-mask
opt.inner_erode_steps = 4;

% lambda 扫描（路径测试）
opt.lambda_list = [ ...
    0, ...
    1e-10, 3e-10, ...
    1e-9, 3e-9, ...
    1e-8, 3e-8, ...
    1e-7, 3e-7, ...
    1e-6];
opt.select_lambda_by_truth = true;   % 数值仿真测试可开；真实实验建议 false
opt.lambda_manual = 1e-8;            % 当 select_lambda_by_truth=false 时使用

% 数值稳定
opt.diag_jitter = 1e-12;

%% ========================= 3. 投影到统一 global 网格 =========================
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

%% ========================= 5. 基线离散融合（加权平均） =========================
Wbd1 = build_boundary_weight(valid1, opt.boundary_soft_width_pix);
Wbd2 = build_boundary_weight(valid2_corr, opt.boundary_soft_width_pix);

W1 = max(Wbd1, opt.data_weight_floor) .* double(valid1);
W2 = max(Wbd2, opt.data_weight_floor) .* double(valid2_corr);

Z_fused_baseline = nan(size(Xg));
Wsum = W1 + W2;

only1 = valid1 & ~valid2_corr;
only2 = valid2_corr & ~valid1;
both  = mask_overlap;

Z_fused_baseline(only1) = Z1_global_grid(only1);
Z_fused_baseline(only2) = Z2_global_grid_corr(only2);

idx_both = both & (Wsum > 0);
Z_fused_baseline(idx_both) = ...
    (W1(idx_both).*Z1_global_grid(idx_both) + W2(idx_both).*Z2_global_grid_corr(idx_both)) ./ Wsum(idx_both);

idx_both_zero = both & ~(Wsum > 0);
Z_fused_baseline(idx_both_zero) = 0.5 * (Z1_global_grid(idx_both_zero) + Z2_global_grid_corr(idx_both_zero));

valid_baseline = mask_union & isfinite(Z_fused_baseline);

%% ========================= 6. 基线误差统计 =========================
boundary_dist_map = boundary_distance_layers(mask_union);
inner_mask = erode_binary_mask(mask_union, opt.inner_erode_steps);

err_baseline = Z_fused_baseline - Z_true;
err_baseline(~valid_baseline) = nan;

metrics_baseline_full  = calc_field_metrics(err_baseline, valid_baseline);
metrics_baseline_inner = calc_field_metrics(err_baseline, inner_mask);

fprintf('Baseline discrete fused RMSE (full)  = %.6e mm\n', metrics_baseline_full.rmse);
fprintf('Baseline discrete fused RMSE (inner) = %.6e mm\n', metrics_baseline_inner.rmse);

%% ========================= 7. lambda 扫描：联合正则化融合 =========================
fprintf('---------------- Solving joint regularized fusion ----------------\n');

lambda_list = opt.lambda_list(:);
num_lambda = numel(lambda_list);

scan_result = struct();
scan_result.lambda = lambda_list;
scan_result.rmse_full = nan(num_lambda,1);
scan_result.rmse_inner = nan(num_lambda,1);
scan_result.mae_full = nan(num_lambda,1);
scan_result.std_full = nan(num_lambda,1);
scan_result.pv_full = nan(num_lambda,1);
scan_result.data_residual = nan(num_lambda,1);
scan_result.reg_energy = nan(num_lambda,1);

Z_joint_candidates = cell(num_lambda,1);

for k = 1:num_lambda
    lambda = lambda_list(k);

    [Z_joint_k, solver_info_k] = solve_joint_regularized_fusion( ...
        Z1_global_grid, Z2_global_grid_corr, ...
        W1, W2, valid1, valid2_corr, mask_union, ...
        dx, dy, lambda, opt.diag_jitter);

    Z_joint_candidates{k} = Z_joint_k;

    err_k = Z_joint_k - Z_true;
    err_k(~mask_union) = nan;

    met_full_k  = calc_field_metrics(err_k, mask_union);
    met_inner_k = calc_field_metrics(err_k, inner_mask);

    scan_result.rmse_full(k) = met_full_k.rmse;
    scan_result.rmse_inner(k) = met_inner_k.rmse;
    scan_result.mae_full(k) = met_full_k.mae;
    scan_result.std_full(k) = met_full_k.std;
    scan_result.pv_full(k) = met_full_k.pv;
    scan_result.data_residual(k) = solver_info_k.data_residual;
    scan_result.reg_energy(k) = solver_info_k.reg_energy;

    fprintf('lambda = %-10.3e | RMSE(full) = %.6e | RMSE(inner) = %.6e | data = %.6e | reg = %.6e\n', ...
        lambda, scan_result.rmse_full(k), scan_result.rmse_inner(k), ...
        scan_result.data_residual(k), scan_result.reg_energy(k));
end

%% ========================= 8. 选择最终 lambda =========================
if opt.select_lambda_by_truth
    [~, idx_best] = min(scan_result.rmse_full);
else
    [~, idx_best] = min(abs(lambda_list - opt.lambda_manual));
end

lambda_best = lambda_list(idx_best);
Z_joint_fused = Z_joint_candidates{idx_best};
valid_joint = mask_union & isfinite(Z_joint_fused);

err_joint = Z_joint_fused - Z_true;
err_joint(~valid_joint) = nan;

metrics_joint_full  = calc_field_metrics(err_joint, valid_joint);
metrics_joint_inner = calc_field_metrics(err_joint, inner_mask);

fprintf('---------------- Selected joint fusion result ----------------\n');
fprintf('Selected lambda = %.6e\n', lambda_best);
fprintf('Joint fused RMSE (full)  = %.6e mm\n', metrics_joint_full.rmse);
fprintf('Joint fused RMSE (inner) = %.6e mm\n', metrics_joint_inner.rmse);

%% ========================= 9. 重叠区一致性诊断 =========================
overlap_diff_before = Z1_global_grid - Z2_global_grid;
overlap_diff_after  = Z1_global_grid - Z2_global_grid_corr;
overlap_diff_joint  = Z1_global_grid - Z_joint_fused;

overlap_diff_before(~mask_overlap_raw) = nan;
overlap_diff_after(~mask_overlap) = nan;
overlap_diff_joint(~mask_overlap) = nan;

metrics_overlap_before = calc_field_metrics(overlap_diff_before, mask_overlap_raw);
metrics_overlap_after  = calc_field_metrics(overlap_diff_after,  mask_overlap);
metrics_overlap_joint  = calc_field_metrics(overlap_diff_joint,  mask_overlap);

fprintf('Overlap difference before correction RMSE = %.6e mm\n', metrics_overlap_before.rmse);
fprintf('Overlap difference after  correction RMSE = %.6e mm\n', metrics_overlap_after.rmse);
fprintf('Overlap difference sub1-to-joint   RMSE   = %.6e mm\n', metrics_overlap_joint.rmse);

%% ========================= 10. 可视化 =========================
figure('Position', [30, 40, 1780, 980], 'Name', 'Step5 Joint Regularized Fusion', 'Color', 'w');

subplot(2,3,1);
imagesc(x_vec, y_vec, Z1_global_grid);
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('1. Sub1 projected surface');
xlabel('Global X (mm)'); ylabel('Global Y (mm)');

subplot(2,3,2);
imagesc(x_vec, y_vec, Z2_global_grid_corr);
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('2. Sub2 final projected surface');
xlabel('Global X (mm)'); ylabel('Global Y (mm)');

subplot(2,3,3);
imagesc(x_vec, y_vec, Z_joint_fused);
set(gca, 'YDir', 'normal'); axis image; colorbar;
title(sprintf('3. Joint regularized fused surface\n\\lambda = %.1e', lambda_best));
xlabel('Global X (mm)'); ylabel('Global Y (mm)');

subplot(2,3,4);
imagesc(x_vec, y_vec, err_baseline);
set(gca, 'YDir', 'normal'); axis image; colorbar;
title(sprintf('4. Baseline fused error\nFull = %.2e, Inner = %.2e mm', ...
    metrics_baseline_full.rmse, metrics_baseline_inner.rmse));
xlabel('Global X (mm)'); ylabel('Global Y (mm)');

subplot(2,3,5);
imagesc(x_vec, y_vec, err_joint);
set(gca, 'YDir', 'normal'); axis image; colorbar;
title(sprintf('5. Joint fused error\nFull = %.2e, Inner = %.2e mm', ...
    metrics_joint_full.rmse, metrics_joint_inner.rmse));
xlabel('Global X (mm)'); ylabel('Global Y (mm)');

subplot(2,3,6);
err_hist = err_joint(valid_joint);
err_hist = err_hist(isfinite(err_hist));
if numel(err_hist) > 20
    histogram(err_hist, 60, 'Normalization', 'pdf');
    xlabel('Residual (mm)');
    ylabel('PDF');
    title('6. Joint fused residual histogram');
    grid on;
else
    text(0.5, 0.5, 'Insufficient valid residuals', 'HorizontalAlignment', 'center');
    axis off;
end

%% ========================= 11. lambda 曲线 =========================
figure('Position', [90, 90, 1400, 420], 'Name', 'Lambda Scan Curves', 'Color', 'w');

subplot(1,3,1);
semilogx(scan_result.lambda, scan_result.rmse_full, 'o-', 'LineWidth', 1.2); hold on;
semilogx(lambda_best, metrics_joint_full.rmse, 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
grid on;
xlabel('\lambda'); ylabel('RMSE (full)');
title('RMSE(full) vs \lambda');

subplot(1,3,2);
semilogx(scan_result.lambda, scan_result.rmse_inner, 'o-', 'LineWidth', 1.2); hold on;
semilogx(lambda_best, metrics_joint_inner.rmse, 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
grid on;
xlabel('\lambda'); ylabel('RMSE (inner)');
title('RMSE(inner) vs \lambda');

subplot(1,3,3);
yyaxis left;
semilogx(scan_result.lambda, scan_result.data_residual, 'o-', 'LineWidth', 1.2);
ylabel('Data residual');
yyaxis right;
semilogx(scan_result.lambda, scan_result.reg_energy, 's-', 'LineWidth', 1.2);
ylabel('Regularization energy');
grid on;
xlabel('\lambda');
title('Data-fit / regularization tradeoff');

%% ========================= 12. 边缘与内部诊断 =========================
figure('Position', [70, 80, 1600, 520], 'Name', 'Boundary / Inner Diagnostics', 'Color', 'w');

subplot(1,4,1);
imagesc(x_vec, y_vec, double(mask_union));
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('1. Union support');
xlabel('Global X (mm)'); ylabel('Global Y (mm)');

subplot(1,4,2);
imagesc(x_vec, y_vec, boundary_dist_map);
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('2. Boundary distance layers');
xlabel('Global X (mm)'); ylabel('Global Y (mm)');

subplot(1,4,3);
imagesc(x_vec, y_vec, double(inner_mask));
set(gca, 'YDir', 'normal'); axis image; colorbar;
title(sprintf('3. Inner mask (erode %d px)', opt.inner_erode_steps));
xlabel('Global X (mm)'); ylabel('Global Y (mm)');

subplot(1,4,4);
edge_only_err = err_joint;
edge_only_err(inner_mask) = nan;
imagesc(x_vec, y_vec, edge_only_err);
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('4. Joint-fusion error on edge region');
xlabel('Global X (mm)'); ylabel('Global Y (mm)');

%% ========================= 13. 输出最终结果 =========================
fprintf('\n================ Final global error result =================\n');
fprintf('Baseline discrete fused surface:\n');
fprintf('  Full  RMSE = %.6e mm\n', metrics_baseline_full.rmse);
fprintf('  Inner RMSE = %.6e mm\n', metrics_baseline_inner.rmse);
fprintf('  MAE        = %.6e mm\n', metrics_baseline_full.mae);
fprintf('  PV         = %.6e mm\n', metrics_baseline_full.pv);
fprintf('  STD        = %.6e mm\n', metrics_baseline_full.std);

fprintf('Joint regularized fused surface:\n');
fprintf('  lambda     = %.6e\n', lambda_best);
fprintf('  Full  RMSE = %.6e mm\n', metrics_joint_full.rmse);
fprintf('  Inner RMSE = %.6e mm\n', metrics_joint_inner.rmse);
fprintf('  MAE        = %.6e mm\n', metrics_joint_full.mae);
fprintf('  PV         = %.6e mm\n', metrics_joint_full.pv);
fprintf('  STD        = %.6e mm\n', metrics_joint_full.std);

%% ========================= 14. 保存结果 =========================
result_step5 = struct();

result_step5.metrics.baseline_full = metrics_baseline_full;
result_step5.metrics.baseline_inner = metrics_baseline_inner;
result_step5.metrics.joint_full = metrics_joint_full;
result_step5.metrics.joint_inner = metrics_joint_inner;

result_step5.metrics.overlap_before = metrics_overlap_before;
result_step5.metrics.overlap_after  = metrics_overlap_after;
result_step5.metrics.overlap_joint  = metrics_overlap_joint;

result_step5.scan = scan_result;
result_step5.lambda_best = lambda_best;

result_step5.support.sub1_count = nnz(valid1);
result_step5.support.sub2_count = nnz(valid2_corr);
result_step5.support.overlap_count = nnz(mask_overlap);
result_step5.support.union_count = nnz(valid_joint);

result_step5.correction = corr_model;
result_step5.options = opt;

save_name = 'step5_joint_regularized_fusion_results.mat';
save(save_name, ...
    'result_step5', ...
    'Z1_global_grid', 'Z2_global_grid', 'Z2_global_grid_corr', ...
    'mask1_grid', 'mask2_grid', ...
    'W1', 'W2', ...
    'mask_overlap', 'mask_union', ...
    'boundary_dist_map', 'inner_mask', ...
    'Z_fused_baseline', 'err_baseline', ...
    'Z_joint_fused', 'err_joint', ...
    'overlap_diff_before', 'overlap_diff_after', 'overlap_diff_joint', ...
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
        error('仅支持 1 阶或 2 阶校正。');
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
        error('仅支持 1 阶或 2 阶校正。');
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

function [Z_joint, info] = solve_joint_regularized_fusion( ...
    Z1, Z2, W1, W2, valid1, valid2, mask_union, dx, dy, lambda, diag_jitter)

[Ny, Nx] = size(Z1);

id_map = zeros(Ny, Nx);
id_map(mask_union) = 1:nnz(mask_union);
N = nnz(mask_union);

% ---------------- 数据项：对角阵 ----------------
w_data = zeros(N,1);
rhs = zeros(N,1);

idx1 = id_map(valid1);
z1v = Z1(valid1);
w1v = W1(valid1);

w_data(idx1) = w_data(idx1) + w1v;
rhs(idx1) = rhs(idx1) + w1v .* z1v;

idx2 = id_map(valid2);
z2v = Z2(valid2);
w2v = W2(valid2);

w_data(idx2) = w_data(idx2) + w2v;
rhs(idx2) = rhs(idx2) + w2v .* z2v;

H_data = spdiags(w_data + diag_jitter, 0, N, N);

% ---------------- 正则项矩阵 ----------------
[Dxx, Dyy, Dxy] = build_second_derivative_operators(mask_union, id_map, dx, dy);

if lambda > 0
    Areg = [sqrt(lambda) * Dxx; sqrt(lambda) * Dyy; sqrt(2*lambda) * Dxy];
    H_reg = Areg' * Areg;
else
    Areg = sparse(0, N);
    H_reg = sparse(N, N);
end

% ---------------- 求解 ----------------
H = H_data + H_reg;
z_vec = H \ rhs;

Z_joint = nan(Ny, Nx);
Z_joint(mask_union) = z_vec;

% ---------------- 诊断 ----------------
data_residual = 0;
if ~isempty(idx1)
    data_residual = data_residual + sum(w1v .* (z_vec(idx1) - z1v).^2);
end
if ~isempty(idx2)
    data_residual = data_residual + sum(w2v .* (z_vec(idx2) - z2v).^2);
end

reg_energy = 0;
if lambda > 0
    reg_energy = norm(Dxx * z_vec)^2 + norm(Dyy * z_vec)^2 + 2 * norm(Dxy * z_vec)^2;
end

info = struct();
info.data_residual = data_residual;
info.reg_energy = reg_energy;
info.num_unknown = N;
info.num_dxx = size(Dxx,1);
info.num_dyy = size(Dyy,1);
info.num_dxy = size(Dxy,1);
end

function [Dxx, Dyy, Dxy] = build_second_derivative_operators(mask_union, id_map, dx, dy)
[Ny, Nx] = size(mask_union);

% 预分配近似上界
max_rows = nnz(mask_union);
rows_xx = zeros(3*max_rows, 1);
cols_xx = zeros(3*max_rows, 1);
vals_xx = zeros(3*max_rows, 1);
row_count_xx = 0;
nnz_count_xx = 0;

rows_yy = zeros(3*max_rows, 1);
cols_yy = zeros(3*max_rows, 1);
vals_yy = zeros(3*max_rows, 1);
row_count_yy = 0;
nnz_count_yy = 0;

rows_xy = zeros(4*max_rows, 1);
cols_xy = zeros(4*max_rows, 1);
vals_xy = zeros(4*max_rows, 1);
row_count_xy = 0;
nnz_count_xy = 0;

for iy = 2:Ny-1
    for ix = 2:Nx-1
        if ~mask_union(iy, ix)
            continue;
        end

        % Dxx
        if mask_union(iy, ix-1) && mask_union(iy, ix) && mask_union(iy, ix+1)
            row_count_xx = row_count_xx + 1;

            c1 = id_map(iy, ix-1);
            c2 = id_map(iy, ix);
            c3 = id_map(iy, ix+1);

            base = nnz_count_xx;
            rows_xx(base+1:base+3) = row_count_xx;
            cols_xx(base+1:base+3) = [c1; c2; c3];
            vals_xx(base+1:base+3) = [1; -2; 1] / (dx^2);
            nnz_count_xx = nnz_count_xx + 3;
        end

        % Dyy
        if mask_union(iy-1, ix) && mask_union(iy, ix) && mask_union(iy+1, ix)
            row_count_yy = row_count_yy + 1;

            c1 = id_map(iy-1, ix);
            c2 = id_map(iy, ix);
            c3 = id_map(iy+1, ix);

            base = nnz_count_yy;
            rows_yy(base+1:base+3) = row_count_yy;
            cols_yy(base+1:base+3) = [c1; c2; c3];
            vals_yy(base+1:base+3) = [1; -2; 1] / (dy^2);
            nnz_count_yy = nnz_count_yy + 3;
        end

        % Dxy
        if mask_union(iy-1, ix-1) && mask_union(iy-1, ix+1) && ...
           mask_union(iy+1, ix-1) && mask_union(iy+1, ix+1)

            row_count_xy = row_count_xy + 1;

            c11 = id_map(iy-1, ix-1);
            c13 = id_map(iy-1, ix+1);
            c31 = id_map(iy+1, ix-1);
            c33 = id_map(iy+1, ix+1);

            base = nnz_count_xy;
            rows_xy(base+1:base+4) = row_count_xy;
            cols_xy(base+1:base+4) = [c33; c31; c13; c11];
            vals_xy(base+1:base+4) = [1; -1; -1; 1] / (4 * dx * dy);
            nnz_count_xy = nnz_count_xy + 4;
        end
    end
end

Dxx = sparse(rows_xx(1:nnz_count_xx), cols_xx(1:nnz_count_xx), vals_xx(1:nnz_count_xx), row_count_xx, nnz(mask_union));
Dyy = sparse(rows_yy(1:nnz_count_yy), cols_yy(1:nnz_count_yy), vals_yy(1:nnz_count_yy), row_count_yy, nnz(mask_union));
Dxy = sparse(rows_xy(1:nnz_count_xy), cols_xy(1:nnz_count_xy), vals_xy(1:nnz_count_xy), row_count_xy, nnz(mask_union));
end