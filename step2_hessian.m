%% step2_hessian.m
% 从 Step1 生成的 phs_freeform_fit_data_generalized.mat 中读取数据，
% 对两个子孔径分别计算：
% 1) 真值 Hessian（由无噪声真值规则网格局部二次拟合得到）
% 2) Poly + PHS 模型解析 Hessian
% 3) Hessian 派生特征 tau, eta, alpha
% 4) alpha 在非退化区域上的误差统计
%
% 说明：
% - 本版本适配“sub2 已施加随机 6-DOF 扰动”的新版 Step1
% - sub2 的真值 Hessian 不再依赖全局解析坐标，而由其局部真值网格直接估计
% - alpha 误差同时给出：
%   a) full / inner 区域统计
%   b) 非退化方向区域（alpha_valid_mask）统计

clear; clc; close all;

%% ========================= 1. 读取 Step1 数据 =========================
data_name = 'phs_freeform_fit_data_generalized.mat';
if ~isfile(data_name)
    error('找不到 %s，请先运行 Step1。', data_name);
end

S = load(data_name);

required_vars = { ...
    'surface_params', ...
    'model_sub1', 'model_sub2', ...
    'sub1_grid', 'sub2_grid', ...
    'sub1_mask', 'sub2_mask' ...
    };

for k = 1:numel(required_vars)
    if ~isfield(S, required_vars{k})
        error('MAT 文件缺少变量：%s', required_vars{k});
    end
end

surface_params = S.surface_params;
model_sub1 = S.model_sub1;
model_sub2 = S.model_sub2;
sub1_grid = S.sub1_grid;
sub2_grid = S.sub2_grid;
sub1_mask = logical(S.sub1_mask);
sub2_mask = logical(S.sub2_mask);

dx1 = abs(sub1_grid.x_vec(2) - sub1_grid.x_vec(1));
dy1 = abs(sub1_grid.y_vec(2) - sub1_grid.y_vec(1));
dx2 = abs(sub2_grid.x_vec(2) - sub2_grid.x_vec(1));
dy2 = abs(sub2_grid.y_vec(2) - sub2_grid.y_vec(1));

%% ========================= 2. 内缩 mask，降低边界二阶导误差 =========================
erode_steps = 2;
sub1_inner_mask = erode_binary_mask(sub1_mask, erode_steps);
sub2_inner_mask = erode_binary_mask(sub2_mask, erode_steps);

if nnz(sub1_inner_mask) < 100
    warning('Sub1 inner-mask 有效点过少，退回 full-mask。');
    sub1_inner_mask = sub1_mask;
end

if nnz(sub2_inner_mask) < 100
    warning('Sub2 inner-mask 有效点过少，退回 full-mask。');
    sub2_inner_mask = sub2_mask;
end

%% ========================= 3. 真值 Hessian：由局部真值网格二次拟合估计 =========================
quad_radius = 2;   % 使用 5x5 局部窗口
[Hxx_true_1, Hxy_true_1, Hyy_true_1] = estimate_grid_hessian_quadratic( ...
    sub1_grid.Z_true, sub1_mask, dx1, dy1, quad_radius);

[Hxx_true_2, Hxy_true_2, Hyy_true_2] = estimate_grid_hessian_quadratic( ...
    sub2_grid.Z_true, sub2_mask, dx2, dy2, quad_radius);

truth_method_sub1 = 'local_quadratic_fit_on_true_grid';
truth_method_sub2 = 'local_quadratic_fit_on_true_grid';

%% ========================= 4. 拟合 Hessian：Poly + PHS 解析求导 =========================
[Hxx_fit_1, Hxy_fit_1, Hyy_fit_1] = eval_model_hessian(model_sub1, sub1_grid.X, sub1_grid.Y);
[Hxx_fit_2, Hxy_fit_2, Hyy_fit_2] = eval_model_hessian(model_sub2, sub2_grid.X, sub2_grid.Y);

%% ========================= 5. mask 外设为 NaN =========================
[Hxx_true_1, Hxy_true_1, Hyy_true_1] = apply_mask3(Hxx_true_1, Hxy_true_1, Hyy_true_1, sub1_mask);
[Hxx_fit_1,  Hxy_fit_1,  Hyy_fit_1 ] = apply_mask3(Hxx_fit_1,  Hxy_fit_1,  Hyy_fit_1,  sub1_mask);

[Hxx_true_2, Hxy_true_2, Hyy_true_2] = apply_mask3(Hxx_true_2, Hxy_true_2, Hyy_true_2, sub2_mask);
[Hxx_fit_2,  Hxy_fit_2,  Hyy_fit_2 ] = apply_mask3(Hxx_fit_2,  Hxy_fit_2,  Hyy_fit_2,  sub2_mask);

%% ========================= 6. Hessian 误差 =========================
Hxx_err_1 = Hxx_fit_1 - Hxx_true_1;
Hxy_err_1 = Hxy_fit_1 - Hxy_true_1;
Hyy_err_1 = Hyy_fit_1 - Hyy_true_1;

Hxx_err_2 = Hxx_fit_2 - Hxx_true_2;
Hxy_err_2 = Hxy_fit_2 - Hxy_true_2;
Hyy_err_2 = Hyy_fit_2 - Hyy_true_2;

%% ========================= 7. 派生特征场 =========================
[tau_true_1, eta_true_1, alpha_true_1] = hessian_features(Hxx_true_1, Hxy_true_1, Hyy_true_1);
[tau_fit_1,  eta_fit_1,  alpha_fit_1 ] = hessian_features(Hxx_fit_1,  Hxy_fit_1,  Hyy_fit_1);

[tau_true_2, eta_true_2, alpha_true_2] = hessian_features(Hxx_true_2, Hxy_true_2, Hyy_true_2);
[tau_fit_2,  eta_fit_2,  alpha_fit_2 ] = hessian_features(Hxx_fit_2,  Hxy_fit_2,  Hyy_fit_2);

tau_err_1 = tau_fit_1 - tau_true_1;
eta_err_1 = eta_fit_1 - eta_true_1;
alpha_err_1 = wrap_orientation_error(alpha_fit_1 - alpha_true_1);
alpha_err_deg_1 = rad2deg(alpha_err_1);

tau_err_2 = tau_fit_2 - tau_true_2;
eta_err_2 = eta_fit_2 - eta_true_2;
alpha_err_2 = wrap_orientation_error(alpha_fit_2 - alpha_true_2);
alpha_err_deg_2 = rad2deg(alpha_err_2);

%% ========================= 8. alpha 非退化区域掩膜 =========================
% 仅在 eta 足够大时，主方向 alpha 才具有稳定物理意义
eta_ratio = 0.05;

tau_eta_1 = eta_ratio * masked_max_finite(eta_true_1, sub1_inner_mask);
tau_eta_2 = eta_ratio * masked_max_finite(eta_true_2, sub2_inner_mask);

sub1_alpha_valid_mask = sub1_inner_mask & ...
    isfinite(eta_true_1) & isfinite(eta_fit_1) & ...
    (eta_true_1 > tau_eta_1) & (eta_fit_1 > tau_eta_1);

sub2_alpha_valid_mask = sub2_inner_mask & ...
    isfinite(eta_true_2) & isfinite(eta_fit_2) & ...
    (eta_true_2 > tau_eta_2) & (eta_fit_2 > tau_eta_2);

if nnz(sub1_alpha_valid_mask) < 50
    warning('Sub1 alpha 非退化区域点数过少，建议降低 eta_ratio。');
end
if nnz(sub2_alpha_valid_mask) < 50
    warning('Sub2 alpha 非退化区域点数过少，建议降低 eta_ratio。');
end

%% ========================= 9. 误差统计 =========================
metrics = struct();

metrics.sub1.Hxx = calc_masked_metrics(Hxx_err_1, sub1_mask, sub1_inner_mask);
metrics.sub1.Hxy = calc_masked_metrics(Hxy_err_1, sub1_mask, sub1_inner_mask);
metrics.sub1.Hyy = calc_masked_metrics(Hyy_err_1, sub1_mask, sub1_inner_mask);
metrics.sub1.tau = calc_masked_metrics(tau_err_1, sub1_mask, sub1_inner_mask);
metrics.sub1.eta = calc_masked_metrics(eta_err_1, sub1_mask, sub1_inner_mask);

metrics.sub2.Hxx = calc_masked_metrics(Hxx_err_2, sub2_mask, sub2_inner_mask);
metrics.sub2.Hxy = calc_masked_metrics(Hxy_err_2, sub2_mask, sub2_inner_mask);
metrics.sub2.Hyy = calc_masked_metrics(Hyy_err_2, sub2_mask, sub2_inner_mask);
metrics.sub2.tau = calc_masked_metrics(tau_err_2, sub2_mask, sub2_inner_mask);
metrics.sub2.eta = calc_masked_metrics(eta_err_2, sub2_mask, sub2_inner_mask);

% alpha：同时保存 full/inner 统计与非退化区域统计
metrics.sub1.alpha_full = calc_masked_metrics(alpha_err_deg_1, sub1_mask, sub1_inner_mask);
metrics.sub2.alpha_full = calc_masked_metrics(alpha_err_deg_2, sub2_mask, sub2_inner_mask);

metrics.sub1.alpha_valid = calc_metrics_on_mask(alpha_err_deg_1, sub1_alpha_valid_mask);
metrics.sub2.alpha_valid = calc_metrics_on_mask(alpha_err_deg_2, sub2_alpha_valid_mask);

metrics.sub1.alpha_valid_ratio = nnz(sub1_alpha_valid_mask) / max(nnz(sub1_inner_mask), 1);
metrics.sub2.alpha_valid_ratio = nnz(sub2_alpha_valid_mask) / max(nnz(sub2_inner_mask), 1);

metrics.sub1.alpha_eta_threshold = tau_eta_1;
metrics.sub2.alpha_eta_threshold = tau_eta_2;

%% ========================= 10. 终端输出 =========================
disp('================ Hessian 场误差统计（full-mask / inner-mask） ================');
disp(['Sub1 truth method: ', truth_method_sub1]);
disp(['Sub2 truth method: ', truth_method_sub2]);

print_metric_line('Sub1 Hxx', metrics.sub1.Hxx, '');
print_metric_line('Sub1 Hxy', metrics.sub1.Hxy, '');
print_metric_line('Sub1 Hyy', metrics.sub1.Hyy, '');
print_metric_line('Sub1 tau', metrics.sub1.tau, '');
print_metric_line('Sub1 eta', metrics.sub1.eta, '');

fprintf('Sub1 alpha(full)  | full RMSE = %.6e deg, inner RMSE = %.6e deg, full max = %.6e deg, inner max = %.6e deg\n', ...
    metrics.sub1.alpha_full.full.rmse, ...
    metrics.sub1.alpha_full.inner.rmse, ...
    metrics.sub1.alpha_full.full.max_abs, ...
    metrics.sub1.alpha_full.inner.max_abs);

fprintf('Sub1 alpha(valid) | valid ratio = %.4f, valid RMSE = %.6e deg, valid max = %.6e deg\n', ...
    metrics.sub1.alpha_valid_ratio, ...
    metrics.sub1.alpha_valid.rmse, ...
    metrics.sub1.alpha_valid.max_abs);

disp('-------------------------------------------------------------------------');

print_metric_line('Sub2 Hxx', metrics.sub2.Hxx, '');
print_metric_line('Sub2 Hxy', metrics.sub2.Hxy, '');
print_metric_line('Sub2 Hyy', metrics.sub2.Hyy, '');
print_metric_line('Sub2 tau', metrics.sub2.tau, '');
print_metric_line('Sub2 eta', metrics.sub2.eta, '');

fprintf('Sub2 alpha(full)  | full RMSE = %.6e deg, inner RMSE = %.6e deg, full max = %.6e deg, inner max = %.6e deg\n', ...
    metrics.sub2.alpha_full.full.rmse, ...
    metrics.sub2.alpha_full.inner.rmse, ...
    metrics.sub2.alpha_full.full.max_abs, ...
    metrics.sub2.alpha_full.inner.max_abs);

fprintf('Sub2 alpha(valid) | valid ratio = %.4f, valid RMSE = %.6e deg, valid max = %.6e deg\n', ...
    metrics.sub2.alpha_valid_ratio, ...
    metrics.sub2.alpha_valid.rmse, ...
    metrics.sub2.alpha_valid.max_abs);

%% ========================= 11. Hessian 三分量可视化 =========================
field_names_H = {'H_{xx}', 'H_{xy}', 'H_{yy}'};

draw_triplet_figure( ...
    sub1_grid.x_vec, sub1_grid.y_vec, sub1_mask, ...
    {Hxx_true_1, Hxy_true_1, Hyy_true_1}, ...
    {Hxx_fit_1,  Hxy_fit_1,  Hyy_fit_1 }, ...
    {Hxx_err_1,  Hxy_err_1,  Hyy_err_1 }, ...
    field_names_H, ...
    'Sub-aperture 1 Hessian Validation');

draw_triplet_figure( ...
    sub2_grid.x_vec, sub2_grid.y_vec, sub2_mask, ...
    {Hxx_true_2, Hxy_true_2, Hyy_true_2}, ...
    {Hxx_fit_2,  Hxy_fit_2,  Hyy_fit_2 }, ...
    {Hxx_err_2,  Hxy_err_2,  Hyy_err_2 }, ...
    field_names_H, ...
    'Sub-aperture 2 Hessian Validation');

%% ========================= 12. 派生特征场可视化 =========================
% tau / eta 正常显示全 mask
draw_triplet_figure( ...
    sub1_grid.x_vec, sub1_grid.y_vec, sub1_mask, ...
    {tau_true_1, eta_true_1, rad2deg(alpha_true_1)}, ...
    {tau_fit_1,  eta_fit_1,  rad2deg(alpha_fit_1) }, ...
    {tau_err_1,  eta_err_1,  alpha_err_deg_1      }, ...
    {'\tau', '\eta', '\alpha (deg)'}, ...
    'Sub-aperture 1 Feature Validation (full alpha)');

draw_triplet_figure( ...
    sub2_grid.x_vec, sub2_grid.y_vec, sub2_mask, ...
    {tau_true_2, eta_true_2, rad2deg(alpha_true_2)}, ...
    {tau_fit_2,  eta_fit_2,  rad2deg(alpha_fit_2) }, ...
    {tau_err_2,  eta_err_2,  alpha_err_deg_2      }, ...
    {'\tau', '\eta', '\alpha (deg)'}, ...
    'Sub-aperture 2 Feature Validation (full alpha)');

%% ========================= 13. alpha 非退化区域可视化 =========================
alpha_true_deg_1_valid = rad2deg(alpha_true_1);
alpha_fit_deg_1_valid  = rad2deg(alpha_fit_1);
alpha_err_deg_1_valid_plot = alpha_err_deg_1;

alpha_true_deg_2_valid = rad2deg(alpha_true_2);
alpha_fit_deg_2_valid  = rad2deg(alpha_fit_2);
alpha_err_deg_2_valid_plot = alpha_err_deg_2;

alpha_true_deg_1_valid(~sub1_alpha_valid_mask) = nan;
alpha_fit_deg_1_valid(~sub1_alpha_valid_mask)  = nan;
alpha_err_deg_1_valid_plot(~sub1_alpha_valid_mask) = nan;

alpha_true_deg_2_valid(~sub2_alpha_valid_mask) = nan;
alpha_fit_deg_2_valid(~sub2_alpha_valid_mask)  = nan;
alpha_err_deg_2_valid_plot(~sub2_alpha_valid_mask) = nan;

figure('Position', [80, 80, 1350, 520], 'Name', 'Alpha Validation on Non-degenerate Region');

subplot(2,3,1);
imagesc(sub1_grid.x_vec, sub1_grid.y_vec, alpha_true_deg_1_valid);
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('Sub1 \alpha true (valid region)');

subplot(2,3,2);
imagesc(sub1_grid.x_vec, sub1_grid.y_vec, alpha_fit_deg_1_valid);
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('Sub1 \alpha fit (valid region)');

subplot(2,3,3);
imagesc(sub1_grid.x_vec, sub1_grid.y_vec, alpha_err_deg_1_valid_plot);
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('Sub1 \alpha error (valid region)');

subplot(2,3,4);
imagesc(sub2_grid.x_vec, sub2_grid.y_vec, alpha_true_deg_2_valid);
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('Sub2 \alpha true (valid region)');

subplot(2,3,5);
imagesc(sub2_grid.x_vec, sub2_grid.y_vec, alpha_fit_deg_2_valid);
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('Sub2 \alpha fit (valid region)');

subplot(2,3,6);
imagesc(sub2_grid.x_vec, sub2_grid.y_vec, alpha_err_deg_2_valid_plot);
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('Sub2 \alpha error (valid region)');

%% ========================= 14. mask 与 alpha-valid-mask 可视化 =========================
figure('Position', [100, 100, 1000, 700], 'Name', 'Mask and Alpha-valid Mask');

subplot(2,2,1);
imagesc(sub1_grid.x_vec, sub1_grid.y_vec, double(sub1_mask));
hold on;
contour(sub1_grid.x_vec, sub1_grid.y_vec, double(sub1_inner_mask), [0.5 0.5], 'w-', 'LineWidth', 1.2);
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('Sub1 full / inner mask');

subplot(2,2,2);
imagesc(sub2_grid.x_vec, sub2_grid.y_vec, double(sub2_mask));
hold on;
contour(sub2_grid.x_vec, sub2_grid.y_vec, double(sub2_inner_mask), [0.5 0.5], 'w-', 'LineWidth', 1.2);
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('Sub2 full / inner mask');

subplot(2,2,3);
imagesc(sub1_grid.x_vec, sub1_grid.y_vec, double(sub1_alpha_valid_mask));
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('Sub1 alpha-valid mask');

subplot(2,2,4);
imagesc(sub2_grid.x_vec, sub2_grid.y_vec, double(sub2_alpha_valid_mask));
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('Sub2 alpha-valid mask');

%% ========================= 15. 保存 Step2 结果 =========================
save_name = 'hessian_validation_from_generalized_fit.mat';
save(save_name, ...
    'surface_params', ...
    'sub1_grid', 'sub2_grid', ...
    'sub1_mask', 'sub2_mask', ...
    'sub1_inner_mask', 'sub2_inner_mask', ...
    'sub1_alpha_valid_mask', 'sub2_alpha_valid_mask', ...
    'dx1', 'dy1', 'dx2', 'dy2', ...
    'truth_method_sub1', 'truth_method_sub2', ...
    'quad_radius', 'erode_steps', 'eta_ratio', 'tau_eta_1', 'tau_eta_2', ...
    'Hxx_true_1', 'Hxy_true_1', 'Hyy_true_1', ...
    'Hxx_fit_1',  'Hxy_fit_1',  'Hyy_fit_1', ...
    'Hxx_err_1',  'Hxy_err_1',  'Hyy_err_1', ...
    'Hxx_true_2', 'Hxy_true_2', 'Hyy_true_2', ...
    'Hxx_fit_2',  'Hxy_fit_2',  'Hyy_fit_2', ...
    'Hxx_err_2',  'Hxy_err_2',  'Hyy_err_2', ...
    'tau_true_1', 'eta_true_1', 'alpha_true_1', ...
    'tau_fit_1',  'eta_fit_1',  'alpha_fit_1', ...
    'tau_err_1',  'eta_err_1',  'alpha_err_1', 'alpha_err_deg_1', ...
    'tau_true_2', 'eta_true_2', 'alpha_true_2', ...
    'tau_fit_2',  'eta_fit_2',  'alpha_fit_2', ...
    'tau_err_2',  'eta_err_2',  'alpha_err_2', 'alpha_err_deg_2', ...
    'metrics', ...
    '-v7.3');

disp(['已保存 MAT 文件：', save_name]);

%% ========================= 局部函数 =========================

function [A, B, C] = apply_mask3(A, B, C, mask)
A(~mask) = nan;
B(~mask) = nan;
C(~mask) = nan;
end

function mask_in = erode_binary_mask(mask_in, n_step)
mask_in = logical(mask_in);
if n_step <= 0
    return;
end
for k = 1:n_step
    mask_in = (conv2(double(mask_in), ones(3), 'same') == 9) & mask_in;
end
end

function m = calc_masked_metrics(err_field, full_mask, inner_mask)
m = struct();

vf = err_field(full_mask);
vi = err_field(inner_mask);

vf = vf(isfinite(vf));
vi = vi(isfinite(vi));

m.full = calc_basic_metrics(vf);
m.inner = calc_basic_metrics(vi);
end

function m = calc_metrics_on_mask(err_field, mask)
v = err_field(mask);
v = v(isfinite(v));
m = calc_basic_metrics(v);
end

function m = calc_basic_metrics(v)
if isempty(v)
    m.rmse = nan;
    m.max_abs = nan;
    m.mean_abs = nan;
    m.std = nan;
    return;
end

m.rmse = sqrt(mean(v.^2));
m.max_abs = max(abs(v));
m.mean_abs = mean(abs(v));
m.std = std(v);
end

function vmax = masked_max_finite(A, mask)
v = A(mask);
v = v(isfinite(v));
if isempty(v)
    vmax = nan;
else
    vmax = max(v);
end
end

function print_metric_line(name_str, m, unit_str)
if isempty(unit_str)
    fprintf('%-12s | full RMSE = %.6e, inner RMSE = %.6e, full max = %.6e, inner max = %.6e\n', ...
        name_str, m.full.rmse, m.inner.rmse, m.full.max_abs, m.inner.max_abs);
else
    fprintf('%-12s | full RMSE = %.6e %s, inner RMSE = %.6e %s, full max = %.6e %s, inner max = %.6e %s\n', ...
        name_str, ...
        m.full.rmse, unit_str, ...
        m.inner.rmse, unit_str, ...
        m.full.max_abs, unit_str, ...
        m.inner.max_abs, unit_str);
end
end

function [tau, eta, alpha] = hessian_features(Hxx, Hxy, Hyy)
tau = Hxx + Hyy;
eta = sqrt((Hxx - Hyy).^2 + 4 * Hxy.^2);
alpha = 0.5 * atan2(2 * Hxy, Hxx - Hyy);
end

function e = wrap_orientation_error(e)
% alpha 具有 pi 周期，因此使用双角包裹
e = 0.5 * atan2(sin(2 * e), cos(2 * e));
end

function draw_triplet_figure(x_vec, y_vec, mask, fields_true, fields_fit, fields_err, names3, fig_name)
figure('Position', [80, 80, 1350, 780], 'Name', fig_name);

for k = 1:3
    A = fields_true{k};
    B = fields_fit{k};
    C = fields_err{k};

    A(~mask) = nan;
    B(~mask) = nan;
    C(~mask) = nan;

    subplot(3,3,k);
    imagesc(x_vec, y_vec, A);
    set(gca, 'YDir', 'normal');
    axis image;
    colorbar;
    title([names3{k}, ' true']);

    subplot(3,3,k+3);
    imagesc(x_vec, y_vec, B);
    set(gca, 'YDir', 'normal');
    axis image;
    colorbar;
    title([names3{k}, ' fit']);

    subplot(3,3,k+6);
    imagesc(x_vec, y_vec, C);
    set(gca, 'YDir', 'normal');
    axis image;
    colorbar;
    title([names3{k}, ' error']);
end
end

function [Hxx, Hxy, Hyy] = estimate_grid_hessian_quadratic(Z, mask, dx, dy, radius)
% 对每个有效点使用局部二次拟合：
% z = a0 + a1 x + a2 y + a3 x^2 + a4 xy + a5 y^2
% Hessian = [2a3, a4; a4, 2a5]

[Ny, Nx] = size(Z);
Hxx = nan(Ny, Nx);
Hxy = nan(Ny, Nx);
Hyy = nan(Ny, Nx);

min_pts = 10;

for iy = 1:Ny
    for ix = 1:Nx
        if ~mask(iy, ix)
            continue;
        end

        r1 = max(1, iy - radius);
        r2 = min(Ny, iy + radius);
        c1 = max(1, ix - radius);
        c2 = min(Nx, ix + radius);

        Zloc = Z(r1:r2, c1:c2);
        Mloc = mask(r1:r2, c1:c2) & isfinite(Zloc);

        if nnz(Mloc) < min_pts
            continue;
        end

        [yy_loc, xx_loc] = ndgrid(r1:r2, c1:c2);
        x_local = (xx_loc - ix) * dx;
        y_local = (yy_loc - iy) * dy;

        xv = x_local(Mloc);
        yv = y_local(Mloc);
        zv = Zloc(Mloc);

        P = [ones(size(xv)), xv, yv, xv.^2, xv .* yv, yv.^2];
        beta = P \ zv;

        Hxx(iy, ix) = 2 * beta(4);
        Hxy(iy, ix) = beta(5);
        Hyy(iy, ix) = 2 * beta(6);
    end
end
end

function [Hxx, Hxy, Hyy] = eval_model_hessian(model, X, Y)
[Hxx_poly, Hxy_poly, Hyy_poly] = eval_poly_hessian(model.polyModel, X, Y);
[Hxx_phs,  Hxy_phs,  Hyy_phs ] = eval_phs_hessian(model.phsModel, X, Y);

Hxx = Hxx_poly + Hxx_phs;
Hxy = Hxy_poly + Hxy_phs;
Hyy = Hyy_poly + Hyy_phs;
end

function [Hxx, Hxy, Hyy] = eval_poly_hessian(polyModel, X, Y)
sz = size(X);
x = X(:);
y = Y(:);

if polyModel.normalize
    xn = (x - polyModel.x_mu) ./ polyModel.x_s;
    yn = (y - polyModel.y_mu) ./ polyModel.y_s;
    xs = polyModel.x_s;
    ys = polyModel.y_s;
else
    xn = x;
    yn = y;
    xs = 1;
    ys = 1;
end

b = polyModel.beta;

Hxx_n = 2 * b(4) + 6 * b(7) .* xn + 2 * b(8) .* yn;
Hxy_n =     b(5) + 2 * b(8) .* xn + 2 * b(9) .* yn;
Hyy_n = 2 * b(6) + 2 * b(9) .* xn + 6 * b(10) .* yn;

Hxx = reshape(Hxx_n ./ (xs^2), sz);
Hxy = reshape(Hxy_n ./ (xs * ys), sz);
Hyy = reshape(Hyy_n ./ (ys^2), sz);
end

function [Hxx, Hxy, Hyy] = eval_phs_hessian(phsModel, X, Y)
sz = size(X);
x = X(:);
y = Y(:);

if ~phsModel.use_phs || isempty(phsModel.weights)
    Hxx = zeros(sz);
    Hxy = zeros(sz);
    Hyy = zeros(sz);
    return;
end

if phsModel.normalize
    xn = (x - phsModel.x_mu) ./ phsModel.x_s;
    yn = (y - phsModel.y_mu) ./ phsModel.y_s;
    xs = phsModel.x_s;
    ys = phsModel.y_s;
else
    xn = x;
    yn = y;
    xs = 1;
    ys = 1;
end

xc = phsModel.xc(:).';
yc = phsModel.yc(:).';
w  = phsModel.weights(:);

dx = xn - xc;
dy = yn - yc;
s  = dx.^2 + dy.^2;

Hxx_n = zeros(size(s));
Hxy_n = zeros(size(s));
Hyy_n = zeros(size(s));

mask = s > 0;

switch lower(phsModel.kernel)
    case 'r4logr'
        ss = s(mask);
        ddx = dx(mask);
        ddy = dy(mask);

        c1 = 2 * log(ss) + 1;
        c2 = 2 * log(ss) + 3;

        Hxx_n(mask) = ss .* c1 + 2 * (ddx.^2) .* c2;
        Hxy_n(mask) = 2 * ddx .* ddy .* c2;
        Hyy_n(mask) = ss .* c1 + 2 * (ddy.^2) .* c2;

    case 'r6logr'
        ss = s(mask);
        ddx = dx(mask);
        ddy = dy(mask);

        c1 = 3 * log(ss) + 1;
        c2 = 6 * log(ss) + 5;

        Hxx_n(mask) = (ss.^2) .* c1 + 2 * (ddx.^2) .* ss .* c2;
        Hxy_n(mask) = 2 * ddx .* ddy .* ss .* c2;
        Hyy_n(mask) = (ss.^2) .* c1 + 2 * (ddy.^2) .* ss .* c2;

    otherwise
        error('未知 PHS 核函数: %s', phsModel.kernel);
end

Hxx = reshape(Hxx_n * w, sz) ./ (xs^2);
Hxy = reshape(Hxy_n * w, sz) ./ (xs * ys);
Hyy = reshape(Hyy_n * w, sz) ./ (ys^2);
end