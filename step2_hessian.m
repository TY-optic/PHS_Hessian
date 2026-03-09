%% run_hessian_validation_from_generalized_fit.m
% 从 phs_freeform_fit_data_generalized.mat 计算 Hessian 场及其误差
% 当前目标：
% 1) 不做拼接，只验证 Hessian 场恢复质量
% 2) 分别计算 Hxx, Hxy, Hyy 的真值、拟合值与误差
% 3) 进一步计算 tau, eta, alpha 三个派生特征场
% 4) 同时给出 full-mask 与 inner-mask 两套误差统计
% 5) 保存 MAT 文件，供下一步粗配准实验使用

clear; clc; close all;

%% 1. 读取上一阶段结果
data_name = 'phs_freeform_fit_data_generalized.mat';
S = load(data_name);

required_vars = {'surface_params','model_sub1','model_sub2','sub1_grid','sub2_grid','sub1_mask','sub2_mask'};
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
sub1_mask = S.sub1_mask;
sub2_mask = S.sub2_mask;

%% 2. 构造 inner-mask（边界内缩，避免边界处二阶导数误差主导统计）
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

%% 3. 计算真值 Hessian
[Hxx_true_1, Hxy_true_1, Hyy_true_1] = surface_true_hessian(sub1_grid.X, sub1_grid.Y, surface_params);
[Hxx_true_2, Hxy_true_2, Hyy_true_2] = surface_true_hessian(sub2_grid.X, sub2_grid.Y, surface_params);

%% 4. 计算拟合 Hessian
[Hxx_fit_1, Hxy_fit_1, Hyy_fit_1] = eval_model_hessian(model_sub1, sub1_grid.X, sub1_grid.Y);
[Hxx_fit_2, Hxy_fit_2, Hyy_fit_2] = eval_model_hessian(model_sub2, sub2_grid.X, sub2_grid.Y);

%% 5. 仅保留 mask 内部
[Hxx_true_1, Hxy_true_1, Hyy_true_1] = apply_mask3(Hxx_true_1, Hxy_true_1, Hyy_true_1, sub1_mask);
[Hxx_fit_1,  Hxy_fit_1,  Hyy_fit_1 ] = apply_mask3(Hxx_fit_1,  Hxy_fit_1,  Hyy_fit_1,  sub1_mask);

[Hxx_true_2, Hxy_true_2, Hyy_true_2] = apply_mask3(Hxx_true_2, Hxy_true_2, Hyy_true_2, sub2_mask);
[Hxx_fit_2,  Hxy_fit_2,  Hyy_fit_2 ] = apply_mask3(Hxx_fit_2,  Hxy_fit_2,  Hyy_fit_2,  sub2_mask);

%% 6. Hessian 误差
Hxx_err_1 = Hxx_fit_1 - Hxx_true_1;
Hxy_err_1 = Hxy_fit_1 - Hxy_true_1;
Hyy_err_1 = Hyy_fit_1 - Hyy_true_1;

Hxx_err_2 = Hxx_fit_2 - Hxx_true_2;
Hxy_err_2 = Hxy_fit_2 - Hxy_true_2;
Hyy_err_2 = Hyy_fit_2 - Hyy_true_2;

%% 7. 计算 Hessian 派生特征场
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

%% 8. 误差统计
metrics = struct();

metrics.sub1.Hxx = calc_masked_metrics(Hxx_err_1, sub1_mask, sub1_inner_mask);
metrics.sub1.Hxy = calc_masked_metrics(Hxy_err_1, sub1_mask, sub1_inner_mask);
metrics.sub1.Hyy = calc_masked_metrics(Hyy_err_1, sub1_mask, sub1_inner_mask);

metrics.sub1.tau   = calc_masked_metrics(tau_err_1, sub1_mask, sub1_inner_mask);
metrics.sub1.eta   = calc_masked_metrics(eta_err_1, sub1_mask, sub1_inner_mask);
metrics.sub1.alpha = calc_masked_metrics(alpha_err_deg_1, sub1_mask, sub1_inner_mask);

metrics.sub2.Hxx = calc_masked_metrics(Hxx_err_2, sub2_mask, sub2_inner_mask);
metrics.sub2.Hxy = calc_masked_metrics(Hxy_err_2, sub2_mask, sub2_inner_mask);
metrics.sub2.Hyy = calc_masked_metrics(Hyy_err_2, sub2_mask, sub2_inner_mask);

metrics.sub2.tau   = calc_masked_metrics(tau_err_2, sub2_mask, sub2_inner_mask);
metrics.sub2.eta   = calc_masked_metrics(eta_err_2, sub2_mask, sub2_inner_mask);
metrics.sub2.alpha = calc_masked_metrics(alpha_err_deg_2, sub2_mask, sub2_inner_mask);

%% 9. 打印结果
disp('================ Hessian 场误差统计（full-mask / inner-mask） ================');

print_metric_line('Sub1 Hxx', metrics.sub1.Hxx, '');
print_metric_line('Sub1 Hxy', metrics.sub1.Hxy, '');
print_metric_line('Sub1 Hyy', metrics.sub1.Hyy, '');
print_metric_line('Sub1 tau', metrics.sub1.tau, '');
print_metric_line('Sub1 eta', metrics.sub1.eta, '');
print_metric_line('Sub1 alpha', metrics.sub1.alpha, 'deg');

disp('-------------------------------------------------------------------------');

print_metric_line('Sub2 Hxx', metrics.sub2.Hxx, '');
print_metric_line('Sub2 Hxy', metrics.sub2.Hxy, '');
print_metric_line('Sub2 Hyy', metrics.sub2.Hyy, '');
print_metric_line('Sub2 tau', metrics.sub2.tau, '');
print_metric_line('Sub2 eta', metrics.sub2.eta, '');
print_metric_line('Sub2 alpha', metrics.sub2.alpha, 'deg');

%% 10. Hessian 组件可视化
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

%% 11. 派生特征场可视化
field_names_F = {'\tau', '\eta', '\alpha (deg)'};

draw_triplet_figure( ...
    sub1_grid.x_vec, sub1_grid.y_vec, sub1_mask, ...
    {tau_true_1, eta_true_1, rad2deg(alpha_true_1)}, ...
    {tau_fit_1,  eta_fit_1,  rad2deg(alpha_fit_1) }, ...
    {tau_err_1,  eta_err_1,  alpha_err_deg_1      }, ...
    field_names_F, ...
    'Sub-aperture 1 Feature Validation');

draw_triplet_figure( ...
    sub2_grid.x_vec, sub2_grid.y_vec, sub2_mask, ...
    {tau_true_2, eta_true_2, rad2deg(alpha_true_2)}, ...
    {tau_fit_2,  eta_fit_2,  rad2deg(alpha_fit_2) }, ...
    {tau_err_2,  eta_err_2,  alpha_err_deg_2      }, ...
    field_names_F, ...
    'Sub-aperture 2 Feature Validation');

%% 12. inner-mask 可视化
figure('Position', [120, 120, 900, 380], 'Name', 'Full Mask and Inner Mask');
subplot(1,2,1);
imagesc(sub1_grid.x_vec, sub1_grid.y_vec, double(sub1_mask)); hold on;
contour(sub1_grid.x_vec, sub1_grid.y_vec, double(sub1_inner_mask), [0.5 0.5], 'w-', 'LineWidth', 1.2);
set(gca, 'YDir', 'normal');
title('Sub1 Full Mask + Inner Mask');
xlabel('X'); ylabel('Y');
axis image; colorbar;

subplot(1,2,2);
imagesc(sub2_grid.x_vec, sub2_grid.y_vec, double(sub2_mask)); hold on;
contour(sub2_grid.x_vec, sub2_grid.y_vec, double(sub2_inner_mask), [0.5 0.5], 'w-', 'LineWidth', 1.2);
set(gca, 'YDir', 'normal');
title('Sub2 Full Mask + Inner Mask');
xlabel('X'); ylabel('Y');
axis image; colorbar;

%% 13. 保存 MAT 文件
save_name = 'hessian_validation_from_generalized_fit.mat';

save(save_name, ...
    'surface_params', ...
    'sub1_grid', 'sub2_grid', ...
    'sub1_mask', 'sub2_mask', ...
    'sub1_inner_mask', 'sub2_inner_mask', ...
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

    ef = err_field(full_mask);
    ei = err_field(inner_mask);

    ef = ef(isfinite(ef));
    ei = ei(isfinite(ei));

    m.full.rmse = sqrt(mean(ef.^2));
    m.full.max_abs = max(abs(ef));
    m.full.mean_abs = mean(abs(ef));
    m.full.std = std(ef);

    m.inner.rmse = sqrt(mean(ei.^2));
    m.inner.max_abs = max(abs(ei));
    m.inner.mean_abs = mean(abs(ei));
    m.inner.std = std(ei);
end

function print_metric_line(name_str, m, unit_str)
    if isempty(unit_str)
        fprintf('%-12s | full RMSE = %.6e, inner RMSE = %.6e, full max = %.6e, inner max = %.6e\n', ...
            name_str, m.full.rmse, m.inner.rmse, m.full.max_abs, m.inner.max_abs);
    else
        fprintf('%-12s | full RMSE = %.6e %s, inner RMSE = %.6e %s, full max = %.6e %s, inner max = %.6e %s\n', ...
            name_str, m.full.rmse, unit_str, m.inner.rmse, unit_str, m.full.max_abs, unit_str, m.inner.max_abs, unit_str);
    end
end

function draw_triplet_figure(x_vec, y_vec, mask, true_list, fit_list, err_list, field_names, fig_name)
    figure('Position', [60, 60, 1500, 900], 'Name', fig_name);

    nfield = numel(field_names);

    for k = 1:nfield
        T = true_list{k};
        F = fit_list{k};
        E = err_list{k};

        subplot(nfield, 3, (k-1)*3 + 1);
        imagesc(x_vec, y_vec, T);
        set(gca, 'YDir', 'normal');
        axis image;
        colorbar;
        title([field_names{k}, ' True']);
        xlabel('X'); ylabel('Y');

        subplot(nfield, 3, (k-1)*3 + 2);
        imagesc(x_vec, y_vec, F);
        set(gca, 'YDir', 'normal');
        axis image;
        colorbar;
        title([field_names{k}, ' Fit']);
        xlabel('X'); ylabel('Y');

        subplot(nfield, 3, (k-1)*3 + 3);
        imagesc(x_vec, y_vec, E);
        set(gca, 'YDir', 'normal');
        axis image;
        colorbar;
        title([field_names{k}, ' Error']);
        xlabel('X'); ylabel('Y');

        % 将 error 图的颜色范围设为对称
        ee = E(mask);
        ee = ee(isfinite(ee));
        if ~isempty(ee)
            lim = max(abs(ee));
            if lim > 0
                caxis([-lim, lim]);
            end
        end
    end
end

function [tau, eta, alpha] = hessian_features(Hxx, Hxy, Hyy)
    tau = Hxx + Hyy;
    eta = sqrt((Hxx - Hyy).^2 + 4 * Hxy.^2);
    alpha = 0.5 * atan2(2 * Hxy, Hxx - Hyy);
end

function d = wrap_orientation_error(d)
    d = 0.5 * angle(exp(1i * 2 * d));
end

function [Hxx, Hxy, Hyy] = eval_model_hessian(model, Xq, Yq)
    [Hxx_poly, Hxy_poly, Hyy_poly] = eval_poly_hessian(model.polyModel, Xq, Yq);
    [Hxx_phs,  Hxy_phs,  Hyy_phs ] = eval_phs_hessian(model.phsModel, Xq, Yq);

    Hxx = Hxx_poly + Hxx_phs;
    Hxy = Hxy_poly + Hxy_phs;
    Hyy = Hyy_poly + Hyy_phs;
end

function [Hxx, Hxy, Hyy] = eval_poly_hessian(polyModel, Xq, Yq)
    sz = size(Xq);
    x = Xq(:);
    y = Yq(:);

    if polyModel.normalize
        xn = (x - polyModel.center_xy(1)) ./ polyModel.scale_xy(1);
        yn = (y - polyModel.center_xy(2)) ./ polyModel.scale_xy(2);
        sx = polyModel.scale_xy(1);
        sy = polyModel.scale_xy(2);
    else
        xn = x;
        yn = y;
        sx = 1;
        sy = 1;
    end

    P_duu = poly_basis_cubic_duu(xn, yn);
    P_duv = poly_basis_cubic_duv(xn, yn);
    P_dvv = poly_basis_cubic_dvv(xn, yn);

    Hxx_n = P_duu * polyModel.coef;
    Hxy_n = P_duv * polyModel.coef;
    Hyy_n = P_dvv * polyModel.coef;

    Hxx = reshape(Hxx_n ./ (sx^2), sz);
    Hxy = reshape(Hxy_n ./ (sx * sy), sz);
    Hyy = reshape(Hyy_n ./ (sy^2), sz);
end

function [Hxx, Hxy, Hyy] = eval_phs_hessian(phsModel, Xq, Yq)
    sz = size(Xq);
    x = Xq(:);
    y = Yq(:);

    if isempty(phsModel.omega)
        Hxx = zeros(sz);
        Hxy = zeros(sz);
        Hyy = zeros(sz);
        return;
    end

    if phsModel.normalize
        xn = (x - phsModel.center_xy(1)) ./ phsModel.scale_xy(1);
        yn = (y - phsModel.center_xy(2)) ./ phsModel.scale_xy(2);
        sx = phsModel.scale_xy(1);
        sy = phsModel.scale_xy(2);
    else
        xn = x;
        yn = y;
        sx = 1;
        sy = 1;
    end

    xc = phsModel.centers_n(:,1).';
    yc = phsModel.centers_n(:,2).';

    dx = xn - xc;
    dy = yn - yc;
    s = dx.^2 + dy.^2;

    Phi_xx = zeros(size(s));
    Phi_xy = zeros(size(s));
    Phi_yy = zeros(size(s));

    mask = s > 0;
    sm = s(mask);
    dxm = dx(mask);
    dym = dy(mask);
    lsm = log(sm);

    % r^6*ln(r) 
    % Phi_xx(mask) = sm.^2 .* (3 .* lsm + 1) + 2 .* dxm.^2 .* sm .* (6 .* lsm + 5);
    % Phi_xy(mask) = 2 .* dxm .* dym .* sm .* (6 .* lsm + 5);
    % Phi_yy(mask) = sm.^2 .* (3 .* lsm + 1) + 2 .* dym.^2 .* sm .* (6 .* lsm + 5);

    % r^4*ln(r) 
    Phi_xx(mask) = sm .* (2 .* lsm + 1) + 2 .* dxm.^2 .* (2 .* lsm + 3);
    Phi_xy(mask) = 2 .* dxm .* dym .* (2 .* lsm + 3);
    Phi_yy(mask) = sm .* (2 .* lsm + 1) + 2 .* dym.^2 .* (2 .* lsm + 3);

    Hxx_n = Phi_xx * phsModel.omega;
    Hxy_n = Phi_xy * phsModel.omega;
    Hyy_n = Phi_yy * phsModel.omega;

    Hxx = reshape(Hxx_n ./ (sx^2), sz);
    Hxy = reshape(Hxy_n ./ (sx * sy), sz);
    Hyy = reshape(Hyy_n ./ (sy^2), sz);
end

function [Hxx, Hxy, Hyy] = surface_true_hessian(X, Y, p)
    % 1) 三次多项式主形貌
    Hxx_cubic = 6 * p.c1 .* X + 2 * p.c2 .* Y + 2 * p.c5;
    Hxy_cubic = 2 * p.c2 .* X + 2 * p.c3 .* Y;
    Hyy_cubic = 2 * p.c3 .* X + 6 * p.c4 .* Y + 2 * p.c6;

    % 2) 三个高斯包残差
    [G1_xx, G1_xy, G1_yy] = gaussian_hessian(X, Y, p.bump1_amp, p.bump1_x0, p.bump1_y0, p.bump1_sx, p.bump1_sy);
    [G2_xx, G2_xy, G2_yy] = gaussian_hessian(X, Y, p.bump2_amp, p.bump2_x0, p.bump2_y0, p.bump2_sx, p.bump2_sy);
    [G3_xx, G3_xy, G3_yy] = gaussian_hessian(X, Y, p.bump3_amp, p.bump3_x0, p.bump3_y0, p.bump3_sx, p.bump3_sy);

    % 3) 正弦波纹项
    kx = 2 * pi / p.ripple_Lx;
    ky = 2 * pi / p.ripple_Ly;
    s1 = kx .* X + p.ripple_phi;
    t1 = ky .* Y;

    R1_xx = -p.ripple_amp .* (kx^2) .* sin(s1) .* cos(t1);
    R1_xy = -p.ripple_amp .* (kx * ky) .* cos(s1) .* sin(t1);
    R1_yy = -p.ripple_amp .* (ky^2) .* sin(s1) .* cos(t1);

    % 4) 余弦调制项
    px = 2 * pi / p.cos_Lx;
    py = 0.4 * 2 * pi / p.cos_Lx;
    qx = -0.2 * 2 * pi / p.cos_Ly;
    qy = 2 * pi / p.cos_Ly;

    P1 = 2 * pi * (X + 0.4 * Y) / p.cos_Lx;
    Q1 = 2 * pi * (Y - 0.2 * X) / p.cos_Ly;

    C = cos(P1) .* cos(Q1);
    S = sin(P1) .* sin(Q1);

    R2_xx = p.cos_amp .* (-(px^2 + qx^2) .* C + 2 * px * qx .* S);
    R2_xy = p.cos_amp .* (-(px * py + qx * qy) .* C + (px * qy + qx * py) .* S);
    R2_yy = p.cos_amp .* (-(py^2 + qy^2) .* C + 2 * py * qy .* S);

    Hxx = Hxx_cubic + G1_xx + G2_xx + G3_xx + R1_xx + R2_xx;
    Hxy = Hxy_cubic + G1_xy + G2_xy + G3_xy + R1_xy + R2_xy;
    Hyy = Hyy_cubic + G1_yy + G2_yy + G3_yy + R1_yy + R2_yy;
end

function [Gxx, Gxy, Gyy] = gaussian_hessian(X, Y, amp, x0, y0, sx, sy)
    dx = X - x0;
    dy = Y - y0;

    G = amp .* exp(-(dx.^2 ./ (2 * sx^2) + dy.^2 ./ (2 * sy^2)));

    Gxx = G .* (dx.^2 ./ sx^4 - 1 ./ sx^2);
    Gxy = G .* (dx .* dy ./ (sx^2 * sy^2));
    Gyy = G .* (dy.^2 ./ sy^4 - 1 ./ sy^2);
end

function P = poly_basis_cubic_duu(x, y)
    x = x(:);
    y = y(:);

    P = [ ...
        zeros(size(x)), ...
        zeros(size(x)), ...
        zeros(size(x)), ...
        2 * ones(size(x)), ...
        zeros(size(x)), ...
        zeros(size(x)), ...
        6 * x, ...
        2 * y, ...
        zeros(size(x)), ...
        zeros(size(x))];
end

function P = poly_basis_cubic_duv(x, y)
    x = x(:);
    y = y(:);

    P = [ ...
        zeros(size(x)), ...
        zeros(size(x)), ...
        zeros(size(x)), ...
        zeros(size(x)), ...
        ones(size(x)), ...
        zeros(size(x)), ...
        zeros(size(x)), ...
        2 * x, ...
        2 * y, ...
        zeros(size(x))];
end

function P = poly_basis_cubic_dvv(x, y)
    x = x(:);
    y = y(:);

    P = [ ...
        zeros(size(x)), ...
        zeros(size(x)), ...
        zeros(size(x)), ...
        zeros(size(x)), ...
        zeros(size(x)), ...
        2 * ones(size(x)), ...
        zeros(size(x)), ...
        zeros(size(x)), ...
        2 * x, ...
        6 * y];
end