%% step3_fft_registration.m
% 基于 Step2 输出的 Hessian 场，执行二维粗配准：
% 1) 角度离散搜索 theta
% 2) 在每个 theta 下，将 sub2 的 Hessian 张量旋转到候选方向
% 3) 通过 FFT 计算不同平移下的“加权张量平方失配”能量
% 4) 选出全局最优 (theta, tx, ty)
%
% 修正版要点：
% - 目标函数改为加权张量平方失配最小化，不再只做相关最大化
% - 不再奖励大 overlap，只保留最小重叠阈值约束
% - 不直接使用 alpha 做匹配，只使用 Hxx / Hxy / Hyy 三分量
% - 本步骤只估计二维粗配准参数 (theta, tx, ty)
% - 完整 6-DOF 中的 Rx, Ry, tz 留给 Step4 处理

clear; clc; close all;

%% ========================= 1. 读取数据 =========================
fit_name  = 'phs_freeform_fit_data_generalized.mat';
hess_name = 'hessian_validation_from_generalized_fit.mat';

if ~isfile(fit_name)
    error('找不到 %s，请先运行 Step1。', fit_name);
end
if ~isfile(hess_name)
    error('找不到 %s，请先运行 Step2。', hess_name);
end

S1 = load(fit_name);
S2 = load(hess_name);

required_s2 = { ...
    'sub1_grid', 'sub2_grid', ...
    'sub1_mask', 'sub2_mask', ...
    'Hxx_fit_1', 'Hxy_fit_1', 'Hyy_fit_1', ...
    'Hxx_fit_2', 'Hxy_fit_2', 'Hyy_fit_2', ...
    'eta_fit_1', 'eta_fit_2' ...
    };

for k = 1:numel(required_s2)
    if ~isfield(S2, required_s2{k})
        error('Step2 MAT 文件缺少变量：%s', required_s2{k});
    end
end

sub1_grid = S2.sub1_grid;
sub2_grid = S2.sub2_grid;

sub1_mask = logical(S2.sub1_mask);
sub2_mask = logical(S2.sub2_mask);

Hxx1 = S2.Hxx_fit_1;
Hxy1 = S2.Hxy_fit_1;
Hyy1 = S2.Hyy_fit_1;

Hxx2 = S2.Hxx_fit_2;
Hxy2 = S2.Hxy_fit_2;
Hyy2 = S2.Hyy_fit_2;

eta1 = S2.eta_fit_1;
eta2 = S2.eta_fit_2;

dx1 = abs(sub1_grid.x_vec(2) - sub1_grid.x_vec(1));
dy1 = abs(sub1_grid.y_vec(2) - sub1_grid.y_vec(1));
dx2 = abs(sub2_grid.x_vec(2) - sub2_grid.x_vec(1));
dy2 = abs(sub2_grid.y_vec(2) - sub2_grid.y_vec(1));

if abs(dx1 - dx2) > 1e-12 || abs(dy1 - dy2) > 1e-12
    warning('sub1 与 sub2 网格步长不完全一致，后续以平均步长处理。');
end

dx = 0.5 * (dx1 + dx2);
dy = 0.5 * (dy1 + dy2);

%% ========================= 2. 参数设置 =========================
search_cfg = struct();

% 角度搜索范围
if isfield(S1, 'surface_params') && isfield(S1.surface_params, 'sub2_perturb') ...
        && isfield(S1.surface_params.sub2_perturb, 'rz_range_deg')
    rz_range = S1.surface_params.sub2_perturb.rz_range_deg;
    ang_max = max(abs(rz_range)) + 2.0;
else
    ang_max = 6.0;
end

search_cfg.theta_min_deg = -ang_max;
search_cfg.theta_max_deg = +ang_max;
search_cfg.theta_step_deg = 0.25;

% 边界权重参数
search_cfg.bd_soft_width_pix = 6;

% eta 特征权重参数
search_cfg.eta_gate_ratio = 0.05;
search_cfg.eta_power = 1.0;

% 最小重叠约束
search_cfg.min_overlap_ratio = 0.20;
search_cfg.min_overlap_pix   = 1500;

theta_list_deg = search_cfg.theta_min_deg : search_cfg.theta_step_deg : search_cfg.theta_max_deg;
num_theta = numel(theta_list_deg);

%% ========================= 3. 构造权重场 =========================
Wbd1 = build_boundary_weight(sub1_mask, search_cfg.bd_soft_width_pix);
Wbd2 = build_boundary_weight(sub2_mask, search_cfg.bd_soft_width_pix);

Weta1 = build_eta_weight(eta1, sub1_mask, search_cfg.eta_gate_ratio, search_cfg.eta_power);
Weta2 = build_eta_weight(eta2, sub2_mask, search_cfg.eta_gate_ratio, search_cfg.eta_power);

W1 = Wbd1 .* Weta1;
W2 = Wbd2 .* Weta2;

W1(~sub1_mask) = 0;
W2(~sub2_mask) = 0;

M1 = double(W1 > 0);
N1_eff = nnz(M1);

% 零填充版本
Hxx1_z = zero_out_invalid(Hxx1, sub1_mask);
Hxy1_z = zero_out_invalid(Hxy1, sub1_mask);
Hyy1_z = zero_out_invalid(Hyy1, sub1_mask);

%% ========================= 4. 估计近似二维真值（仅用于诊断） =========================
has_planar_gt = false;
theta_gt_deg = nan;
tx_gt = nan;
ty_gt = nan;

if isfield(S1, 'point_cloud_sub2_true') && isfield(S1, 'point_cloud_sub2_true_global')
    X2_local  = S1.point_cloud_sub2_true(:,1:2);
    X2_global = S1.point_cloud_sub2_true_global(:,1:2);

    [R_gt_2d, t_gt_2d] = estimate_rigid_2d(X2_local, X2_global);
    theta_gt_deg = atan2d(R_gt_2d(2,1), R_gt_2d(1,1));
    tx_gt = t_gt_2d(1);
    ty_gt = t_gt_2d(2);
    has_planar_gt = true;
end

%% ========================= 5. 角度搜索 + FFT 平移搜索 =========================
angle_results = struct();
angle_results.theta_deg = theta_list_deg(:);
angle_results.best_tx = nan(num_theta, 1);
angle_results.best_ty = nan(num_theta, 1);
angle_results.best_energy = inf(num_theta, 1);
angle_results.best_overlap_ratio = nan(num_theta, 1);
angle_results.best_overlap_pix = nan(num_theta, 1);
angle_results.best_lag_x = nan(num_theta, 1);
angle_results.best_lag_y = nan(num_theta, 1);

global_best = struct();
global_best.energy = inf;
global_best.theta_deg = nan;
global_best.tx = nan;
global_best.ty = nan;
global_best.overlap_ratio = nan;
global_best.overlap_pix = nan;
global_best.energy_map = [];
global_best.overlap_ratio_map = [];
global_best.overlap_pix_map = [];
global_best.tx_axis = [];
global_best.ty_axis = [];
global_best.rotated_fields = struct();

disp('================ Step3: FFT coarse registration (revised SSE form) =================');

for it = 1:num_theta
    theta_deg = theta_list_deg(it);
    theta_rad = deg2rad(theta_deg);

    % ---- 5.1 旋转 sub2 到候选方向坐标系 ----
    rot_data = rotate_sub2_fields_to_rotated_grid( ...
        sub2_grid.x_vec, sub2_grid.y_vec, ...
        Hxx2, Hxy2, Hyy2, W2, sub2_mask, ...
        theta_rad, dx, dy);

    Hxx2r = rot_data.Hxx_rot;
    Hxy2r = rot_data.Hxy_rot;
    Hyy2r = rot_data.Hyy_rot;
    W2r   = rot_data.W_rot;
    M2r   = rot_data.mask_rot;
    x2r_vec = rot_data.x_vec_rot;
    y2r_vec = rot_data.y_vec_rot;

    if nnz(M2r) < search_cfg.min_overlap_pix
        continue;
    end

    Hxx2r_z = zero_out_invalid(Hxx2r, M2r);
    Hxy2r_z = zero_out_invalid(Hxy2r, M2r);
    Hyy2r_z = zero_out_invalid(Hyy2r, M2r);

    M2 = double(W2r > 0);
    N2_eff = nnz(M2);
    if N2_eff < search_cfg.min_overlap_pix
        continue;
    end

    % ---- 5.2 计算 overlap 图 ----
    overlap_weight_map = xcorr2_fft(W1, W2r);
    overlap_pix_map    = xcorr2_fft(M1, M2);

    min_overlap_needed = max(search_cfg.min_overlap_pix, ...
        round(search_cfg.min_overlap_ratio * min(N1_eff, N2_eff)));
    valid_map = overlap_pix_map >= min_overlap_needed;

    % lag 轴与对应物理平移
    lag_y = -(size(Hxx2r_z,1)-1):(size(Hxx1_z,1)-1);
    lag_x = -(size(Hxx2r_z,2)-1):(size(Hxx1_z,2)-1);

    tx_axis = sub1_grid.x_vec(1) - x2r_vec(1) + lag_x .* dx;
    ty_axis = sub1_grid.y_vec(1) - y2r_vec(1) + lag_y .* dy;

    % ---- 5.3 基于 FFT 的加权平方失配展开 ----
    % E = sum W1*W2_shift * [ (Hxx1-Hxx2)^2 + 2(Hxy1-Hxy2)^2 + (Hyy1-Hyy2)^2 ]
    %
    % 对每个分量 k：
    % sum W1*W2_shift*(A-B)^2
    % = sum (W1*A^2) * W2_shift + sum W1 * (W2*B^2)_shift - 2*sum (W1*A)*(W2*B)_shift

    [E_xx, C_xx] = weighted_component_sse_fft(Hxx1_z, W1, Hxx2r_z, W2r);
    [E_xy, C_xy] = weighted_component_sse_fft(Hxy1_z, W1, Hxy2r_z, W2r);
    [E_yy, C_yy] = weighted_component_sse_fft(Hyy1_z, W1, Hyy2r_z, W2r);

    energy_sum_map = E_xx + 2 * E_xy + E_yy;
    energy_map = energy_sum_map ./ (overlap_weight_map + eps);

    % 为诊断保留相关项
    corr_map = C_xx + 2 * C_xy + C_yy; %#ok<NASGU>

    energy_map(~valid_map) = inf;

    % ---- 5.4 当前角度下最优平移 ----
    [best_energy_this, idx_best] = min(energy_map(:));
    if isfinite(best_energy_this)
        [iy_best, ix_best] = ind2sub(size(energy_map), idx_best);

        best_overlap_ratio_this = overlap_pix_map(iy_best, ix_best) / max(min(N1_eff, N2_eff), 1);
        best_overlap_pix_this   = overlap_pix_map(iy_best, ix_best);
        tx_best_this = tx_axis(ix_best);
        ty_best_this = ty_axis(iy_best);

        angle_results.best_tx(it) = tx_best_this;
        angle_results.best_ty(it) = ty_best_this;
        angle_results.best_energy(it) = best_energy_this;
        angle_results.best_overlap_ratio(it) = best_overlap_ratio_this;
        angle_results.best_overlap_pix(it) = best_overlap_pix_this;
        angle_results.best_lag_x(it) = lag_x(ix_best);
        angle_results.best_lag_y(it) = lag_y(iy_best);

        if best_energy_this < global_best.energy
            global_best.energy = best_energy_this;
            global_best.theta_deg = theta_deg;
            global_best.tx = tx_best_this;
            global_best.ty = ty_best_this;
            global_best.overlap_ratio = best_overlap_ratio_this;
            global_best.overlap_pix = best_overlap_pix_this;
            global_best.energy_map = energy_map;
            global_best.overlap_ratio_map = overlap_pix_map ./ max(min(N1_eff, N2_eff), 1);
            global_best.overlap_pix_map = overlap_pix_map;
            global_best.tx_axis = tx_axis;
            global_best.ty_axis = ty_axis;
            global_best.rotated_fields = rot_data;
        end
    end

    fprintf('theta = %+7.3f deg | best energy = %.6e | tx = %+9.4f | ty = %+9.4f | overlap = %.4f\n', ...
        theta_deg, angle_results.best_energy(it), ...
        angle_results.best_tx(it), angle_results.best_ty(it), ...
        angle_results.best_overlap_ratio(it));
end

if ~isfinite(global_best.energy)
    error('Step3 未找到有效粗配准结果。建议适当放宽角度范围、降低 overlap 阈值或检查 Step2 输出。');
end

%% ========================= 6. 在 sub1 网格上重建最优对齐后的 sub2 场 =========================
best_theta_rad = deg2rad(global_best.theta_deg);

aligned_on_sub1 = sample_aligned_sub2_on_sub1_grid( ...
    sub1_grid.X, sub1_grid.Y, ...
    sub2_grid.x_vec, sub2_grid.y_vec, ...
    Hxx2, Hxy2, Hyy2, W2, sub2_mask, ...
    best_theta_rad, global_best.tx, global_best.ty);

Hxx2_best_on1 = aligned_on_sub1.Hxx;
Hxy2_best_on1 = aligned_on_sub1.Hxy;
Hyy2_best_on1 = aligned_on_sub1.Hyy;
W2_best_on1   = aligned_on_sub1.W;
M2_best_on1   = aligned_on_sub1.mask;

overlap_mask_best = sub1_mask & M2_best_on1 ...
    & isfinite(Hxx2_best_on1) & isfinite(Hxy2_best_on1) & isfinite(Hyy2_best_on1);

tensor_misfit_best = nan(size(sub1_mask));
tensor_misfit_best(overlap_mask_best) = ...
    (Hxx1(overlap_mask_best) - Hxx2_best_on1(overlap_mask_best)).^2 + ...
    2 * (Hxy1(overlap_mask_best) - Hxy2_best_on1(overlap_mask_best)).^2 + ...
    (Hyy1(overlap_mask_best) - Hyy2_best_on1(overlap_mask_best)).^2;

best_overlap_ratio_check = nnz(overlap_mask_best) / max(min(nnz(sub1_mask), nnz(sub2_mask)), 1);

%% ========================= 7. 与近似二维真值比较（若可用） =========================
if has_planar_gt
    theta_err_deg = wrap_to_180(global_best.theta_deg - theta_gt_deg);
    tx_err = global_best.tx - tx_gt;
    ty_err = global_best.ty - ty_gt;
else
    theta_err_deg = nan;
    tx_err = nan;
    ty_err = nan;
end

%% ========================= 8. 输出结果 =========================
disp(' ');
disp('================ Step3 final result =================');
fprintf('Best theta = %+9.4f deg\n', global_best.theta_deg);
fprintf('Best tx    = %+9.4f mm\n', global_best.tx);
fprintf('Best ty    = %+9.4f mm\n', global_best.ty);
fprintf('Best energy = %.6e\n', global_best.energy);
fprintf('Best overlap ratio (FFT map) = %.6f\n', global_best.overlap_ratio);
fprintf('Best overlap ratio (check)   = %.6f\n', best_overlap_ratio_check);
fprintf('Best overlap pixels          = %d\n', round(global_best.overlap_pix));

if has_planar_gt
    disp('---------------- Approx planar GT comparison ----------------');
    fprintf('GT theta = %+9.4f deg\n', theta_gt_deg);
    fprintf('GT tx    = %+9.4f mm\n', tx_gt);
    fprintf('GT ty    = %+9.4f mm\n', ty_gt);
    fprintf('theta error = %+9.4f deg\n', theta_err_deg);
    fprintf('tx error    = %+9.4f mm\n', tx_err);
    fprintf('ty error    = %+9.4f mm\n', ty_err);
end

%% ========================= 9. 可视化：角度搜索曲线 =========================
figure('Position', [100, 100, 900, 420], 'Name', 'Angle Search Curve');
plot(angle_results.theta_deg, angle_results.best_energy, 'o-', 'LineWidth', 1.2);
hold on;
plot(global_best.theta_deg, global_best.energy, 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
xlabel('\theta (deg)');
ylabel('Best energy');
title('Angle search curve');
grid on;

%% ========================= 10. 可视化：最优角度下的 energy map =========================
figure('Position', [120, 120, 1200, 460], 'Name', 'Best FFT energy map');

subplot(1,2,1);
imagesc(global_best.tx_axis, global_best.ty_axis, global_best.energy_map);
set(gca, 'YDir', 'normal');
xlabel('t_x (mm)');
ylabel('t_y (mm)');
title(sprintf('Energy map at \\theta = %.3f deg', global_best.theta_deg));
axis image;
colorbar;

subplot(1,2,2);
imagesc(global_best.tx_axis, global_best.ty_axis, global_best.overlap_ratio_map);
set(gca, 'YDir', 'normal');
xlabel('t_x (mm)');
ylabel('t_y (mm)');
title('Overlap ratio map');
axis image;
colorbar;

%% ========================= 11. 可视化：配准前后 Hessian 对比 =========================
figure('Position', [80, 80, 1350, 760], 'Name', 'Best coarse alignment on sub1 grid');

subplot(3,3,1);
imagesc(sub1_grid.x_vec, sub1_grid.y_vec, mask_to_nan(Hxx1, sub1_mask));
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('Sub1 Hxx');

subplot(3,3,2);
imagesc(sub1_grid.x_vec, sub1_grid.y_vec, mask_to_nan(Hxx2_best_on1, overlap_mask_best));
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('Aligned Sub2 Hxx on Sub1');

subplot(3,3,3);
imagesc(sub1_grid.x_vec, sub1_grid.y_vec, mask_to_nan(Hxx1 - Hxx2_best_on1, overlap_mask_best));
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('Hxx difference');

subplot(3,3,4);
imagesc(sub1_grid.x_vec, sub1_grid.y_vec, mask_to_nan(Hxy1, sub1_mask));
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('Sub1 Hxy');

subplot(3,3,5);
imagesc(sub1_grid.x_vec, sub1_grid.y_vec, mask_to_nan(Hxy2_best_on1, overlap_mask_best));
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('Aligned Sub2 Hxy on Sub1');

subplot(3,3,6);
imagesc(sub1_grid.x_vec, sub1_grid.y_vec, mask_to_nan(Hxy1 - Hxy2_best_on1, overlap_mask_best));
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('Hxy difference');

subplot(3,3,7);
imagesc(sub1_grid.x_vec, sub1_grid.y_vec, mask_to_nan(Hyy1, sub1_mask));
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('Sub1 Hyy');

subplot(3,3,8);
imagesc(sub1_grid.x_vec, sub1_grid.y_vec, mask_to_nan(Hyy2_best_on1, overlap_mask_best));
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('Aligned Sub2 Hyy on Sub1');

subplot(3,3,9);
imagesc(sub1_grid.x_vec, sub1_grid.y_vec, mask_to_nan(tensor_misfit_best, overlap_mask_best));
set(gca, 'YDir', 'normal'); axis image; colorbar;
title('Tensor mismatch');

%% ========================= 12. 保存 Step3 结果 =========================
coarse_result = struct();
coarse_result.theta_deg = global_best.theta_deg;
coarse_result.theta_rad = best_theta_rad;
coarse_result.tx = global_best.tx;
coarse_result.ty = global_best.ty;
coarse_result.energy = global_best.energy;
coarse_result.overlap_ratio_fft = global_best.overlap_ratio;
coarse_result.overlap_ratio_check = best_overlap_ratio_check;
coarse_result.overlap_pix = global_best.overlap_pix;

if has_planar_gt
    coarse_result.theta_gt_deg = theta_gt_deg;
    coarse_result.tx_gt = tx_gt;
    coarse_result.ty_gt = ty_gt;
    coarse_result.theta_err_deg = theta_err_deg;
    coarse_result.tx_err = tx_err;
    coarse_result.ty_err = ty_err;
else
    coarse_result.theta_gt_deg = nan;
    coarse_result.tx_gt = nan;
    coarse_result.ty_gt = nan;
    coarse_result.theta_err_deg = nan;
    coarse_result.tx_err = nan;
    coarse_result.ty_err = nan;
end

save_name = 'fft_registration_results_generalized.mat';
save(save_name, ...
    'search_cfg', ...
    'theta_list_deg', ...
    'angle_results', ...
    'coarse_result', ...
    'W1', 'W2', ...
    'global_best', ...
    'aligned_on_sub1', ...
    'Hxx2_best_on1', 'Hxy2_best_on1', 'Hyy2_best_on1', ...
    'W2_best_on1', 'M2_best_on1', ...
    'overlap_mask_best', ...
    'tensor_misfit_best', ...
    'has_planar_gt', 'theta_gt_deg', 'tx_gt', 'ty_gt', ...
    '-v7.3');

disp(['已保存 MAT 文件：', save_name]);

%% ========================= 局部函数 =========================

function V = zero_out_invalid(V, mask)
V(~mask) = 0;
V(~isfinite(V)) = 0;
end

function V = mask_to_nan(V, mask)
V(~mask) = nan;
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

function Weta = build_eta_weight(eta, mask, gate_ratio, eta_power)
mask = logical(mask);
eta_max = masked_max_finite(eta, mask);

if ~isfinite(eta_max) || eta_max <= 0
    Weta = zeros(size(eta));
    return;
end

eta_norm = eta / (eta_max + eps);
eta_norm(~isfinite(eta_norm)) = 0;
eta_norm(~mask) = 0;

Weta = max(eta_norm, 0) .^ eta_power;
Weta(eta_norm < gate_ratio) = 0;
Weta(~mask) = 0;
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

function rot_data = rotate_sub2_fields_to_rotated_grid(x_vec, y_vec, Hxx, Hxy, Hyy, W, mask, theta, dx, dy)
c = cos(theta);
s = sin(theta);

x_min = x_vec(1); x_max = x_vec(end);
y_min = y_vec(1); y_max = y_vec(end);

corners = [ ...
    x_min, y_min;
    x_min, y_max;
    x_max, y_min;
    x_max, y_max];

R = [c, -s; s, c];
corners_rot = (R * corners.').';

x_rot_min = floor(min(corners_rot(:,1)) / dx) * dx;
x_rot_max = ceil(max(corners_rot(:,1)) / dx) * dx;
y_rot_min = floor(min(corners_rot(:,2)) / dy) * dy;
y_rot_max = ceil(max(corners_rot(:,2)) / dy) * dy;

x_vec_rot = x_rot_min : dx : x_rot_max;
y_vec_rot = y_rot_min : dy : y_rot_max;
[Xr, Yr] = meshgrid(x_vec_rot, y_vec_rot);

% x' = R x  ->  x = R^T x'
Xq =  c * Xr + s * Yr;
Yq = -s * Xr + c * Yr;

Hxx0 = zero_out_invalid(Hxx, mask);
Hxy0 = zero_out_invalid(Hxy, mask);
Hyy0 = zero_out_invalid(Hyy, mask);
W0   = zero_out_invalid(W,   mask);
M0   = double(mask);

Hxx_q = interp2(x_vec, y_vec, Hxx0, Xq, Yq, 'linear', 0);
Hxy_q = interp2(x_vec, y_vec, Hxy0, Xq, Yq, 'linear', 0);
Hyy_q = interp2(x_vec, y_vec, Hyy0, Xq, Yq, 'linear', 0);
W_q   = interp2(x_vec, y_vec, W0,   Xq, Yq, 'linear', 0);
M_q   = interp2(x_vec, y_vec, M0,   Xq, Yq, 'linear', 0) > 0.5;

% H' = R H R^T
Hxx_rot = c^2 .* Hxx_q - 2*c*s .* Hxy_q + s^2 .* Hyy_q;
Hxy_rot = c*s .* Hxx_q + (c^2 - s^2) .* Hxy_q - c*s .* Hyy_q;
Hyy_rot = s^2 .* Hxx_q + 2*c*s .* Hxy_q + c^2 .* Hyy_q;

Hxx_rot(~M_q) = nan;
Hxy_rot(~M_q) = nan;
Hyy_rot(~M_q) = nan;
W_q(~M_q) = 0;

rot_data = struct();
rot_data.theta = theta;
rot_data.x_vec_rot = x_vec_rot;
rot_data.y_vec_rot = y_vec_rot;
rot_data.Hxx_rot = Hxx_rot;
rot_data.Hxy_rot = Hxy_rot;
rot_data.Hyy_rot = Hyy_rot;
rot_data.W_rot = W_q;
rot_data.mask_rot = M_q;
rot_data.X_rot = Xr;
rot_data.Y_rot = Yr;
end

function aligned = sample_aligned_sub2_on_sub1_grid(X1, Y1, x2_vec, y2_vec, Hxx2, Hxy2, Hyy2, W2, mask2, theta, tx, ty)
c = cos(theta);
s = sin(theta);

% x1 = R(theta) * x2 + t
% => x2 = R(theta)^T * (x1 - t)
Xrot = X1 - tx;
Yrot = Y1 - ty;

Xq =  c * Xrot + s * Yrot;
Yq = -s * Xrot + c * Yrot;

Hxx0 = zero_out_invalid(Hxx2, mask2);
Hxy0 = zero_out_invalid(Hxy2, mask2);
Hyy0 = zero_out_invalid(Hyy2, mask2);
W20  = zero_out_invalid(W2,   mask2);
M20  = double(mask2);

Hxx_q = interp2(x2_vec, y2_vec, Hxx0, Xq, Yq, 'linear', 0);
Hxy_q = interp2(x2_vec, y2_vec, Hxy0, Xq, Yq, 'linear', 0);
Hyy_q = interp2(x2_vec, y2_vec, Hyy0, Xq, Yq, 'linear', 0);
W_q   = interp2(x2_vec, y2_vec, W20,  Xq, Yq, 'linear', 0);
M_q   = interp2(x2_vec, y2_vec, M20,  Xq, Yq, 'linear', 0) > 0.5;

Hxx_al = c^2 .* Hxx_q - 2*c*s .* Hxy_q + s^2 .* Hyy_q;
Hxy_al = c*s .* Hxx_q + (c^2 - s^2) .* Hxy_q - c*s .* Hyy_q;
Hyy_al = s^2 .* Hxx_q + 2*c*s .* Hxy_q + c^2 .* Hyy_q;

Hxx_al(~M_q) = nan;
Hxy_al(~M_q) = nan;
Hyy_al(~M_q) = nan;
W_q(~M_q) = 0;

aligned = struct();
aligned.Hxx = Hxx_al;
aligned.Hxy = Hxy_al;
aligned.Hyy = Hyy_al;
aligned.W = W_q;
aligned.mask = M_q;
end

function [Ecomp, Ccross] = weighted_component_sse_fft(A1, W1, A2, W2)
% 计算：
% Ecomp = sum W1*W2_shift*(A1 - A2_shift)^2
% 通过 FFT 展开为三项
T1 = xcorr2_fft(W1 .* (A1.^2), W2);
T2 = xcorr2_fft(W1, W2 .* (A2.^2));
Ccross = xcorr2_fft(W1 .* A1, W2 .* A2);
Ecomp = T1 + T2 - 2 * Ccross;
Ecomp(Ecomp < 0 & Ecomp > -1e-12) = 0;
end

function C = xcorr2_fft(A, B)
A = double(A);
B = double(B);

sa = size(A);
sb = size(B);
nout = sa + sb - 1;

Bflip = B(end:-1:1, end:-1:1);
C = real(ifft2(fft2(A, nout(1), nout(2)) .* fft2(Bflip, nout(1), nout(2))));
end

function [R, t] = estimate_rigid_2d(Xsrc, Xdst)
% 估计 Xdst ≈ Xsrc * R^T + t
mu_src = mean(Xsrc, 1);
mu_dst = mean(Xdst, 1);

Xs = Xsrc - mu_src;
Xd = Xdst - mu_dst;

H = Xs.' * Xd;
[U, ~, V] = svd(H);
R = V * U.';

if det(R) < 0
    V(:,end) = -V(:,end);
    R = V * U.';
end

t = mu_dst(:) - R * mu_src(:);
t = t(:).';
end

function ang = wrap_to_180(ang)
ang = mod(ang + 180, 360) - 180;
end