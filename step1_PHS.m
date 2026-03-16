%% step1_PHS.m
% Step 1:
% 1) 生成全局自由曲面真值
% 2) 提取两个子孔径
% 3) 对 sub2 施加随机 6-DOF 扰动，形成独立局部测量坐标系
% 4) 分别对 sub1 / sub2 做 Poly + PHS 拟合
% 5) 保存与原仓库兼容的 MAT 文件，并附加 sub2 的真值位姿信息

clear; clc; close all;
rng(1);

%% ========================= 1. 面形与采样参数 =========================
surface_params = struct();

% 全口径矩形范围（mm）
surface_params.Lx = 100;
surface_params.Ly = 60;

% 全局规则采样分辨率
surface_params.num_points_X = 250;
surface_params.num_points_Y = 150;

% 两个圆形子孔径（在全局坐标中的设计位置）
surface_params.center1 = [-25, 0];
surface_params.center2 = [ 25, 0];
surface_params.R = 40;

% 三次主形貌参数
surface_params.c1 = 5e-5;
surface_params.c2 = 2e-5;
surface_params.c3 = -2e-5;
surface_params.c4 = 8e-5;
surface_params.c5 = 5e-3;
surface_params.c6 = 8e-3;

% 局部自由曲面残差（高斯凸起/凹陷）
surface_params.bump1_amp = 0.030;
surface_params.bump1_x0  = -18;
surface_params.bump1_y0  = 10;
surface_params.bump1_sx  = 10;
surface_params.bump1_sy  = 7;

surface_params.bump2_amp = -0.022;
surface_params.bump2_x0  = 22;
surface_params.bump2_y0  = -12;
surface_params.bump2_sx  = 9;
surface_params.bump2_sy  = 8;

surface_params.bump3_amp = 0.015;
surface_params.bump3_x0  = 5;
surface_params.bump3_y0  = 18;
surface_params.bump3_sx  = 6;
surface_params.bump3_sy  = 6;

% 中频波纹
surface_params.ripple_amp = 0.004;
surface_params.ripple_Lx  = 28;
surface_params.ripple_Ly  = 22;
surface_params.ripple_phi = pi/7;

% 余弦调制
surface_params.cos_amp = 0.003;
surface_params.cos_Lx  = 40;
surface_params.cos_Ly  = 32;

% 测量噪声（单位 mm），假设体现在传感器局部 z 方向
surface_params.add_noise = true;
surface_params.noise_std = 5e-6;

% sub2 的随机 6-DOF 扰动范围
surface_params.sub2_perturb = struct();
surface_params.sub2_perturb.rx_range_deg = [-1.2, 1.2];
surface_params.sub2_perturb.ry_range_deg = [-1.2, 1.2];
surface_params.sub2_perturb.rz_range_deg = [-4.0, 4.0];
surface_params.sub2_perturb.tx_range_mm  = [-3.0, 3.0];
surface_params.sub2_perturb.ty_range_mm  = [-3.0, 3.0];
surface_params.sub2_perturb.tz_range_mm  = [-0.3, 0.3];

%% ========================= 2. 拟合参数 =========================
fit_opts = struct();
fit_opts.normalize_poly     = true;
fit_opts.normalize_phs      = true;
fit_opts.kernel             = 'r4logr';   % 'r4logr' or 'r6logr'
fit_opts.lambda             = 1e-10;
fit_opts.max_centers        = 1500;
fit_opts.skip_phs_if_small  = true;
fit_opts.residual_tol       = 1e-12;

%% ========================= 3. 全局自由曲面真值 =========================
x_vec = linspace(-surface_params.Lx/2, surface_params.Lx/2, surface_params.num_points_X);
y_vec = linspace(-surface_params.Ly/2, surface_params.Ly/2, surface_params.num_points_Y);
[X, Y] = meshgrid(x_vec, y_vec);

[Z_true, Z_cubic_true, Z_residual_true] = surface_freeform_general(X, Y, surface_params);

% 全局点云真值 / 测量值
point_cloud_full_true = [X(:), Y(:), Z_true(:)];
point_cloud_full = point_cloud_full_true;
if surface_params.add_noise
    point_cloud_full(:,3) = point_cloud_full(:,3) + surface_params.noise_std .* randn(size(point_cloud_full,1),1);
end
Z_meas = reshape(point_cloud_full(:,3), size(X));

%% ========================= 4. 提取两个子孔径 =========================
% sub1：保持全局坐标系
[point_cloud_sub1_true_global, idx_sub1] = extract_subaperture(point_cloud_full_true, surface_params.center1, surface_params.R);
point_cloud_sub1 = point_cloud_sub1_true_global;
if surface_params.add_noise
    point_cloud_sub1(:,3) = point_cloud_sub1(:,3) + surface_params.noise_std .* randn(size(point_cloud_sub1,1),1);
end
point_cloud_sub1_true = point_cloud_sub1_true_global;   % 对 sub1，local = global

% sub2：先取全局真值，再施加随机 6-DOF 扰动，转为独立局部测量坐标
[point_cloud_sub2_true_global, idx_sub2] = extract_subaperture(point_cloud_full_true, surface_params.center2, surface_params.R);

sub2_pose_gt = sample_random_pose(surface_params.sub2_perturb, point_cloud_sub2_true_global);

point_cloud_sub2_true = apply_pose_global_to_local(point_cloud_sub2_true_global, sub2_pose_gt);

point_cloud_sub2 = point_cloud_sub2_true;
if surface_params.add_noise
    point_cloud_sub2(:,3) = point_cloud_sub2(:,3) + surface_params.noise_std .* randn(size(point_cloud_sub2,1),1);
end

mask_full_sub1 = reshape(idx_sub1, size(X));
mask_full_sub2 = reshape(idx_sub2, size(X));

%% ========================= 5. 构造规则支撑域网格 =========================
dx = abs(x_vec(2) - x_vec(1));
dy = abs(y_vec(2) - y_vec(1));

% sub1：仍在全局规则网格上裁切
[sub1_grid, sub1_mask] = build_support_grid_from_global_mask( ...
    X, Y, Z_true, Z_meas, Z_cubic_true, Z_residual_true, mask_full_sub1);

sub1_grid.X_global      = sub1_grid.X;
sub1_grid.Y_global      = sub1_grid.Y;
sub1_grid.Z_true_global = sub1_grid.Z_true;
sub1_grid.frame         = 'global';

% sub2：在“扰动后的局部测量坐标系”中重建规则支撑域
[sub2_grid, sub2_mask] = build_support_grid_from_local_points( ...
    point_cloud_sub2_true, point_cloud_sub2, dx, dy);

sub2_grid.frame = 'local_perturbed';

% 将 sub2 局部真值网格反变换回全局，用于后续真值校核
P2_grid_local_true = [sub2_grid.X(:), sub2_grid.Y(:), sub2_grid.Z_true(:)];
valid_grid_2 = isfinite(P2_grid_local_true(:,3));
P2_grid_global_true = nan(size(P2_grid_local_true));
P2_grid_global_true(valid_grid_2,:) = apply_pose_local_to_global(P2_grid_local_true(valid_grid_2,:), sub2_pose_gt);

sub2_grid.X_global      = reshape(P2_grid_global_true(:,1), size(sub2_grid.X));
sub2_grid.Y_global      = reshape(P2_grid_global_true(:,2), size(sub2_grid.Y));
sub2_grid.Z_true_global = reshape(P2_grid_global_true(:,3), size(sub2_grid.X));

%% ========================= 6. 子孔径 1：Poly + PHS 拟合 =========================
disp('开始对子孔径 1 进行拟合...');
model_sub1 = fit_poly_plus_phs( ...
    point_cloud_sub1(:,1), point_cloud_sub1(:,2), point_cloud_sub1(:,3), fit_opts);

%% ========================= 7. 子孔径 2：Poly + PHS 拟合 =========================
disp('开始对子孔径 2（随机 6-DOF 扰动后局部坐标）进行拟合...');
model_sub2 = fit_poly_plus_phs( ...
    point_cloud_sub2(:,1), point_cloud_sub2(:,2), point_cloud_sub2(:,3), fit_opts);

%% ========================= 8. 采样点上的拟合结果 =========================
z_fit_sub1_pts  = eval_poly_plus_phs(model_sub1, point_cloud_sub1(:,1), point_cloud_sub1(:,2));
z_fit_sub2_pts  = eval_poly_plus_phs(model_sub2, point_cloud_sub2(:,1), point_cloud_sub2(:,2));

z_poly_sub1_pts = eval_poly_model(model_sub1.polyModel, point_cloud_sub1(:,1), point_cloud_sub1(:,2));
z_poly_sub2_pts = eval_poly_model(model_sub2.polyModel, point_cloud_sub2(:,1), point_cloud_sub2(:,2));

z_true_sub1_pts = point_cloud_sub1_true(:,3);
z_true_sub2_pts = point_cloud_sub2_true(:,3);

res_sub1_meas      = z_fit_sub1_pts  - point_cloud_sub1(:,3);
res_sub2_meas      = z_fit_sub2_pts  - point_cloud_sub2(:,3);
res_sub1_true      = z_fit_sub1_pts  - z_true_sub1_pts;
res_sub2_true      = z_fit_sub2_pts  - z_true_sub2_pts;
res_sub1_poly_true = z_poly_sub1_pts - z_true_sub1_pts;
res_sub2_poly_true = z_poly_sub2_pts - z_true_sub2_pts;

metrics_sub1_meas      = calc_metrics(res_sub1_meas);
metrics_sub2_meas      = calc_metrics(res_sub2_meas);
metrics_sub1_true      = calc_metrics(res_sub1_true);
metrics_sub2_true      = calc_metrics(res_sub2_true);
metrics_sub1_poly_true = calc_metrics(res_sub1_poly_true);
metrics_sub2_poly_true = calc_metrics(res_sub2_poly_true);

disp('================ 采样点误差统计 ================');
disp(['Sub1 相对测量值 RMSE = ', num2str(metrics_sub1_meas.rmse, '%.6e')]);
disp(['Sub2 相对测量值 RMSE = ', num2str(metrics_sub2_meas.rmse, '%.6e')]);
disp(['Sub1 相对真值   RMSE = ', num2str(metrics_sub1_true.rmse, '%.6e')]);
disp(['Sub2 相对真值   RMSE = ', num2str(metrics_sub2_true.rmse, '%.6e')]);
disp(['Sub1 Poly-only  RMSE = ', num2str(metrics_sub1_poly_true.rmse, '%.6e')]);
disp(['Sub2 Poly-only  RMSE = ', num2str(metrics_sub2_poly_true.rmse, '%.6e')]);

%% ========================= 9. 局部规则网格上的拟合结果 =========================
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

Res1_grid      = Z1_fit_grid  - Z1_true_grid;
Res2_grid      = Z2_fit_grid  - Z2_true_grid;
Res1_poly_grid = Z1_poly_grid - Z1_true_grid;
Res2_poly_grid = Z2_poly_grid - Z2_true_grid;

metrics_sub1_grid = calc_metrics(Res1_grid(sub1_mask));
metrics_sub2_grid = calc_metrics(Res2_grid(sub2_mask));
metrics_sub1_poly = calc_metrics(Res1_poly_grid(sub1_mask));
metrics_sub2_poly = calc_metrics(Res2_poly_grid(sub2_mask));

disp('================ 规则网格误差统计 ================');
disp(['Sub1 Final Grid RMSE    = ', num2str(metrics_sub1_grid.rmse, '%.6e')]);
disp(['Sub2 Final Grid RMSE    = ', num2str(metrics_sub2_grid.rmse, '%.6e')]);
disp(['Sub1 Poly-only Grid RMSE= ', num2str(metrics_sub1_poly.rmse, '%.6e')]);
disp(['Sub2 Poly-only Grid RMSE= ', num2str(metrics_sub2_poly.rmse, '%.6e')]);

%% ========================= 10. 残差面真实值与拟合值 =========================
Z1_res_true_grid = Z1_true_grid - Z1_poly_grid;
Z2_res_true_grid = Z2_true_grid - Z2_poly_grid;

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

disp('================ PHS 残差面误差 ================');
disp(['Sub1 Residual-surface RMSE = ', num2str(metrics_sub1_residual.rmse, '%.6e')]);
disp(['Sub2 Residual-surface RMSE = ', num2str(metrics_sub2_residual.rmse, '%.6e')]);

%% ========================= 11. 保存 MAT 文件 =========================
save_name = 'phs_freeform_fit_data_generalized.mat';

save(save_name, ...
    'surface_params', ...
    'fit_opts', ...
    'x_vec', 'y_vec', 'X', 'Y', ...
    'Z_true', 'Z_meas', 'Z_cubic_true', 'Z_residual_true', ...
    'point_cloud_full', 'point_cloud_full_true', ...
    'idx_sub1', 'idx_sub2', ...
    'mask_full_sub1', 'mask_full_sub2', ...
    'point_cloud_sub1', 'point_cloud_sub2', ...
    'point_cloud_sub1_true', 'point_cloud_sub2_true', ...
    'point_cloud_sub1_true_global', 'point_cloud_sub2_true_global', ...
    'sub1_grid', 'sub2_grid', ...
    'sub1_mask', 'sub2_mask', ...
    'model_sub1', 'model_sub2', ...
    'sub2_pose_gt', ...
    'z_fit_sub1_pts', 'z_fit_sub2_pts', ...
    'z_poly_sub1_pts', 'z_poly_sub2_pts', ...
    'z_true_sub1_pts', 'z_true_sub2_pts', ...
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
disp('================ sub2 真值扰动参数 ================');
fprintf('rx = %+8.4f deg\n', sub2_pose_gt.rx_deg);
fprintf('ry = %+8.4f deg\n', sub2_pose_gt.ry_deg);
fprintf('rz = %+8.4f deg\n', sub2_pose_gt.rz_deg);
fprintf('tx = %+8.4f mm\n',  sub2_pose_gt.t_local_from_anchor(1));
fprintf('ty = %+8.4f mm\n',  sub2_pose_gt.t_local_from_anchor(2));
fprintf('tz = %+8.4f mm\n',  sub2_pose_gt.t_local_from_anchor(3));

%% ========================= 局部函数 =========================

function [Z, Z_cubic, Z_residual] = surface_freeform_general(X, Y, p)
Z_cubic = p.c1 .* X.^3 ...
        + p.c2 .* (X.^2) .* Y ...
        + p.c3 .* X .* (Y.^2) ...
        + p.c4 .* Y.^3 ...
        + p.c5 .* X ...
        + p.c6 .* Y;

B1 = p.bump1_amp .* exp(-((X - p.bump1_x0).^2 ./ (2*p.bump1_sx^2) + (Y - p.bump1_y0).^2 ./ (2*p.bump1_sy^2)));
B2 = p.bump2_amp .* exp(-((X - p.bump2_x0).^2 ./ (2*p.bump2_sx^2) + (Y - p.bump2_y0).^2 ./ (2*p.bump2_sy^2)));
B3 = p.bump3_amp .* exp(-((X - p.bump3_x0).^2 ./ (2*p.bump3_sx^2) + (Y - p.bump3_y0).^2 ./ (2*p.bump3_sy^2)));

Ripple = p.ripple_amp .* sin(2*pi*X ./ p.ripple_Lx + p.ripple_phi) .* cos(2*pi*Y ./ p.ripple_Ly);
CosMod = p.cos_amp .* cos(2*pi*X ./ p.cos_Lx) .* cos(2*pi*Y ./ p.cos_Ly);

Z_residual = B1 + B2 + B3 + Ripple + CosMod;
Z = Z_cubic + Z_residual;
end

function [pc_sub, idx] = extract_subaperture(point_cloud, center_xy, R)
dx = point_cloud(:,1) - center_xy(1);
dy = point_cloud(:,2) - center_xy(2);
idx = (dx.^2 + dy.^2) <= R^2;
pc_sub = point_cloud(idx, :);
end

function pose = sample_random_pose(cfg, pc_global)
anchor = mean(pc_global, 1);

rx_deg = uniform_rand(cfg.rx_range_deg);
ry_deg = uniform_rand(cfg.ry_range_deg);
rz_deg = uniform_rand(cfg.rz_range_deg);

tx = uniform_rand(cfg.tx_range_mm);
ty = uniform_rand(cfg.ty_range_mm);
tz = uniform_rand(cfg.tz_range_mm);

Rx = [1, 0, 0; ...
      0, cosd(rx_deg), -sind(rx_deg); ...
      0, sind(rx_deg),  cosd(rx_deg)];

Ry = [ cosd(ry_deg), 0, sind(ry_deg); ...
       0,            1, 0; ...
      -sind(ry_deg), 0, cosd(ry_deg)];

Rz = [cosd(rz_deg), -sind(rz_deg), 0; ...
      sind(rz_deg),  cosd(rz_deg), 0; ...
      0,             0,            1];

% 采用 Z-Y-X 顺序
R_global_to_local = Rz * Ry * Rx;
t_local_from_anchor = [tx, ty, tz];

pose = struct();
pose.rx_deg = rx_deg;
pose.ry_deg = ry_deg;
pose.rz_deg = rz_deg;
pose.R_global_to_local = R_global_to_local;
pose.R_local_to_global = R_global_to_local.';
pose.anchor_global = anchor;
pose.t_local_from_anchor = t_local_from_anchor;
end

function P_local = apply_pose_global_to_local(P_global, pose)
Pg = P_global - pose.anchor_global;
P_local = (pose.R_global_to_local * Pg.').';
P_local = P_local + pose.t_local_from_anchor;
end

function P_global = apply_pose_local_to_global(P_local, pose)
Pl = P_local - pose.t_local_from_anchor;
P_global = (pose.R_local_to_global * Pl.').';
P_global = P_global + pose.anchor_global;
end

function [grid, mask_crop] = build_support_grid_from_global_mask(X, Y, Z_true, Z_meas, Z_cubic_true, Z_residual_true, mask_full)
rows = any(mask_full, 2);
cols = any(mask_full, 1);

grid = struct();
grid.X = X(rows, cols);
grid.Y = Y(rows, cols);
grid.x_vec = grid.X(1,:);
grid.y_vec = grid.Y(:,1);
grid.Z_true = Z_true(rows, cols);
grid.Z_meas = Z_meas(rows, cols);
grid.Z_cubic_true = Z_cubic_true(rows, cols);
grid.Z_residual_true = Z_residual_true(rows, cols);

mask_crop = mask_full(rows, cols);

grid.Z_true(~mask_crop) = nan;
grid.Z_meas(~mask_crop) = nan;
grid.Z_cubic_true(~mask_crop) = nan;
grid.Z_residual_true(~mask_crop) = nan;
end

function [grid, mask] = build_support_grid_from_local_points(pc_true, pc_meas, dx, dy)
x_true = pc_true(:,1);
y_true = pc_true(:,2);
z_true = pc_true(:,3);

x_meas = pc_meas(:,1);
y_meas = pc_meas(:,2);
z_meas = pc_meas(:,3);

xmin = floor(min(x_true) / dx) * dx;
xmax = ceil(max(x_true)  / dx) * dx;
ymin = floor(min(y_true) / dy) * dy;
ymax = ceil(max(y_true)  / dy) * dy;

xv = xmin:dx:xmax;
yv = ymin:dy:ymax;
[Xg, Yg] = meshgrid(xv, yv);

k = boundary(x_true, y_true, 0.95);
bx = x_true(k);
by = y_true(k);
mask_geom = inpolygon(Xg, Yg, bx, by);

F_true = scatteredInterpolant(x_true, y_true, z_true, 'natural', 'none');
F_meas = scatteredInterpolant(x_meas, y_meas, z_meas, 'natural', 'none');

Z_true_grid = F_true(Xg, Yg);
Z_meas_grid = F_meas(Xg, Yg);

mask = mask_geom & isfinite(Z_true_grid) & isfinite(Z_meas_grid);

Z_true_grid(~mask) = nan;
Z_meas_grid(~mask) = nan;

grid = struct();
grid.X = Xg;
grid.Y = Yg;
grid.x_vec = xv;
grid.y_vec = yv;
grid.Z_true = Z_true_grid;
grid.Z_meas = Z_meas_grid;
grid.Z_cubic_true = nan(size(Xg));
grid.Z_residual_true = nan(size(Xg));
end

function model = fit_poly_plus_phs(x, y, z, fit_opts)
x = x(:);
y = y(:);
z = z(:);

polyModel = fit_cubic_poly_model(x, y, z, fit_opts.normalize_poly);
z_poly = eval_poly_model(polyModel, x, y);

residual = z - z_poly;

use_phs = true;
if fit_opts.skip_phs_if_small
    if calc_metrics(residual).rmse < fit_opts.residual_tol
        use_phs = false;
    end
end

if use_phs
    phsModel = fit_phs_model(x, y, residual, fit_opts);
else
    phsModel = empty_phs_model(fit_opts.kernel);
end

model = struct();
model.polyModel = polyModel;
model.phsModel = phsModel;
end

function polyModel = fit_cubic_poly_model(x, y, z, do_normalize)
x = x(:);
y = y(:);
z = z(:);

if do_normalize
    x_mu = mean(x);
    y_mu = mean(y);
    x_s = max(std(x), eps);
    y_s = max(std(y), eps);
    xn = (x - x_mu) ./ x_s;
    yn = (y - y_mu) ./ y_s;
else
    x_mu = 0; y_mu = 0; x_s = 1; y_s = 1;
    xn = x; yn = y;
end

P = cubic_design_matrix(xn, yn);
beta = P \ z;

polyModel = struct();
polyModel.beta = beta;
polyModel.normalize = do_normalize;
polyModel.x_mu = x_mu;
polyModel.y_mu = y_mu;
polyModel.x_s = x_s;
polyModel.y_s = y_s;
end

function z = eval_poly_model(polyModel, x, y)
sz = size(x);
x = x(:);
y = y(:);

if polyModel.normalize
    xn = (x - polyModel.x_mu) ./ polyModel.x_s;
    yn = (y - polyModel.y_mu) ./ polyModel.y_s;
else
    xn = x;
    yn = y;
end

P = cubic_design_matrix(xn, yn);
z = P * polyModel.beta;
z = reshape(z, sz);
end

function P = cubic_design_matrix(x, y)
P = [ ...
    ones(size(x)), ...
    x, y, ...
    x.^2, x.*y, y.^2, ...
    x.^3, (x.^2).*y, x.*(y.^2), y.^3 ...
    ];
end

function phsModel = fit_phs_model(x, y, r, fit_opts)
N = numel(x);

if fit_opts.normalize_phs
    x_mu = mean(x);
    y_mu = mean(y);
    x_s = max(std(x), eps);
    y_s = max(std(y), eps);
    xn = (x - x_mu) ./ x_s;
    yn = (y - y_mu) ./ y_s;
else
    x_mu = 0; y_mu = 0; x_s = 1; y_s = 1;
    xn = x; yn = y;
end

if N <= fit_opts.max_centers
    idx_center = (1:N).';
else
    idx_center = round(linspace(1, N, fit_opts.max_centers)).';
end

xc = xn(idx_center);
yc = yn(idx_center);

A = phs_kernel_matrix(xn, yn, xc, yc, fit_opts.kernel);
M = size(A, 2);

lambda = fit_opts.lambda;
w = (A.' * A + lambda * eye(M)) \ (A.' * r);

phsModel = struct();
phsModel.use_phs = true;
phsModel.kernel = fit_opts.kernel;
phsModel.weights = w;
phsModel.xc = xc;
phsModel.yc = yc;
phsModel.normalize = fit_opts.normalize_phs;
phsModel.x_mu = x_mu;
phsModel.y_mu = y_mu;
phsModel.x_s = x_s;
phsModel.y_s = y_s;
end

function phsModel = empty_phs_model(kernel_name)
phsModel = struct();
phsModel.use_phs = false;
phsModel.kernel = kernel_name;
phsModel.weights = [];
phsModel.xc = [];
phsModel.yc = [];
phsModel.normalize = false;
phsModel.x_mu = 0;
phsModel.y_mu = 0;
phsModel.x_s = 1;
phsModel.y_s = 1;
end

function z_phs = eval_phs_residual_model(phsModel, x, y)
sz = size(x);
x = x(:);
y = y(:);

if ~phsModel.use_phs
    z_phs = zeros(size(x));
    z_phs = reshape(z_phs, sz);
    return;
end

if phsModel.normalize
    xn = (x - phsModel.x_mu) ./ phsModel.x_s;
    yn = (y - phsModel.y_mu) ./ phsModel.y_s;
else
    xn = x;
    yn = y;
end

A = phs_kernel_matrix(xn, yn, phsModel.xc, phsModel.yc, phsModel.kernel);
z_phs = A * phsModel.weights;
z_phs = reshape(z_phs, sz);
end

function z = eval_poly_plus_phs(model, x, y)
sz = size(x);
xv = x(:);
yv = y(:);

z_poly = eval_poly_model(model.polyModel, xv, yv);
z_phs  = eval_phs_residual_model(model.phsModel, xv, yv);

z = z_poly + z_phs(:);
z = reshape(z, sz);
end

function A = phs_kernel_matrix(x, y, xc, yc, kernel_name)
dx = x - xc.';
dy = y - yc.';
rr = sqrt(dx.^2 + dy.^2);

switch lower(kernel_name)
    case 'r4logr'
        A = phs_r4logr(rr);
    case 'r6logr'
        A = phs_r6logr(rr);
    otherwise
        error('未知 PHS 核函数: %s', kernel_name);
end
end

function v = phs_r4logr(r)
v = zeros(size(r));
mask = r > 0;
v(mask) = (r(mask).^4) .* log(r(mask));
end

function v = phs_r6logr(r)
v = zeros(size(r));
mask = r > 0;
v(mask) = (r(mask).^6) .* log(r(mask));
end

function metrics = calc_metrics(residual)
residual = residual(:);
residual = residual(isfinite(residual));

if isempty(residual)
    metrics = struct('rmse', nan, 'mae', nan, 'maxabs', nan, 'mean', nan, 'std', nan);
    return;
end

metrics = struct();
metrics.rmse   = sqrt(mean(residual.^2));
metrics.mae    = mean(abs(residual));
metrics.maxabs = max(abs(residual));
metrics.mean   = mean(residual);
metrics.std    = std(residual);
end

function val = uniform_rand(range2)
val = range2(1) + (range2(2) - range2(1)) * rand();
end