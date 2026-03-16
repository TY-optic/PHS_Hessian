%% step4_3D_reconstruction.m
% 基于 Step3 的二维粗配准结果，使用“连续参考曲面 + 法向残差”
% 进行完整 6-DOF 三维精配准，并最终整合为连续曲面模型。
%
% 主要流程：
% 1) 读取 Step1 / Step3 数据
% 2) 以 model_sub1 作为连续参考曲面 z = f1(x,y)
% 3) 用 Step3 的 (theta, tx, ty) 构造 coarse 初值，并由连续曲面估计 tz0
% 4) 基于法向残差做小量线性修正，恢复 Rx, Ry, Rz, tx, ty, tz 的增量
% 5) 用 lsqnonlin 在连续参考曲面上做完整 6-DOF 非线性优化
% 6) 用最终配准结果整合 sub1 / sub2 点云，并拟合联合连续曲面
%
% 说明：
% - sub1 固定在 global 坐标系中，作为连续参考曲面
% - sub2 位于扰动后的 local 坐标系中
% - Step3 给出的 (theta, tx, ty) 已经很准确，Step4 重点恢复 Rx, Ry, tz
% - 不再使用通用 ICP 最近邻点对点/点到平面配准

clear; clc; close all;

%% ========================= 1. 读取数据 =========================
data_step1 = 'phs_freeform_fit_data_generalized.mat';
data_step3 = 'fft_registration_results_generalized.mat';

if ~isfile(data_step1)
    error('找不到 %s，请先运行 Step1。', data_step1);
end
if ~isfile(data_step3)
    error('找不到 %s，请先运行 Step3。', data_step3);
end

S1 = load(data_step1);
S3 = load(data_step3);

required_s1 = { ...
    'point_cloud_sub1', 'point_cloud_sub2', ...
    'point_cloud_sub1_true_global', 'point_cloud_sub2_true_global', ...
    'model_sub1', 'fit_opts', ...
    'sub1_grid', 'sub1_mask', ...
    'X', 'Y', 'Z_true', 'x_vec', 'y_vec', 'surface_params' ...
    };

for k = 1:numel(required_s1)
    if ~isfield(S1, required_s1{k})
        error('Step1 MAT 文件缺少变量：%s', required_s1{k});
    end
end

if ~isfield(S3, 'coarse_result')
    error('Step3 MAT 文件缺少 coarse_result。');
end

if exist('lsqnonlin', 'file') ~= 2
    error('需要 Optimization Toolbox 中的 lsqnonlin。');
end

pc1_global = S1.point_cloud_sub1;             % sub1 测量点，global frame
pc2_local  = S1.point_cloud_sub2;             % sub2 测量点，local perturbed frame

pc1_true_global = S1.point_cloud_sub1_true_global;
pc2_true_global = S1.point_cloud_sub2_true_global;

model_ref = S1.model_sub1;
fit_opts_combined = S1.fit_opts;

sub1_grid = S1.sub1_grid;
sub1_mask = logical(S1.sub1_mask);

dx = abs(S1.x_vec(2) - S1.x_vec(1));
dy = abs(S1.y_vec(2) - S1.y_vec(1));

theta0_deg = S3.coarse_result.theta_deg;
tx0 = S3.coarse_result.tx;
ty0 = S3.coarse_result.ty;

fprintf('================ Step4: continuous-surface normal-residual registration =================\n');
fprintf('Loaded Step3 coarse result:\n');
fprintf('  theta0 = %+9.4f deg\n', theta0_deg);
fprintf('  tx0    = %+9.4f mm\n', tx0);
fprintf('  ty0    = %+9.4f mm\n', ty0);

%% ========================= 2. 参数设置 =========================
opt = struct();

% 参考支撑域 / 权重
opt.support_soft_thresh = 0.05;
opt.support_candidate_thresh = 0.02;
opt.boundary_soft_width_pix = 6;

% 线性初始化
opt.linear_init_max_iter = 6;
opt.linear_init_lambda = 1e-6;
opt.linear_init_rot_step_max_deg = 0.5;
opt.linear_init_trans_step_max = 0.20;     % mm
opt.linear_init_min_valid = 300;

% 非线性优化
opt.nl_rot_bound_deg = 1.5;                % 相对线性初值的增量约束
opt.nl_trans_bound = 0.50;                 % mm
opt.outside_penalty = 5e-3;                % 离开支撑域惩罚
opt.robust_cauchy_k = 2.5;
opt.lsq_max_iter = 80;
opt.lsq_fun_tol = 1e-14;
opt.lsq_step_tol = 1e-14;
opt.lsq_opt_tol = 1e-14;

% 最终联合连续曲面拟合
fit_opts_combined.lambda = max(fit_opts_combined.lambda, 1e-8);
fit_opts_combined.max_centers = min(fit_opts_combined.max_centers, 1200);

%% ========================= 3. 构造参考支撑域与边界权重 =========================
support_map = double(sub1_mask);
boundary_weight_map = build_boundary_weight(sub1_mask, opt.boundary_soft_width_pix);

x_ref_vec = sub1_grid.x_vec;
y_ref_vec = sub1_grid.y_vec;

%% ========================= 4. coarse 初值：Step3 的 Rz, tx, ty + 连续曲面估计 tz =========================
theta0_rad = deg2rad(theta0_deg);
R0 = rotz3(theta0_rad);
t0 = [tx0; ty0; 0];

pc2_coarse_no_tz = apply_rigid_transform(pc2_local, R0, t0);

[z_ref0, fx_ref0, fy_ref0] = eval_model_surface_and_gradient(model_ref, ...
    pc2_coarse_no_tz(:,1), pc2_coarse_no_tz(:,2)); %#ok<ASGLU>

support_soft0 = interp2(x_ref_vec, y_ref_vec, support_map, ...
    pc2_coarse_no_tz(:,1), pc2_coarse_no_tz(:,2), 'linear', 0);

valid_tz0 = support_soft0 > 0.25 & isfinite(z_ref0);
if nnz(valid_tz0) < 30
    warning('用于估计 tz0 的有效点过少，放宽支撑域阈值。');
    valid_tz0 = support_soft0 > 0.05 & isfinite(z_ref0);
end

if nnz(valid_tz0) < 10
    warning('tz0 初值有效点仍不足，设为 0。');
    tz0 = 0;
else
    dz0 = z_ref0(valid_tz0) - pc2_coarse_no_tz(valid_tz0,3);
    tz0 = median(dz0, 'omitnan');
end

t0(3) = tz0;

fprintf('Estimated tz0 from continuous reference surface = %+9.6f mm\n', tz0);

pc2_coarse = apply_rigid_transform(pc2_local, R0, t0);

%% ========================= 5. 固定候选重叠点集合 =========================
support_soft_cand = interp2(x_ref_vec, y_ref_vec, support_map, ...
    pc2_coarse(:,1), pc2_coarse(:,2), 'linear', 0);

candidate_idx = find(support_soft_cand > opt.support_candidate_thresh);

if numel(candidate_idx) < 500
    warning('候选重叠点偏少，自动放宽阈值。');
    candidate_idx = find(support_soft_cand > 0);
end

if numel(candidate_idx) < 100
    error('有效候选重叠点过少，无法进行 Step4。');
end

P2_cand_local = pc2_local(candidate_idx, :);

fprintf('Candidate overlap points used for refinement = %d\n', size(P2_cand_local,1));

%% ========================= 6. coarse 初值评估 =========================
coarse_eval = evaluate_alignment_to_truth( ...
    pc1_global, pc2_coarse, ...
    pc1_true_global, pc2_true_global, ...
    S1.X, S1.Y, S1.Z_true);

fprintf('---------------- Coarse evaluation ----------------\n');
fprintf('Sub1 surface RMSE (fixed target)  = %.6e mm\n', coarse_eval.rmse_sub1_surface);
fprintf('Sub2 surface RMSE (coarse aligned)= %.6e mm\n', coarse_eval.rmse_sub2_surface);
fprintf('Combined surface RMSE             = %.6e mm\n', coarse_eval.rmse_combined_surface);
fprintf('Sub2 pointwise 3D RMSE            = %.6e mm\n', coarse_eval.rmse_sub2_point3d);
fprintf('Sub2 pointwise Z  RMSE            = %.6e mm\n', coarse_eval.rmse_sub2_pointz);

%% ========================= 7. 小量线性修正：法向残差 Gauss-Newton 初始化 =========================
R_lin = R0;
t_lin = t0;
linear_history = nan(opt.linear_init_max_iter, 4); % [rmse, med, rot_step_deg, trans_step]

disp('---------------- Running linear normal-residual initialization ----------------');
for it = 1:opt.linear_init_max_iter
    [res_pack, J, valid_mask] = build_linearized_normal_system( ...
        R_lin, t_lin, P2_cand_local, ...
        model_ref, x_ref_vec, y_ref_vec, support_map, boundary_weight_map, opt);

    if nnz(valid_mask) < opt.linear_init_min_valid
        warning('线性初始化第 %d 次迭代有效点过少，提前停止。', it);
        break;
    end

    r = res_pack.residual(valid_mask);
    A = J(valid_mask, :);
    w = res_pack.sqrt_weight(valid_mask);

    Aw = A .* w;
    rw = r .* w;

    H = Aw.' * Aw + opt.linear_init_lambda * eye(6);
    g = Aw.' * rw;
    delta = -H \ g;

    % 步长限制
    rot_step_norm = norm(delta(1:3));
    trans_step_norm = norm(delta(4:6));

    rot_step_max = deg2rad(opt.linear_init_rot_step_max_deg);
    if rot_step_norm > rot_step_max
        delta(1:3) = delta(1:3) * (rot_step_max / rot_step_norm);
        rot_step_norm = rot_step_max;
    end

    if trans_step_norm > opt.linear_init_trans_step_max
        delta(4:6) = delta(4:6) * (opt.linear_init_trans_step_max / trans_step_norm);
        trans_step_norm = opt.linear_init_trans_step_max;
    end

    [R_lin, t_lin] = left_update_pose(R_lin, t_lin, delta);

    linear_history(it,1) = sqrt(mean(r.^2, 'omitnan'));
    linear_history(it,2) = median(abs(r), 'omitnan');
    linear_history(it,3) = rad2deg(rot_step_norm);
    linear_history(it,4) = trans_step_norm;

    fprintf('Iter %d | residual RMSE = %.6e | median = %.6e | rot step = %.6f deg | trans step = %.6e mm\n', ...
        it, linear_history(it,1), linear_history(it,2), linear_history(it,3), linear_history(it,4));

    if rad2deg(rot_step_norm) < 1e-4 && trans_step_norm < 1e-6
        break;
    end
end

pc2_linear = apply_rigid_transform(pc2_local, R_lin, t_lin);

linear_eval = evaluate_alignment_to_truth( ...
    pc1_global, pc2_linear, ...
    pc1_true_global, pc2_true_global, ...
    S1.X, S1.Y, S1.Z_true);

fprintf('---------------- Linear-init evaluation ----------------\n');
fprintf('Sub2 surface RMSE (linear init) = %.6e mm\n', linear_eval.rmse_sub2_surface);
fprintf('Combined surface RMSE           = %.6e mm\n', linear_eval.rmse_combined_surface);
fprintf('Sub2 pointwise 3D RMSE          = %.6e mm\n', linear_eval.rmse_sub2_point3d);
fprintf('Sub2 pointwise Z  RMSE          = %.6e mm\n', linear_eval.rmse_sub2_pointz);

%% ========================= 8. 非线性精修：连续曲面法向残差 + lsqnonlin =========================
disp('---------------- Running nonlinear refinement (lsqnonlin) ----------------');

lb = [-deg2rad(opt.nl_rot_bound_deg)*ones(3,1); -opt.nl_trans_bound*ones(3,1)];
ub = [+deg2rad(opt.nl_rot_bound_deg)*ones(3,1); +opt.nl_trans_bound*ones(3,1)];
x0 = zeros(6,1);

resfun = @(xi) build_nonlinear_residual_vector( ...
    xi, R_lin, t_lin, P2_cand_local, ...
    model_ref, x_ref_vec, y_ref_vec, support_map, boundary_weight_map, opt);

lsq_options = optimoptions('lsqnonlin', ...
    'Display', 'iter', ...
    'MaxIterations', opt.lsq_max_iter, ...
    'FunctionTolerance', opt.lsq_fun_tol, ...
    'StepTolerance', opt.lsq_step_tol, ...
    'OptimalityTolerance', opt.lsq_opt_tol);

[x_opt, resnorm_opt, residual_opt, exitflag_opt, output_opt] = lsqnonlin( ...
    resfun, x0, lb, ub, lsq_options);

[R_final, t_final] = compose_incremental_pose(R_lin, t_lin, x_opt);
pc2_final = apply_rigid_transform(pc2_local, R_final, t_final);

fprintf('lsqnonlin exitflag = %d\n', exitflag_opt);
fprintf('lsqnonlin resnorm  = %.6e\n', resnorm_opt);
fprintf('lsqnonlin iterations = %d\n', output_opt.iterations);

%% ========================= 9. 最终结果评估 =========================
final_eval = evaluate_alignment_to_truth( ...
    pc1_global, pc2_final, ...
    pc1_true_global, pc2_true_global, ...
    S1.X, S1.Y, S1.Z_true);

fprintf('---------------- Final evaluation ----------------\n');
fprintf('Sub1 surface RMSE (fixed target) = %.6e mm\n', final_eval.rmse_sub1_surface);
fprintf('Sub2 surface RMSE (final)        = %.6e mm\n', final_eval.rmse_sub2_surface);
fprintf('Combined surface RMSE            = %.6e mm\n', final_eval.rmse_combined_surface);
fprintf('Sub2 pointwise 3D RMSE           = %.6e mm\n', final_eval.rmse_sub2_point3d);
fprintf('Sub2 pointwise Z  RMSE           = %.6e mm\n', final_eval.rmse_sub2_pointz);

%% ========================= 10. 真值位姿诊断 =========================
has_gt_pose = isfield(S1, 'sub2_pose_gt');

if has_gt_pose
    gt_pose = S1.sub2_pose_gt;
    R_gt = gt_pose.R_local_to_global;
    t_gt = gt_pose.anchor_global(:) - R_gt * gt_pose.t_local_from_anchor(:);

    coarse_rot_err_deg = rotation_angle_error_deg(R0, R_gt);
    coarse_trans_err   = norm(t0 - t_gt);

    linear_rot_err_deg = rotation_angle_error_deg(R_lin, R_gt);
    linear_trans_err   = norm(t_lin - t_gt);

    final_rot_err_deg  = rotation_angle_error_deg(R_final, R_gt);
    final_trans_err    = norm(t_final - t_gt);

    fprintf('---------------- Against GT local->global transform ----------------\n');
    fprintf('Coarse: rot err = %.6f deg | trans err = %.6e mm\n', coarse_rot_err_deg, coarse_trans_err);
    fprintf('Linear: rot err = %.6f deg | trans err = %.6e mm\n', linear_rot_err_deg, linear_trans_err);
    fprintf('Final : rot err = %.6f deg | trans err = %.6e mm\n', final_rot_err_deg, final_trans_err);
else
    R_gt = nan(3); t_gt = nan(3,1);
    coarse_rot_err_deg = nan; coarse_trans_err = nan;
    linear_rot_err_deg = nan; linear_trans_err = nan;
    final_rot_err_deg  = nan; final_trans_err  = nan;
end

%% ========================= 11. 重叠区域法向残差统计 =========================
res_final_pack = build_residual_diagnostics( ...
    R_final, t_final, P2_cand_local, ...
    model_ref, x_ref_vec, y_ref_vec, support_map, boundary_weight_map, opt);

fprintf('---------------- Final overlap diagnostics ----------------\n');
fprintf('Valid overlap points     = %d\n', nnz(res_final_pack.valid_mask));
fprintf('Normal residual RMSE     = %.6e mm\n', sqrt(mean(res_final_pack.normal_residual(res_final_pack.valid_mask).^2, 'omitnan')));
fprintf('Normal residual median   = %.6e mm\n', median(abs(res_final_pack.normal_residual(res_final_pack.valid_mask)), 'omitnan'));
fprintf('Vertical residual RMSE   = %.6e mm\n', sqrt(mean(res_final_pack.vertical_residual(res_final_pack.valid_mask).^2, 'omitnan')));
fprintf('Vertical residual median = %.6e mm\n', median(abs(res_final_pack.vertical_residual(res_final_pack.valid_mask)), 'omitnan'));


%% ========================= 12. 保存 Step4（仅配准结果） =========================
result_step4 = struct();

result_step4.coarse.R = R0;
result_step4.coarse.t = t0;
result_step4.coarse.eval = coarse_eval;

result_step4.linear.R = R_lin;
result_step4.linear.t = t_lin;
result_step4.linear.eval = linear_eval;
result_step4.linear.history = linear_history;

result_step4.final.R = R_final;
result_step4.final.t = t_final;
result_step4.final.eval = final_eval;

result_step4.nonlinear.x_opt = x_opt;
result_step4.nonlinear.resnorm = resnorm_opt;
result_step4.nonlinear.exitflag = exitflag_opt;
result_step4.nonlinear.output = output_opt;
result_step4.nonlinear.residual = residual_opt;

result_step4.overlap.valid_count = nnz(res_final_pack.valid_mask);
result_step4.overlap.normal_rmse = sqrt(mean(res_final_pack.normal_residual(res_final_pack.valid_mask).^2, 'omitnan'));
result_step4.overlap.normal_median = median(abs(res_final_pack.normal_residual(res_final_pack.valid_mask)), 'omitnan');
result_step4.overlap.vertical_rmse = sqrt(mean(res_final_pack.vertical_residual(res_final_pack.valid_mask).^2, 'omitnan'));
result_step4.overlap.vertical_median = median(abs(res_final_pack.vertical_residual(res_final_pack.valid_mask)), 'omitnan');

result_step4.options = opt;

if has_gt_pose
    result_step4.gt.R = R_gt;
    result_step4.gt.t = t_gt;
    result_step4.gt.coarse_rot_err_deg = coarse_rot_err_deg;
    result_step4.gt.coarse_trans_err = coarse_trans_err;
    result_step4.gt.linear_rot_err_deg = linear_rot_err_deg;
    result_step4.gt.linear_trans_err = linear_trans_err;
    result_step4.gt.final_rot_err_deg = final_rot_err_deg;
    result_step4.gt.final_trans_err = final_trans_err;
end

save_name = 'step4_registration_only_results_generalized.mat';
save(save_name, ...
    'result_step4', ...
    'pc1_global', 'pc2_local', 'pc2_coarse', 'pc2_linear', 'pc2_final', ...
    'pc1_true_global', 'pc2_true_global', ...
    'model_ref', ...
    'sub1_grid', 'sub1_mask', ...
    'candidate_idx', 'P2_cand_local', ...
    'res_final_pack', ...
    'S1', ...
    '-v7.3');

disp(['Saved Step4 registration-only result file: ', save_name]);



%% ========================= 局部函数 =========================

function R = rotz3(theta)
c = cos(theta); s = sin(theta);
R = [c, -s, 0; s, c, 0; 0, 0, 1];
end

function P_out = apply_rigid_transform(P_in, R, t)
P_out = (R * P_in.').';
P_out = P_out + t(:).';
end

function [R_new, t_new] = left_update_pose(R, t, delta)
dw = delta(1:3);
dt = delta(4:6);
Rd = expSO3(dw);
R_new = Rd * R;
t_new = Rd * t + dt;
end

function [R_new, t_new] = compose_incremental_pose(R_ref, t_ref, xi)
[R_new, t_new] = left_update_pose(R_ref, t_ref, xi);
end

function R = expSO3(w)
theta = norm(w);
if theta < 1e-15
    R = eye(3) + skew3(w);
    return;
end
k = w / theta;
K = skew3(k);
R = eye(3) + sin(theta) * K + (1 - cos(theta)) * (K * K);
end

function S = skew3(v)
S = [   0   -v(3)  v(2); ...
      v(3)    0   -v(1); ...
     -v(2)  v(1)    0 ];
end

function [pack, J, valid_mask] = build_linearized_normal_system(R, t, P_local, model_ref, x_ref_vec, y_ref_vec, support_map, boundary_weight_map, opt)
P = apply_rigid_transform(P_local, R, t);

[x, y, z] = deal(P(:,1), P(:,2), P(:,3));

[z_ref, fx_ref, fy_ref] = eval_model_surface_and_gradient(model_ref, x, y);

support_soft = interp2(x_ref_vec, y_ref_vec, support_map, x, y, 'linear', 0);
boundary_w = interp2(x_ref_vec, y_ref_vec, boundary_weight_map, x, y, 'linear', 0);

valid_mask = support_soft > opt.support_soft_thresh & ...
    isfinite(z_ref) & isfinite(fx_ref) & isfinite(fy_ref);

n = zeros(size(P));
n(:,1) = -fx_ref;
n(:,2) = -fy_ref;
n(:,3) = 1;
n_norm = sqrt(sum(n.^2, 2)) + eps;
n = n ./ n_norm;

Q = [x, y, z_ref];
r = sum(n .* (P - Q), 2);

geom_w = max(boundary_w, 0) .* max(support_soft, 0);
geom_w(~valid_mask) = 0;

% 鲁棒权
rv = r(valid_mask);
sigma = robust_scale(rv);
robust_w = ones(size(r));
robust_w(valid_mask) = 1 ./ sqrt(1 + (rv ./ (opt.robust_cauchy_k * sigma)).^2);

sqrt_w = sqrt(max(geom_w, 0)) .* robust_w;

% Jacobian: delta r = n^T (dw x p + dt) = (p x n)^T dw + n^T dt
pxn = cross(P, n, 2);
J = [pxn, n];

pack = struct();
pack.residual = r;
pack.sqrt_weight = sqrt_w;
pack.support_soft = support_soft;
pack.boundary_w = boundary_w;
pack.normal = n;
end

function rvec = build_nonlinear_residual_vector(xi, R_ref, t_ref, P_local, model_ref, x_ref_vec, y_ref_vec, support_map, boundary_weight_map, opt)
[R, t] = compose_incremental_pose(R_ref, t_ref, xi);
diag_pack = build_residual_diagnostics(R, t, P_local, model_ref, x_ref_vec, y_ref_vec, support_map, boundary_weight_map, opt);

rn = diag_pack.normal_residual;
rv = diag_pack.vertical_residual;
support_soft = diag_pack.support_soft;
boundary_w = diag_pack.boundary_w;
valid = diag_pack.valid_mask;

geom_w = max(boundary_w, 0) .* max(support_soft, 0).^2;
geom_w(~valid) = 0;

sigma = robust_scale(rn(valid));
robust_w = ones(size(rn));
if any(valid)
    robust_w(valid) = 1 ./ sqrt(1 + (rn(valid) ./ (opt.robust_cauchy_k * sigma)).^2);
end

r_normal = sqrt(geom_w) .* robust_w .* rn;

% 离开支撑域惩罚
r_out = opt.outside_penalty * max(0, opt.support_soft_thresh - support_soft);

% 轻微垂直残差辅助约束（防止法向近退化区域过软）
r_vert = 0.20 * sqrt(geom_w) .* robust_w .* rv;

rvec = [r_normal; r_vert; r_out];
rvec(~isfinite(rvec)) = 0;
end

function pack = build_residual_diagnostics(R, t, P_local, model_ref, x_ref_vec, y_ref_vec, support_map, boundary_weight_map, opt)
P = apply_rigid_transform(P_local, R, t);

x = P(:,1);
y = P(:,2);
z = P(:,3);

[z_ref, fx_ref, fy_ref] = eval_model_surface_and_gradient(model_ref, x, y);

support_soft = interp2(x_ref_vec, y_ref_vec, support_map, x, y, 'linear', 0);
boundary_w = interp2(x_ref_vec, y_ref_vec, boundary_weight_map, x, y, 'linear', 0);

valid_mask = support_soft > opt.support_soft_thresh & ...
    isfinite(z_ref) & isfinite(fx_ref) & isfinite(fy_ref);

n = zeros(size(P));
n(:,1) = -fx_ref;
n(:,2) = -fy_ref;
n(:,3) = 1;
n_norm = sqrt(sum(n.^2, 2)) + eps;
n = n ./ n_norm;

Q = [x, y, z_ref];

vertical_residual = z - z_ref;
normal_residual = sum(n .* (P - Q), 2);

pack = struct();
pack.P = P;
pack.Q = Q;
pack.normal = n;
pack.support_soft = support_soft;
pack.boundary_w = boundary_w;
pack.valid_mask = valid_mask;
pack.vertical_residual = vertical_residual;
pack.normal_residual = normal_residual;
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

function s = robust_scale(v)
v = v(isfinite(v));
if isempty(v)
    s = 1;
    return;
end
medv = median(v);
madv = median(abs(v - medv));
s = 1.4826 * madv + eps;
end

function [z, fx, fy] = eval_model_surface_and_gradient(model, x, y)
z_poly = eval_poly_model(model.polyModel, x, y);
[fx_poly, fy_poly] = eval_poly_gradient(model.polyModel, x, y);

z_phs = eval_phs_residual_model(model.phsModel, x, y);
[fx_phs, fy_phs] = eval_phs_gradient(model.phsModel, x, y);

z = z_poly + z_phs;
fx = fx_poly + fx_phs;
fy = fy_poly + fy_phs;
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

function [fx, fy] = eval_poly_gradient(polyModel, x, y)
sz = size(x);
x = x(:);
y = y(:);

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

fx_n = b(2) + 2*b(4).*xn + b(5).*yn + 3*b(7).*xn.^2 + 2*b(8).*xn.*yn + b(9).*yn.^2;
fy_n = b(3) + b(5).*xn + 2*b(6).*yn + b(8).*xn.^2 + 2*b(9).*xn.*yn + 3*b(10).*yn.^2;

fx = reshape(fx_n ./ xs, sz);
fy = reshape(fy_n ./ ys, sz);
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

function [fx, fy] = eval_phs_gradient(phsModel, x, y)
sz = size(x);
x = x(:);
y = y(:);

if ~phsModel.use_phs || isempty(phsModel.weights)
    fx = zeros(sz);
    fy = zeros(sz);
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
w = phsModel.weights(:);

dx = xn - xc;
dy = yn - yc;
s = dx.^2 + dy.^2;

Gx = zeros(size(s));
Gy = zeros(size(s));

mask = s > 0;

switch lower(phsModel.kernel)
    case 'r4logr'
        ss = s(mask);
        ddx = dx(mask);
        ddy = dy(mask);
        c = 2 * log(ss) + 1;
        Gx(mask) = ddx .* ss .* c;
        Gy(mask) = ddy .* ss .* c;

    case 'r6logr'
        ss = s(mask);
        ddx = dx(mask);
        ddy = dy(mask);
        c = 3 * log(ss) + 1;
        Gx(mask) = ddx .* (ss.^2) .* c;
        Gy(mask) = ddy .* (ss.^2) .* c;

    otherwise
        error('未知 PHS 核函数: %s', phsModel.kernel);
end

fx = reshape(Gx * w, sz) ./ xs;
fy = reshape(Gy * w, sz) ./ ys;
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

function P = cubic_design_matrix(x, y)
P = [ ...
    ones(size(x)), ...
    x, y, ...
    x.^2, x.*y, y.^2, ...
    x.^3, (x.^2).*y, x.*(y.^2), y.^3 ...
    ];
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
x = x(:); y = y(:); z = z(:);

if do_normalize
    x_mu = mean(x); y_mu = mean(y);
    x_s = max(std(x), eps); y_s = max(std(y), eps);
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

function phsModel = fit_phs_model(x, y, r, fit_opts)
N = numel(x);

if fit_opts.normalize_phs
    x_mu = mean(x); y_mu = mean(y);
    x_s = max(std(x), eps); y_s = max(std(y), eps);
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

function z = eval_poly_plus_phs(model, x, y)
sz = size(x);
xv = x(:);
yv = y(:);

z_poly = eval_poly_model(model.polyModel, xv, yv);
z_phs  = eval_phs_residual_model(model.phsModel, xv, yv);

z = z_poly + z_phs(:);
z = reshape(z, sz);
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

function err_z = eval_surface_error(P, X, Y, Z)
err_z = nan(size(P,1),1);
z_ref = interp2(X, Y, Z, P(:,1), P(:,2), 'spline', nan);
valid = isfinite(z_ref);
err_z(valid) = P(valid,3) - z_ref(valid);
end

function ev = evaluate_alignment_to_truth(pc1_global, pc2_global, pc1_true_global, pc2_true_global, X, Y, Z_true)
err1_surf = eval_surface_error(pc1_global, X, Y, Z_true);
err2_surf = eval_surface_error(pc2_global, X, Y, Z_true);

ev = struct();
ev.rmse_sub1_surface = calc_rmse(err1_surf);
ev.rmse_sub2_surface = calc_rmse(err2_surf);
ev.rmse_combined_surface = calc_rmse([err1_surf; err2_surf]);

if size(pc2_global,1) == size(pc2_true_global,1)
    dxyz2 = pc2_global - pc2_true_global;
    ev.rmse_sub2_point3d = sqrt(mean(sum(dxyz2.^2,2), 'omitnan'));
    ev.rmse_sub2_pointz = sqrt(mean((pc2_global(:,3) - pc2_true_global(:,3)).^2, 'omitnan'));
else
    ev.rmse_sub2_point3d = nan;
    ev.rmse_sub2_pointz = nan;
end

if size(pc1_global,1) == size(pc1_true_global,1)
    dxyz1 = pc1_global - pc1_true_global;
    ev.rmse_sub1_point3d = sqrt(mean(sum(dxyz1.^2,2), 'omitnan'));
    ev.rmse_sub1_pointz = sqrt(mean((pc1_global(:,3) - pc1_true_global(:,3)).^2, 'omitnan'));
else
    ev.rmse_sub1_point3d = nan;
    ev.rmse_sub1_pointz = nan;
end
end

function v = calc_rmse(x)
x = x(isfinite(x));
if isempty(x)
    v = nan;
else
    v = sqrt(mean(x.^2));
end
end

function [R, t] = estimate_rigid_3d(Xsrc, Xdst)
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
end

function ang_deg = rotation_angle_error_deg(R_est, R_gt)
R_rel = R_est * R_gt.';
tr = (trace(R_rel) - 1) / 2;
tr = max(-1, min(1, tr));
ang_deg = rad2deg(acos(tr));
end