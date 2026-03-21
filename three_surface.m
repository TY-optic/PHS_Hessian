%% =========================================================================
%  generate_three_surfaces_for_stitching_fixed_plot.m
%  修正版：解决 F1/F3 点云为空、局部坐标图完全重合的问题
%% =========================================================================

clear; clc; close all;

%% ======================== 全局参数 =======================================
ds               = 1;      % 采样间隔 (mm)
overlap_target   = 0.50;     % 目标重叠比
rot_deg          = 2.0;      % B 相对 A 的真实面内旋转角 (deg)
theta_true       = deg2rad(rot_deg);

ds_overlap_eval  = 0.05;     % 重叠比数值评估步长

fprintf('============================================================\n');
fprintf('统一设置: ds = %.3f mm, target overlap = %.1f%%, rot = %.2f deg\n', ...
        ds, 100*overlap_target, rot_deg);
fprintf('ground truth 统一定义: p_A = R_true * p_B + t_true\n');
fprintf('============================================================\n\n');

%% ======================== F1: peaks 自由曲面 ============================
fprintf('===== F1: peaks 自由曲面 =====\n');

D_f1    = 100;
Ap      = 0.050;
Rsub1   = 30;

xv1 = -D_f1/2 : ds : D_f1/2;
yv1 = -D_f1/2 : ds : D_f1/2;
[Xm1, Ym1] = meshgrid(xv1, yv1);
Rm1 = hypot(Xm1, Ym1);
mask_full1 = (Rm1 <= D_f1/2);

Xbar1 = 6 * Xm1 / D_f1;
Ybar1 = 6 * Ym1 / D_f1;
Zm1   = Ap * peaks(Xbar1, Ybar1);
Zm1(~mask_full1) = NaN;

% 关键修正：圆口径面型不能再用 griddedInterpolant + NaN 掩膜
valid1 = mask_full1 & isfinite(Zm1);
F1_interp = scatteredInterpolant(Xm1(valid1), Ym1(valid1), Zm1(valid1), ...
                                 'natural', 'none');

sep1 = solve_circle_sep_from_overlap(Rsub1, overlap_target);

phi1 = deg2rad(25);
u1   = [cos(phi1); sin(phi1)];
c0_1 = [0; 0];
cA1  = c0_1 - 0.5 * sep1 * u1;
cB1  = c0_1 + 0.5 * sep1 * u1;

subap_f1_A = sample_circular_subap_local(F1_interp, cA1, 0,          Rsub1, ds, 'F1_A');
subap_f1_B = sample_circular_subap_local(F1_interp, cB1, theta_true, Rsub1, ds, 'F1_B');

gt_f1 = make_ground_truth(cA1, cB1, theta_true, 'F1');

eta1 = 100 * circle_overlap_ratio(norm(cB1 - cA1), Rsub1);

z_valid1 = Zm1(valid1);
pv_f1    = max(z_valid1) - min(z_valid1);
rms_f1   = std(z_valid1, 1);

fprintf('F1 全口径 PV = %.6f mm, RMS = %.6f mm\n', pv_f1, rms_f1);
fprintf('F1 子孔径中心距 = %.6f mm, 实际重叠比 = %.3f %%\n', sep1, eta1);
fprintf('F1 子孔径点数: A = %d, B = %d\n', numel(subap_f1_A.x), numel(subap_f1_B.x));
fprintf('F1 ground truth: tx = %.6f mm, ty = %.6f mm, theta = %.6f deg\n\n', ...
        gt_f1.t(1), gt_f1.t(2), gt_f1.theta_deg);

figure('Name', 'F1: peaks 自由曲面', 'Position', [80 80 1400 450]);

subplot(1,3,1);
surf(Xm1, Ym1, Zm1, 'EdgeColor', 'none'); hold on;
[bxA1, byA1] = circle_boundary_world(cA1, Rsub1, 0);
[bxB1, byB1] = circle_boundary_world(cB1, Rsub1, theta_true);
plot3(bxA1, byA1, max(z_valid1)*ones(size(bxA1)), 'r-', 'LineWidth', 2);
plot3(bxB1, byB1, max(z_valid1)*ones(size(bxB1)), 'b-', 'LineWidth', 2);
axis equal tight; colorbar;
xlabel('x (mm)'); ylabel('y (mm)'); zlabel('z (mm)');
title('母面型与子孔径位置');
legend('母面型', '子孔径 A', '子孔径 B', 'Location', 'best');
view(30,40);

subplot(1,3,2);
scatter3(subap_f1_A.xw, subap_f1_A.yw, subap_f1_A.z, 4, subap_f1_A.z, 'filled'); hold on;
scatter3(subap_f1_B.xw, subap_f1_B.yw, subap_f1_B.z, 4, subap_f1_B.z, 'filled');
axis equal tight; colorbar;
xlabel('x (mm)'); ylabel('y (mm)'); zlabel('z (mm)');
title('两子孔径点云（世界坐标）');
legend('A', 'B', 'Location', 'best');
view(30,40);

subplot(1,3,3);
XYB_in_A_1 = gt_f1.R * [subap_f1_B.x.'; subap_f1_B.y.'] + gt_f1.t(:);
plot(subap_f1_A.x, subap_f1_A.y, 'r.', 'MarkerSize', 4); hold on;
plot(XYB_in_A_1(1,:), XYB_in_A_1(2,:), 'b.', 'MarkerSize', 4);
axis equal tight; grid on;
xlabel('x_A (mm)'); ylabel('y_A (mm)');
title('B 映射到 A 局部坐标');
legend('A-local', 'B \rightarrow A', 'Location', 'best');

sgtitle('F1: peaks 自由曲面');


%% ======================== F2: x-y 多项式矩形自由曲面 ====================
fprintf('===== F2: x-y 多项式矩形自由曲面 =====\n');

Lx_f2   = 120;
Ly_f2   = 80;
Lsub_x  = 60;
Lsub_y  = 60;
dy_f2   = 4.0;

coeff = struct();
coeff.a20 = -2.50e-2;
coeff.a02 = -3.80e-2;
coeff.a30 =  1.20e-3;
coeff.a12 = -8.50e-4;
coeff.a21 =  6.00e-4;
coeff.a40 =  5.60e-4;
coeff.a22 = -1.10e-3;
coeff.a04 =  7.20e-4;
coeff.a31 = -2.20e-4;
coeff.a13 =  1.90e-4;
coeff.a50 = -3.10e-5;
coeff.a32 =  4.50e-5;
coeff.a14 = -2.80e-5;
coeff.a05 =  1.50e-5;

xv2 = -Lx_f2/2 : ds : Lx_f2/2;
yv2 = -Ly_f2/2 : ds : Ly_f2/2;
[Xm2, Ym2] = meshgrid(xv2, yv2);

Xbar2 = 2 * Xm2 / Lx_f2;
Ybar2 = 2 * Ym2 / Ly_f2;

Zm2 = coeff.a20 * Xbar2.^2 ...
    + coeff.a02 * Ybar2.^2 ...
    + coeff.a30 * Xbar2.^3 ...
    + coeff.a12 * Xbar2 .* Ybar2.^2 ...
    + coeff.a21 * Xbar2.^2 .* Ybar2 ...
    + coeff.a40 * Xbar2.^4 ...
    + coeff.a22 * Xbar2.^2 .* Ybar2.^2 ...
    + coeff.a04 * Ybar2.^4 ...
    + coeff.a31 * Xbar2.^3 .* Ybar2 ...
    + coeff.a13 * Xbar2 .* Ybar2.^3 ...
    + coeff.a50 * Xbar2.^5 ...
    + coeff.a32 * Xbar2.^3 .* Ybar2.^2 ...
    + coeff.a14 * Xbar2 .* Ybar2.^4 ...
    + coeff.a05 * Ybar2.^5;

% 统一改成 scatteredInterpolant，接口保持一致
valid2 = isfinite(Zm2);
F2_interp = scatteredInterpolant(Xm2(valid2), Ym2(valid2), Zm2(valid2), ...
                                 'natural', 'none');

sep2x = solve_rect_dx_for_overlap(overlap_target, dy_f2, theta_true, ...
                                  Lsub_x, Lsub_y, ds_overlap_eval);

c0_2 = [0; 0];
cA2  = c0_2 + [-sep2x/2; -dy_f2/2];
cB2  = c0_2 + [ sep2x/2;  dy_f2/2];

subap_f2_A = sample_rect_subap_local(F2_interp, cA2, 0,          Lsub_x, Lsub_y, ds, 'F2_A');
subap_f2_B = sample_rect_subap_local(F2_interp, cB2, theta_true, Lsub_x, Lsub_y, ds, 'F2_B');

gt_f2 = make_ground_truth(cA2, cB2, theta_true, 'F2');

eta2 = 100 * overlap_rect_rot_numerical(sep2x, dy_f2, theta_true, ...
                                        Lsub_x, Lsub_y, ds_overlap_eval);

pv_f2  = max(Zm2(:)) - min(Zm2(:));
rms_f2 = std(Zm2(:), 1);

fprintf('F2 全口径 PV = %.6f mm, RMS = %.6f mm\n', pv_f2, rms_f2);
fprintf('F2 子孔径中心差 = (%.6f, %.6f) mm, 实际重叠比 = %.3f %%\n', sep2x, dy_f2, eta2);
fprintf('F2 子孔径点数: A = %d, B = %d\n', numel(subap_f2_A.x), numel(subap_f2_B.x));
fprintf('F2 ground truth: tx = %.6f mm, ty = %.6f mm, theta = %.6f deg\n\n', ...
        gt_f2.t(1), gt_f2.t(2), gt_f2.theta_deg);

figure('Name', 'F2: x-y 多项式矩形自由曲面', 'Position', [80 500 1400 450]);

subplot(1,3,1);
surf(Xm2, Ym2, Zm2, 'EdgeColor', 'none'); hold on;
[bxA2, byA2] = rect_boundary_world(cA2, Lsub_x, Lsub_y, 0);
[bxB2, byB2] = rect_boundary_world(cB2, Lsub_x, Lsub_y, theta_true);
zmax2 = max(Zm2(:));
plot3(bxA2, byA2, zmax2*ones(size(bxA2)), 'r-', 'LineWidth', 2);
plot3(bxB2, byB2, zmax2*ones(size(bxB2)), 'b-', 'LineWidth', 2);
axis equal tight; colorbar;
xlabel('x (mm)'); ylabel('y (mm)'); zlabel('z (mm)');
title('母面型与子孔径位置');
legend('母面型', '子孔径 A', '子孔径 B', 'Location', 'best');
view(30,40);

subplot(1,3,2);
scatter3(subap_f2_A.xw, subap_f2_A.yw, subap_f2_A.z, 4, subap_f2_A.z, 'filled'); hold on;
scatter3(subap_f2_B.xw, subap_f2_B.yw, subap_f2_B.z, 4, subap_f2_B.z, 'filled');
axis equal tight; colorbar;
xlabel('x (mm)'); ylabel('y (mm)'); zlabel('z (mm)');
title('两子孔径点云（世界坐标）');
legend('A', 'B', 'Location', 'best');
view(30,40);

subplot(1,3,3);
XYB_in_A_2 = gt_f2.R * [subap_f2_B.x.'; subap_f2_B.y.'] + gt_f2.t(:);
plot(subap_f2_A.x, subap_f2_A.y, 'r.', 'MarkerSize', 4); hold on;
plot(XYB_in_A_2(1,:), XYB_in_A_2(2,:), 'b.', 'MarkerSize', 4);
axis equal tight; grid on;
xlabel('x_A (mm)'); ylabel('y_A (mm)');
title('B 映射到 A 局部坐标');
legend('A-local', 'B \rightarrow A', 'Location', 'best');

sgtitle('F2: x-y 多项式矩形自由曲面');


%% ======================== F3: 自由曲面残差面型 ===========================
fprintf('===== F3: 自由曲面残差面型 =====\n');


D_sub_f3          = 80;        % 母口径直径 (mm)
Rsub3             = D_sub_f3/2;
Rsub_ap_f3        = 20;        % 子孔径半径 (mm)

overlap_target_f3 = 0.40;      % F3 使用更低重叠比
rot_deg_f3        = 6.0;       % F3 使用更大面内旋转角
theta_true_f3     = deg2rad(rot_deg_f3);

% --- 母面型网格 ---
xv3 = -Rsub3 : ds : Rsub3;
yv3 = -Rsub3 : ds : Rsub3;
[Xm3, Ym3] = meshgrid(xv3, yv3);
Rm3 = hypot(Xm3, Ym3);
mask_full3 = (Rm3 <= Rsub3);

rho_n = Rm3 / Rsub3;
th3   = atan2(Ym3, Xm3);

% --- Fringe Zernike 基函数 Z4--Z18 ---
Z_terms3 = fringe_zernike_Z4_Z18(rho_n, th3);


b_res_um = [ ...
     8.0,  -7.0,   6.0, ...    % Z9  Z10 Z11
   -18.0,  14.0, ...           % Z12 Z13
   -26.0,  20.0, ...           % Z14 Z15
   -34.0,  26.0, -22.0 ...     % Z16 Z17 Z18
];
b_res = b_res_um * 1e-3;       % mm

Z_res3 = zeros(size(Xm3));

% 对应 Z9--Z18 = 索引 6:15
for k = 1:10
    Z_res3 = Z_res3 + b_res(k) * Z_terms3(:,:,k+5);
end

% --- 边缘增强权重 ---
edge_weight3 = 0.25 + 0.75 * rho_n.^2.4;
edge_weight3(~mask_full3) = 0;

% --- 残差面型 ---
Zm3 = Z_res3 .* edge_weight3;
Zm3(~mask_full3) = NaN;

% --- 去除最佳拟合平面，确保为“残差面型” ---
coef_plane3 = fit_plane_ls(Xm3(mask_full3), Ym3(mask_full3), Zm3(mask_full3));
Z_plane3 = coef_plane3(1) * Xm3 + coef_plane3(2) * Ym3 + coef_plane3(3);
Zm3 = Zm3 - Z_plane3;
Zm3(~mask_full3) = NaN;

% --- 统计量 ---
z_valid3 = Zm3(mask_full3);
pv_f3    = max(z_valid3) - min(z_valid3);
rms_f3   = std(z_valid3, 1);

fprintf('F3 残差面型 PV = %.6f mm (%.2f um)\n', pv_f3, 1e3 * pv_f3);
fprintf('F3 残差面型 RMS = %.6f mm (%.2f um)\n', rms_f3, 1e3 * rms_f3);

% --- 插值器 ---
valid3 = mask_full3 & isfinite(Zm3);
F3_interp = scatteredInterpolant(Xm3(valid3), Ym3(valid3), Zm3(valid3), ...
                                 'natural', 'none');

% --- 根据更低重叠比精确反解圆形子孔径中心距 ---
sep3 = solve_circle_sep_from_overlap(Rsub_ap_f3, overlap_target_f3);

% --- 子孔径中心：整体偏离中心，形成更困难的空间分布 ---
phi3 = deg2rad(-28);
u3   = [cos(phi3); sin(phi3)];

c0_3 = [6; -4];   % 整体偏心
cA3  = c0_3 - 0.5 * sep3 * u3;
cB3  = c0_3 + 0.5 * sep3 * u3;

% --- 边界检查 ---
if norm(cA3) + Rsub_ap_f3 > Rsub3 || norm(cB3) + Rsub_ap_f3 > Rsub3
    error('F3 子孔径越界：请调整 c0_3 / Rsub_ap_f3 / overlap_target_f3。');
end

% --- 采样子孔径（局部坐标系输出） ---
subap_f3_A = sample_circular_subap_local(F3_interp, cA3, 0,              Rsub_ap_f3, ds, 'F3_A');
subap_f3_B = sample_circular_subap_local(F3_interp, cB3, theta_true_f3,  Rsub_ap_f3, ds, 'F3_B');

% --- ground truth: p_A = R_true * p_B + t_true ---
gt_f3 = make_ground_truth(cA3, cB3, theta_true_f3, 'F3');

% --- 实际重叠比 ---
eta3 = 100 * circle_overlap_ratio(norm(cB3 - cA3), Rsub_ap_f3);

fprintf('F3 子孔径中心距 = %.6f mm, 实际重叠比 = %.3f %%\n', sep3, eta3);
fprintf('F3 子孔径点数: A = %d, B = %d\n', numel(subap_f3_A.x), numel(subap_f3_B.x));
fprintf('F3 ground truth: tx = %.6f mm, ty = %.6f mm, theta = %.6f deg\n\n', ...
        gt_f3.t(1), gt_f3.t(2), gt_f3.theta_deg);

% --- 可视化 ---
figure('Name', 'F3: 自由曲面残差面型', 'Position', [200 840 1400 450]);

subplot(1,3,1);
surf(Xm3, Ym3, Zm3, 'EdgeColor', 'none'); hold on;
[bxA3, byA3] = circle_boundary_world(cA3, Rsub_ap_f3, 0);
[bxB3, byB3] = circle_boundary_world(cB3, Rsub_ap_f3, theta_true_f3);
plot3(bxA3, byA3, max(z_valid3) * ones(size(bxA3)), 'r-', 'LineWidth', 2);
plot3(bxB3, byB3, max(z_valid3) * ones(size(bxB3)), 'b-', 'LineWidth', 2);
axis equal tight; colorbar;
xlabel('x (mm)'); ylabel('y (mm)'); zlabel('z (mm)');
title('母面型与子孔径位置');
legend('母面型', '子孔径 A', '子孔径 B', 'Location', 'best');
view(30,40);

subplot(1,3,2);
scatter3(subap_f3_A.xw, subap_f3_A.yw, subap_f3_A.z, 6, subap_f3_A.z, 'filled'); hold on;
scatter3(subap_f3_B.xw, subap_f3_B.yw, subap_f3_B.z, 6, subap_f3_B.z, 'filled');
axis equal tight; colorbar;
xlabel('x (mm)'); ylabel('y (mm)'); zlabel('z (mm)');
title('两子孔径点云（世界坐标）');
legend('A', 'B', 'Location', 'best');
view(30,40);

subplot(1,3,3);
XYB_in_A_3 = gt_f3.R * [subap_f3_B.x.'; subap_f3_B.y.'] + gt_f3.t(:);
plot(subap_f3_A.x, subap_f3_A.y, 'r.', 'MarkerSize', 5); hold on;
plot(XYB_in_A_3(1,:), XYB_in_A_3(2,:), 'b.', 'MarkerSize', 5);
axis equal tight; grid on;
xlabel('x_A (mm)'); ylabel('y_A (mm)');
title('B 映射到 A 局部坐标');
legend('A-local', 'B \rightarrow A', 'Location', 'best');

sgtitle('F3: 自由曲面残差面型');


%% ======================== 汇总输出 ======================================
fprintf('======================== 面型参数汇总 ========================\n');
fprintf('%-6s %-12s %-18s %-12s %-12s %-10s\n', ...
        '面型', '口径形状', '口径尺寸(mm)', 'PV(mm)', 'RMS(mm)', '重叠比(%)');
fprintf('%-6s %-12s %-18s %-12.6f %-12.6f %-10.3f\n', ...
        'F1', '圆形', sprintf('D=%d', D_f1), pv_f1, rms_f1, eta1);
fprintf('%-6s %-12s %-18s %-12.6f %-12.6f %-10.3f\n', ...
        'F2', '矩形', sprintf('%dx%d', Lx_f2, Ly_f2), pv_f2, rms_f2, eta2);
fprintf('%-6s %-12s %-18s %-12.6f %-12.6f %-10.3f\n', ...
        'F3', '圆形', sprintf('D=%d', D_sub_f3), pv_f3, rms_f3, eta3);
fprintf('==============================================================\n\n');

meta = struct();
meta.ds              = ds;
meta.overlap_target  = overlap_target;
meta.rot_deg         = rot_deg;
meta.note            = 'Ground truth uses B-local to A-local mapping: p_A = R_true * p_B + t_true';

save('subaperture_data.mat', ...
     'subap_f1_A', 'subap_f1_B', 'gt_f1', ...
     'subap_f2_A', 'subap_f2_B', 'gt_f2', ...
     'subap_f3_A', 'subap_f3_B', 'gt_f3', ...
     'meta', ...
     'Xm1', 'Ym1', 'Zm1', ...
     'Xm2', 'Ym2', 'Zm2', ...
     'Xm3', 'Ym3', 'Zm3');

fprintf('数据已保存至 subaperture_data.mat\n');

%% ======================== 统一初始面型三维展示（改进版） ==================


if exist('Rsub_ap_f3','var')
    Rsub3_plot = Rsub_ap_f3;
elseif exist('Rsub_ap','var')
    Rsub3_plot = Rsub_ap;
else
    error('未找到 F3 子孔径半径变量。');
end

if exist('theta_true_f3','var')
    theta_plot_f3 = theta_true_f3;
else
    theta_plot_f3 = theta_true;
end

fig_all = figure('Name', '代表性自由曲面初始三维示意', ...
                 'Position', [80 80 1800 560], 'Color', 'w');

tiledlayout(1,3, 'Padding', 'compact', 'TileSpacing', 'compact');

ax1 = nexttile;
plot_scene_circle(ax1, Xm1, Ym1, Zm1, cA1, cB1, Rsub1, 0, theta_true, 'F1: peaks 自由曲面');

ax2 = nexttile;
plot_scene_rect(ax2, Xm2, Ym2, Zm2, cA2, cB2, Lsub_x, Lsub_y, 0, theta_true, 'F2: x-y 多项式矩形自由曲面');

ax3 = nexttile;
plot_scene_circle(ax3, Xm3, Ym3, Zm3, cA3, cB3, Rsub3_plot, 0, theta_plot_f3, 'F3: 自由曲面残差面型');


%% ======================== 局部函数 ======================================
function plot_scene_circle(ax, X, Y, Z, cA, cB, Rsub, thetaA, thetaB, ttl)
    axes(ax); 
    hold(ax, 'on');

    valid = isfinite(Z);
    zmin = min(Z(valid));
    zmax = max(Z(valid));
    zrange = zmax - zmin;

    % 底部圆圈放得更低一些，拉开层次
    zfloor = zmin - 0.65 * max(zrange, 1e-6);

    % 母面型
    hs = surf(ax, X, Y, Z, 'EdgeColor', 'none', 'FaceAlpha', 1.0);
    shading(ax, 'interp');

    % 插值器
    F = scatteredInterpolant(X(valid), Y(valid), Z(valid), 'natural', 'none');

    % 顶部真实子孔径边界
    nB = 240;
    [xAt, yAt] = circle_boundary_world(cA, Rsub, thetaA, nB);
    [xBt, yBt] = circle_boundary_world(cB, Rsub, thetaB, nB);
    zAt = F(xAt, yAt);
    zBt = F(xBt, yBt);

    % 底部圆圈
    patch(ax, xAt, yAt, zfloor * ones(size(xAt)), [1 0 0], 'FaceAlpha', 0.08, 'EdgeColor', 'r', 'LineWidth', 1.8);
    patch(ax, xBt, yBt, zfloor * ones(size(xBt)), [0 0 1], 'FaceAlpha', 0.08, 'EdgeColor', 'b', 'LineWidth', 1.8);

    % 半透明连接侧帘
    draw_curtain(ax, xAt, yAt, zfloor * ones(size(xAt)), zAt, [1 0 0], 0.15);
    draw_curtain(ax, xBt, yBt, zfloor * ones(size(xBt)), zBt, [0 0 1], 0.15);

    % 面型上的真实子孔径边界
    plot3(ax, xAt, yAt, zAt, 'r-', 'LineWidth', 2.2);
    plot3(ax, xBt, yBt, zBt, 'b-', 'LineWidth', 2.2);

    % 中心投影线
    zAc = F(cA(1), cA(2));
    zBc = F(cB(1), cB(2));
    plot3(ax, [cA(1), cA(1)], [cA(2), cA(2)], [zfloor, zAc], 'r--', 'LineWidth', 1.0);
    plot3(ax, [cB(1), cB(1)], [cB(2), cB(2)], [zfloor, zBc], 'b--', 'LineWidth', 1.0);

    % 样式
    colormap(ax, "parula");
    cb = colorbar(ax);
    cb.Box = 'on';

    xlim(ax, [min(X(valid)), max(X(valid))]);
    ylim(ax, [min(Y(valid)), max(Y(valid))]);
    zlim(ax, [zfloor, zmax + 0.10 * max(zrange,1e-6)]);

    grid(ax, 'on');
    box(ax, 'on');
    xlabel(ax, 'x (mm)');
    ylabel(ax, 'y (mm)');
    zlabel(ax, 'z (mm)');
    title(ax, ttl);
    legend(ax, {'母面型', '子孔径 A', '子孔径 B'}, 'Location', 'northeast');

    view(ax, 34, 24);

    % 关键：提高 z 轴显示比例
    pbaspect(ax, [1 1 0.45]);

    camlight(ax, 'headlight');
    lighting(ax, 'gouraud');
    set(ax, 'FontSize', 11, 'LineWidth', 0.8);
end

function plot_scene_rect(ax, X, Y, Z, cA, cB, Lx, Ly, thetaA, thetaB, ttl)
    axes(ax); 
    hold(ax, 'on');

    valid = isfinite(Z);
    zmin = min(Z(valid));
    zmax = max(Z(valid));
    zrange = zmax - zmin;

    zfloor = zmin - 0.65 * max(zrange, 1e-6);

    % 母面型
    hs = surf(ax, X, Y, Z, 'EdgeColor', 'none', 'FaceAlpha', 1.0);
    shading(ax, 'interp');

    % 插值器
    F = scatteredInterpolant(X(valid), Y(valid), Z(valid), 'natural', 'none');

    % 顶部真实子孔径边界
    [xAt, yAt] = rect_boundary_world_dense(cA, Lx, Ly, thetaA, 80);
    [xBt, yBt] = rect_boundary_world_dense(cB, Lx, Ly, thetaB, 80);
    zAt = F(xAt, yAt);
    zBt = F(xBt, yBt);

    % 底部矩形
    patch(ax, xAt, yAt, zfloor * ones(size(xAt)), [1 0 0], 'FaceAlpha', 0.08, 'EdgeColor', 'r', 'LineWidth', 1.8);
    patch(ax, xBt, yBt, zfloor * ones(size(xBt)), [0 0 1], 'FaceAlpha', 0.08, 'EdgeColor', 'b', 'LineWidth', 1.8);

    % 半透明连接侧帘
    draw_curtain(ax, xAt, yAt, zfloor * ones(size(xAt)), zAt, [1 0 0], 0.15);
    draw_curtain(ax, xBt, yBt, zfloor * ones(size(xBt)), zBt, [0 0 1], 0.15);

    % 面型上的真实子孔径边界
    plot3(ax, xAt, yAt, zAt, 'r-', 'LineWidth', 2.2);
    plot3(ax, xBt, yBt, zBt, 'b-', 'LineWidth', 2.2);

    % 中心投影线
    zAc = F(cA(1), cA(2));
    zBc = F(cB(1), cB(2));
    plot3(ax, [cA(1), cA(1)], [cA(2), cA(2)], [zfloor, zAc], 'r--', 'LineWidth', 1.0);
    plot3(ax, [cB(1), cB(1)], [cB(2), cB(2)], [zfloor, zBc], 'b--', 'LineWidth', 1.0);

    % 样式
    colormap(ax,"parula");
    cb = colorbar(ax);
    cb.Box = 'on';

    xlim(ax, [min(X(valid)), max(X(valid))]);
    ylim(ax, [min(Y(valid)), max(Y(valid))]);
    zlim(ax, [zfloor, zmax + 0.10 * max(zrange,1e-6)]);

    grid(ax, 'on');
    box(ax, 'on');
    xlabel(ax, 'x (mm)');
    ylabel(ax, 'y (mm)');
    zlabel(ax, 'z (mm)');
    title(ax, ttl);
    legend(ax, {'母面型', '子孔径 A', '子孔径 B'}, 'Location', 'northeast');

    view(ax, 34, 24);

    % 关键：提高 z 轴显示比例
    pbaspect(ax, [1 1 0.45]);

    camlight(ax, 'headlight');
    lighting(ax, 'gouraud');
    set(ax, 'FontSize', 11, 'LineWidth', 0.8);
end

function draw_curtain(ax, xb, yb, zb0, zb1, faceColor, alphaVal)
    Xs = [xb(:).'; xb(:).'];
    Ys = [yb(:).'; yb(:).'];
    Zs = [zb0(:).'; zb1(:).'];
    surf(ax, Xs, Ys, Zs, ...
        'FaceColor', faceColor, ...
        'FaceAlpha', alphaVal, ...
        'EdgeColor', 'none');
end

function [xb, yb] = circle_boundary_world(center_world, Rsub, theta_world, nPts)
    if nargin < 4
        nPts = 240;
    end
    t = linspace(0, 2*pi, nPts);
    u = Rsub * cos(t);
    v = Rsub * sin(t);
    [xb, yb] = transform_local_to_world(u(:), v(:), center_world, theta_world);
    xb = xb(:).';
    yb = yb(:).';
end

function [xb, yb] = rect_boundary_world_dense(center_world, Lx, Ly, theta_world, nPerEdge)
    if nargin < 5
        nPerEdge = 60;
    end

    t = linspace(-0.5, 0.5, nPerEdge);

    u1 =  Lx * t;                 v1 = -Ly/2 * ones(size(t));
    u2 =  Lx/2 * ones(size(t));   v2 =  Ly * t;
    u3 = -Lx * t;                 v3 =  Ly/2 * ones(size(t));
    u4 = -Lx/2 * ones(size(t));   v4 = -Ly * t;

    u = [u1, u2, u3, u4, u1(1)];
    v = [v1, v2, v3, v4, v1(1)];

    [xb, yb] = transform_local_to_world(u(:), v(:), center_world, theta_world);
    xb = xb(:).';
    yb = yb(:).';
end

function [xw, yw] = transform_local_to_world(u, v, center_world, theta_world)
    R = [cos(theta_world), -sin(theta_world); ...
         sin(theta_world),  cos(theta_world)];
    XY = R * [u(:)'; v(:)'];
    xw = XY(1,:)' + center_world(1);
    yw = XY(2,:)' + center_world(2);
end


function subap = sample_circular_subap_local(Finterp, center_world, theta_world, Rsub, ds, tag)
    [u, v] = make_local_circle_grid(Rsub, ds);
    [xw, yw] = transform_local_to_world(u, v, center_world, theta_world);

    z = Finterp(xw, yw);
    valid = isfinite(z);

    subap = struct();
    subap.tag          = tag;
    subap.shape        = 'circle';
    subap.radius       = Rsub;
    subap.x            = u(valid);
    subap.y            = v(valid);
    subap.z            = z(valid);
    subap.xw           = xw(valid);
    subap.yw           = yw(valid);
    subap.center_world = center_world(:);
    subap.theta_world  = theta_world;
    subap.theta_deg    = rad2deg(theta_world);
end

function subap = sample_rect_subap_local(Finterp, center_world, theta_world, Lx, Ly, ds, tag)
    [u, v] = make_local_rect_grid(Lx, Ly, ds);
    [xw, yw] = transform_local_to_world(u, v, center_world, theta_world);

    % 统一接口
    z = Finterp(xw, yw);
    valid = isfinite(z);

    subap = struct();
    subap.tag          = tag;
    subap.shape        = 'rect';
    subap.Lx           = Lx;
    subap.Ly           = Ly;
    subap.x            = u(valid);
    subap.y            = v(valid);
    subap.z            = z(valid);
    subap.xw           = xw(valid);
    subap.yw           = yw(valid);
    subap.center_world = center_world(:);
    subap.theta_world  = theta_world;
    subap.theta_deg    = rad2deg(theta_world);
end

function gt = make_ground_truth(cA, cB, theta_true, tag)
    R = [cos(theta_true), -sin(theta_true); ...
         sin(theta_true),  cos(theta_true)];

    gt = struct();
    gt.tag        = tag;
    gt.theta_rad  = theta_true;
    gt.theta_deg  = rad2deg(theta_true);
    gt.R          = R;
    gt.t          = (cB(:) - cA(:)).';
    gt.tx         = cB(1) - cA(1);
    gt.ty         = cB(2) - cA(2);
    gt.mapping    = 'p_A = R_true * p_B + t_true';
end

function [u, v] = make_local_circle_grid(Rsub, ds)
    uu = -Rsub : ds : Rsub;
    vv = -Rsub : ds : Rsub;
    [U, V] = meshgrid(uu, vv);
    mask = (U.^2 + V.^2 <= Rsub^2);
    u = U(mask);
    v = V(mask);
end

function [u, v] = make_local_rect_grid(Lx, Ly, ds)
    uu = -Lx/2 : ds : Lx/2;
    vv = -Ly/2 : ds : Ly/2;
    [U, V] = meshgrid(uu, vv);
    u = U(:);
    v = V(:);
end


function [xb, yb] = rect_boundary_world(center_world, Lx, Ly, theta_world)
    u = [-Lx/2,  Lx/2,  Lx/2, -Lx/2, -Lx/2]';
    v = [-Ly/2, -Ly/2,  Ly/2,  Ly/2, -Ly/2]';
    [xb, yb] = transform_local_to_world(u, v, center_world, theta_world);
end

function eta = circle_overlap_ratio(d, R)
    if d <= 0
        eta = 1.0; return;
    end
    if d >= 2*R
        eta = 0.0; return;
    end
    Aint = 2*R^2*acos(d/(2*R)) - 0.5*d*sqrt(4*R^2 - d^2);
    eta = Aint / (pi*R^2);
end

function sep = solve_circle_sep_from_overlap(R, eta_target)
    lo = 0;
    hi = 2*R - 1e-12;
    for k = 1:80
        mid = 0.5 * (lo + hi);
        if circle_overlap_ratio(mid, R) > eta_target
            lo = mid;
        else
            hi = mid;
        end
    end
    sep = 0.5 * (lo + hi);
end

function dx = solve_rect_dx_for_overlap(eta_target, dy, theta, Lx, Ly, ds_eval)
    eta0 = overlap_rect_rot_numerical(0, dy, theta, Lx, Ly, ds_eval);
    if eta0 < eta_target
        error('给定 dy 和 theta 时，最大可达重叠比小于目标值。');
    end

    lo = 0;
    hi = Lx - 1e-6;
    for k = 1:70
        mid = 0.5 * (lo + hi);
        eta_mid = overlap_rect_rot_numerical(mid, dy, theta, Lx, Ly, ds_eval);
        if eta_mid > eta_target
            lo = mid;
        else
            hi = mid;
        end
    end
    dx = 0.5 * (lo + hi);
end

function eta = overlap_rect_rot_numerical(dx, dy, theta, Lx, Ly, ds_eval)
    cA = [-dx/2; -dy/2];
    cB = [ dx/2;  dy/2];

    halfA = [Lx/2; Ly/2];
    halfB = [abs(cos(theta))*Lx/2 + abs(sin(theta))*Ly/2; ...
             abs(sin(theta))*Lx/2 + abs(cos(theta))*Ly/2];

    xlim = max(abs([cA(1)-halfA(1), cA(1)+halfA(1), cB(1)-halfB(1), cB(1)+halfB(1)])) + ds_eval;
    ylim = max(abs([cA(2)-halfA(2), cA(2)+halfA(2), cB(2)-halfB(2), cB(2)+halfB(2)])) + ds_eval;

    xv = -xlim : ds_eval : xlim;
    yv = -ylim : ds_eval : ylim;
    [X, Y] = meshgrid(xv, yv);

    XA = X - cA(1);
    YA = Y - cA(2);
    maskA = (abs(XA) <= Lx/2) & (abs(YA) <= Ly/2);

    ct = cos(theta); st = sin(theta);
    XB = X - cB(1);
    YB = Y - cB(2);
    UB =  ct * XB + st * YB;
    VB = -st * XB + ct * YB;
    maskB = (abs(UB) <= Lx/2) & (abs(VB) <= Ly/2);

    eta = sum(maskA & maskB, 'all') / min(sum(maskA, 'all'), sum(maskB, 'all'));
end

function coef = fit_plane_ls(x, y, z)
    A = [x(:), y(:), ones(numel(x),1)];
    coef = A \ z(:);
end

function Z_terms = fringe_zernike_Z4_Z18(rho, th)
    Z_terms = zeros(size(rho,1), size(rho,2), 15);
    Z_terms(:,:,1)  = 2*rho.^2 - 1;
    Z_terms(:,:,2)  = rho.^2 .* cos(2*th);
    Z_terms(:,:,3)  = rho.^2 .* sin(2*th);
    Z_terms(:,:,4)  = (3*rho.^3 - 2*rho) .* cos(th);
    Z_terms(:,:,5)  = (3*rho.^3 - 2*rho) .* sin(th);
    Z_terms(:,:,6)  = 6*rho.^4 - 6*rho.^2 + 1;
    Z_terms(:,:,7)  = rho.^3 .* cos(3*th);
    Z_terms(:,:,8)  = rho.^3 .* sin(3*th);
    Z_terms(:,:,9)  = (4*rho.^4 - 3*rho.^2) .* cos(2*th);
    Z_terms(:,:,10) = (4*rho.^4 - 3*rho.^2) .* sin(2*th);
    Z_terms(:,:,11) = (10*rho.^5 - 12*rho.^3 + 3*rho) .* cos(th);
    Z_terms(:,:,12) = (10*rho.^5 - 12*rho.^3 + 3*rho) .* sin(th);
    Z_terms(:,:,13) = 20*rho.^6 - 30*rho.^4 + 12*rho.^2 - 1;
    Z_terms(:,:,14) = rho.^4 .* cos(4*th);
    Z_terms(:,:,15) = rho.^4 .* sin(4*th);
end