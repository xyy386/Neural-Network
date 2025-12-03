%% 实验4：使用多层前馈网络解决XOR问题
clear; clc; close all;

fprintf('实验4: 使用多层网络解决XOR问题\n');

% XOR问题数据
P_xor = [0 0 1 1;
         0 1 0 1];

T_xor = [0 1 1 0];

%% 方法1：使用feedforwardnet创建多层网络
fprintf('\n方法1: 使用feedforwardnet\n');
fprintf('---------------------------\n');

% 创建一个具有1个隐藏层（2个神经元）的前馈网络
hiddenLayerSize = 2;
net_mlp = feedforwardnet(hiddenLayerSize, 'trainlm');

% 配置训练参数
net_mlp.trainParam.epochs = 1000;        % 最大训练轮数
net_mlp.trainParam.goal = 1e-5;          % 性能目标
net_mlp.trainParam.show = 100;           % 显示频率
net_mlp.trainParam.lr = 0.1;             % 学习率

% 设置数据划分（训练/验证/测试）
net_mlp.divideParam.trainRatio = 1;      % 100%用于训练
net_mlp.divideParam.valRatio = 0;
net_mlp.divideParam.testRatio = 0;

% 训练网络
fprintf('\n开始训练多层网络...\n');
[net_mlp, tr_mlp] = train(net_mlp, P_xor, T_xor);

% 测试网络
Y_mlp = net_mlp(P_xor);

fprintf('\n多层网络测试结果:\n');
fprintf('x1  x2  | 目标  预测值    预测类别  是否正确\n');
fprintf('----------------------------------------------\n');
correct = 0;
for i = 1:4
    pred_class = round(Y_mlp(i));
    is_correct = (pred_class == T_xor(i));
    if is_correct
        correct = correct + 1;
    end
    fprintf('%d   %d   |  %d    %.4f      %d         %s\n', ...
        P_xor(1,i), P_xor(2,i), T_xor(i), Y_mlp(i), ...
        pred_class, char(string(is_correct)));
end

accuracy = correct / 4 * 100;
fprintf('\n准确率: %.2f%%\n', accuracy);

%% 可视化决策边界
figure('Name', '多层网络解决XOR问题', 'NumberTitle', 'off');
set(gcf, 'Position', [100, 100, 1200, 400]);

% 原始数据
subplot(1,3,1);
hold on;
plot(P_xor(1, T_xor==0), P_xor(2, T_xor==0), 'bo', ...
     'MarkerSize', 12, 'LineWidth', 2);
plot(P_xor(1, T_xor==1), P_xor(2, T_xor==1), 'rx', ...
     'MarkerSize', 12, 'LineWidth', 2);
title('XOR问题');
xlabel('x_1');
ylabel('x_2');
legend('类别 0', '类别 1');
grid on;
axis([-0.5 1.5 -0.5 1.5]);

% 多层网络的决策边界（详细）
subplot(1,3,2);
hold on;

% 创建网格
[X1, X2] = meshgrid(-0.5:0.02:1.5, -0.5:0.02:1.5);
X_grid = [X1(:)'; X2(:)'];

% 对网格点进行预测
Y_grid = net_mlp(X_grid);
Y_grid = reshape(Y_grid, size(X1));

% 绘制决策区域
contourf(X1, X2, Y_grid, 20, 'LineStyle', 'none');
colorbar;
colormap(jet);

% 绘制数据点
plot(P_xor(1, T_xor==0), P_xor(2, T_xor==0), 'bo', ...
     'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'b');
plot(P_xor(1, T_xor==1), P_xor(2, T_xor==1), 'rx', ...
     'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'r');

% 绘制决策边界（0.5等高线）
contour(X1, X2, Y_grid, [0.5 0.5], 'k-', 'LineWidth', 2);

title('多层网络的决策区域');
xlabel('x_1');
ylabel('x_2');
axis([-0.5 1.5 -0.5 1.5]);

% 训练性能
subplot(1,3,3);
plotperform(tr_mlp);
title('训练性能');

fprintf('\n结论: 多层网络成功解决了XOR问题！\n');

%% 显示网络结构
figure('Name', '网络结构', 'NumberTitle', 'off');
view(net_mlp);