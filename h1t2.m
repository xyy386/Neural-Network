%% 实验2：OR逻辑问题及与AND的对比
clear; clc; close all;

fprintf('实验2: OR逻辑问题\n');

% OR问题数据
P_or = [0 0 1 1;
        0 1 0 1];

T_or = [0 1 1 1];

% 创建并训练OR感知器
net_or = perceptron;
net_or.trainParam.epochs = 20;
net_or.trainParam.show = 5;

fprintf('\n训练OR感知器...\n');
[net_or, tr_or] = train(net_or, P_or, T_or);

% 测试
Y_or = net_or(P_or);

fprintf('\nOR逻辑测试结果:\n');
fprintf('x1  x2  | 目标  预测\n');
fprintf('----------------------\n');
for i = 1:4
    fprintf('%d   %d   |  %d     %d\n', ...
        P_or(1,i), P_or(2,i), T_or(i), round(Y_or(i)));
end

% AND问题（重新训练）
P_and = [0 0 1 1;
         0 1 0 1];
T_and = [0 0 0 1];

net_and = perceptron;
net_and.trainParam.epochs = 20;
net_and.trainParam.show = NaN;  % 不显示训练过程

[net_and, tr_and] = train(net_and, P_and, T_and);
Y_and = net_and(P_and);

% 对比可视化
figure('Name', '线性可分问题对比', 'NumberTitle', 'off');
set(gcf, 'Position', [100, 100, 1200, 400]);

% AND问题
subplot(1,3,1);
plotpv(P_and, T_and);
hold on;
plotpc(net_and.IW{1,1}, net_and.b{1});
title('AND逻辑');
xlabel('x_1');
ylabel('x_2');
legend('类别 0', '类别 1', '决策边界');
grid on;

% OR问题
subplot(1,3,2);
plotpv(P_or, T_or);
hold on;
plotpc(net_or.IW{1,1}, net_or.b{1});
title('OR逻辑');
xlabel('x_1');
ylabel('x_2');
legend('类别 0', '类别 1', '决策边界');
grid on;

% NOT问题
P_not = [0 1];
T_not = [1 0];

net_not = perceptron;
net_not.trainParam.epochs = 20;
net_not.trainParam.show = NaN;
[net_not, tr_not] = train(net_not, P_not, T_not);

subplot(1,3,3);
plot(P_not(T_not==1), zeros(1,sum(T_not==1)), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
plot(P_not(T_not==0), zeros(1,sum(T_not==0)), 'bx', 'MarkerSize', 10, 'LineWidth', 2);
% 绘制决策边界
x_line = -0.5:0.1:1.5;
y_line = zeros(size(x_line));
plot(x_line, y_line, 'g-', 'LineWidth', 2);
% 标注决策点
decision_point = -net_not.b{1} / net_not.IW{1,1};
plot(decision_point, 0, 'g*', 'MarkerSize', 15, 'LineWidth', 2);
title('NOT逻辑');
xlabel('x');
ylabel('');
xlim([-0.5, 1.5]);
ylim([-0.5, 0.5]);
legend('类别 1', '类别 0', '决策边界', '决策点');
grid on;