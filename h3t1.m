%% 实验3：BP网络训练过程分析
fprintf('实验3：BP网络训练过程分析\n');

% XOR问题数据
P_xor = [0 0 1 1;
         0 1 0 1];
T_xor = [0 1 1 0];

fprintf('\nXOR逻辑真值表:\n');
fprintf('x1  x2  | Target\n');
fprintf('-----------------\n');
for i = 1:4
    fprintf('%d   %d   |   %d\n', P_xor(1,i), P_xor(2,i), T_xor(i));
end

% 创建BP网络
hiddenSize = 4;  % 隐藏层神经元数
net_xor = feedforwardnet(hiddenSize, 'trainlm');

% 配置激活函数
net_xor.layers{1}.transferFcn = 'tansig';  % 隐藏层：双曲正切
net_xor.layers{2}.transferFcn = 'logsig';  % 输出层：sigmoid

% 训练参数
net_xor.trainParam.epochs = 1000;
net_xor.trainParam.goal = 1e-5;
net_xor.trainParam.show = 50;
net_xor.trainParam.showWindow = false;

% 数据划分（全部用于训练）
net_xor.divideParam.trainRatio = 1;
net_xor.divideParam.valRatio = 0;
net_xor.divideParam.testRatio = 0;

% 训练网络
fprintf('\n开始训练BP网络...\n');
fprintf('网络结构: 2-%d-1\n', hiddenSize);
fprintf('隐藏层激活函数: tansig\n');
fprintf('输出层激活函数: logsig\n');
fprintf('训练算法: Levenberg-Marquardt\n\n');

[net_xor, tr_xor] = train(net_xor, P_xor, T_xor);

% 测试网络
Y_xor = net_xor(P_xor);

fprintf('\n训练完成！\n');
fprintf('训练轮数: %d\n', tr_xor.num_epochs);
fprintf('最终MSE: %.8f\n', tr_xor.best_perf);

fprintf('\n测试结果:\n');
fprintf('x1  x2  | 目标  网络输出  预测类别  是否正确\n');
fprintf('-----------------------------------------------\n');
correct = 0;
for i = 1:4
    pred_class = round(Y_xor(i));
    is_correct = (pred_class == T_xor(i));
    if is_correct
        correct = correct + 1;
    end
    fprintf('%d   %d   |  %d    %.6f      %d         %s\n', ...
        P_xor(1,i), P_xor(2,i), T_xor(i), Y_xor(i), ...
        pred_class, char(string(is_correct)));
end

fprintf('\n准确率: %.2f%%\n', correct/4*100);

% 可视化训练过程
figure('Name', 'BP网络训练过程分析', 'NumberTitle', 'off');
set(gcf, 'Position', [100, 100, 1400, 900]);

% 训练性能曲线
subplot(2,3,1);
if ~isempty(tr_xor.perf)
    semilogy(0:length(tr_xor.perf)-1, tr_xor.perf, 'b-', 'LineWidth', 2);
    xlabel('训练轮数 (Epochs)', 'FontSize', 11);
    ylabel('均方误差 (MSE)', 'FontSize', 11);
    title('训练性能曲线', 'FontSize', 13, 'FontWeight', 'bold');
    grid on;
end

% 梯度变化
subplot(2,3,2);
if isfield(tr_xor, 'gradient') && ~isempty(tr_xor.gradient)
    semilogy(0:length(tr_xor.gradient)-1, tr_xor.gradient, 'r-', 'LineWidth', 2);
    xlabel('训练轮数 (Epochs)', 'FontSize', 11);
    ylabel('梯度', 'FontSize', 11);
    title('梯度下降过程', 'FontSize', 13, 'FontWeight', 'bold');
    grid on;
end

% 决策边界
subplot(2,3,3);
[X1, X2] = meshgrid(-0.5:0.02:1.5, -0.5:0.02:1.5);
X_grid = [X1(:)'; X2(:)'];
Y_grid = net_xor(X_grid);
Y_grid = reshape(Y_grid, size(X1));

contourf(X1, X2, Y_grid, 20);
hold on;
colormap(jet);
colorbar;

plot(P_xor(1, T_xor==0), P_xor(2, T_xor==0), 'bo', ...
     'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'b');
plot(P_xor(1, T_xor==1), P_xor(2, T_xor==1), 'rx', ...
     'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'r');
contour(X1, X2, Y_grid, [0.5 0.5], 'k-', 'LineWidth', 2);

title('XOR决策边界', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('x_1', 'FontSize', 11);
ylabel('x_2', 'FontSize', 11);
legend('类别 0', '类别 1', '决策边界', 'Location', 'best');
axis([-0.5 1.5 -0.5 1.5]);

% 网络结构图
subplot(2,3,4);
view(net_xor);

% 权重分布
subplot(2,3,5);
weights = [net_xor.IW{1,1}(:); net_xor.LW{2,1}(:)];
histogram(weights, 20, 'FaceColor', [0.3 0.5 0.8]);
xlabel('权重值', 'FontSize', 11);
ylabel('频数', 'FontSize', 11);
title('权重分布直方图', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

% 误差分布
subplot(2,3,6);
errors = T_xor - Y_xor;
stem(1:4, errors, 'filled', 'LineWidth', 2);
hold on;
plot([0 5], [0 0], 'r--', 'LineWidth', 1);
xlabel('样本编号', 'FontSize', 11);
ylabel('误差 (目标-输出)', 'FontSize', 11);
title('各样本预测误差', 'FontSize', 13, 'FontWeight', 'bold');
xlim([0 5]);
grid on;

fprintf('\n训练过程分析可视化完成！\n');