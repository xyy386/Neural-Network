%% 实验2：BP网络函数逼近 - 非线性函数拟合
fprintf('实验2：BP网络函数逼近 - 非线性函数拟合\n');

fprintf('目标函数: y = sin(x) + 0.5*sin(3x) + 0.3*cos(5x)\n');

%% 生成训练数据
x_train = -2*pi:0.1:2*pi;  % 训练集
x_test = -2*pi:0.05:2*pi;   % 测试集

% 目标函数
target_function = @(x) sin(x) + 0.5*sin(3*x) + 0.3*cos(5*x);

y_train = target_function(x_train);
y_test = target_function(x_test);

% 添加噪声到训练数据
noise_level = 0.1;
y_train_noisy = y_train + noise_level * randn(size(y_train));

fprintf('数据集信息:\n');
fprintf('训练样本数: %d\n', length(x_train));
fprintf('测试样本数: %d\n', length(x_test));
fprintf('噪声水平: %.2f\n', noise_level);

%% 测试不同隐藏层神经元数量
hidden_sizes = [5, 10, 20, 30];
colors_approx = {'b', 'g', 'r', 'm'};

fprintf('\n【实验】测试不同隐藏层大小的影响\n');
fprintf('隐藏层神经元数 | 训练MSE    测试MSE    训练时间(s)\n');
fprintf('----------------------------------------------------------\n');

results_approx = struct();

for idx = 1:length(hidden_sizes)
    h_size = hidden_sizes(idx);
    
    % 创建网络
    net_approx = fitnet(h_size, 'trainlm');
    
    % 配置
    net_approx.layers{1}.transferFcn = 'tansig';
    net_approx.layers{2}.transferFcn = 'purelin';  % 输出层线性
    
    % 训练参数
    net_approx.trainParam.epochs = 1000;
    net_approx.trainParam.goal = 1e-6;
    net_approx.trainParam.show = NaN;
    net_approx.trainParam.showWindow = false;
    
    % 数据划分
    net_approx.divideParam.trainRatio = 1;
    net_approx.divideParam.valRatio = 0;
    net_approx.divideParam.testRatio = 0;
    
    % 训练
    tic;
    [net_approx, tr_approx] = train(net_approx, x_train, y_train_noisy);
    train_time = toc;
    
    % 预测
    y_train_pred = net_approx(x_train);
    y_test_pred = net_approx(x_test);
    
    % 计算误差
    mse_train = mean((y_train - y_train_pred).^2);
    mse_test = mean((y_test - y_test_pred).^2);
    
    % 保存结果
    results_approx(idx).hidden_size = h_size;
    results_approx(idx).net = net_approx;
    results_approx(idx).y_pred = y_test_pred;
    results_approx(idx).mse_train = mse_train;
    results_approx(idx).mse_test = mse_test;
    results_approx(idx).train_time = train_time;
    results_approx(idx).color = colors_approx{idx};
    
    fprintf('%15d     | %.6f   %.6f   %.4f\n', ...
        h_size, mse_train, mse_test, train_time);
end

%% 选择最佳模型进行详细分析
[~, best_idx] = min([results_approx.mse_test]);
best_model = results_approx(best_idx);

fprintf('\n最佳模型: %d个隐藏层神经元\n', best_model.hidden_size);
fprintf('测试集MSE: %.6f\n', best_model.mse_test);

%% 可视化函数逼近结果
figure('Name', 'BP网络非线性函数逼近', 'NumberTitle', 'off');
set(gcf, 'Position', [200, 50, 1400, 900]);

% 原始数据和拟合结果
subplot(2,3,1);
plot(x_test, y_test, 'k-', 'LineWidth', 2, 'DisplayName', '真实函数');
hold on;
scatter(x_train, y_train_noisy, 30, 'b', 'filled', 'MarkerFaceAlpha', 0.4, ...
        'DisplayName', '训练数据（含噪声）');
plot(x_test, best_model.y_pred, 'r--', 'LineWidth', 2, 'DisplayName', ...
     sprintf('BP逼近(%d神经元)', best_model.hidden_size));
xlabel('x', 'FontSize', 11);
ylabel('y', 'FontSize', 11);
title('最佳模型函数逼近效果', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;

% 不同隐藏层大小的对比
subplot(2,3,2);
plot(x_test, y_test, 'k-', 'LineWidth', 2.5, 'DisplayName', '真实函数');
hold on;
for idx = 1:length(results_approx)
    plot(x_test, results_approx(idx).y_pred, '--', ...
         'LineWidth', 1.5, 'Color', results_approx(idx).color, ...
         'DisplayName', sprintf('%d神经元', results_approx(idx).hidden_size));
end
xlabel('x', 'FontSize', 11);
ylabel('y', 'FontSize', 11);
title('不同网络规模对比', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;

% 逼近误差分析
subplot(2,3,3);
error_best = y_test - best_model.y_pred;
plot(x_test, error_best, 'r-', 'LineWidth', 1.5);
hold on;
plot(x_test, zeros(size(x_test)), 'k--', 'LineWidth', 1);
xlabel('x', 'FontSize', 11);
ylabel('误差', 'FontSize', 11);
title(sprintf('逼近误差（%d神经元）', best_model.hidden_size), ...
      'FontSize', 13, 'FontWeight', 'bold');
grid on;

% MSE对比
subplot(2,3,4);
mse_train_all = [results_approx.mse_train];
mse_test_all = [results_approx.mse_test];
hidden_labels = arrayfun(@(x) sprintf('%d', x), hidden_sizes, 'UniformOutput', false);

bar_data = [mse_train_all; mse_test_all]';
bar(bar_data);
set(gca, 'XTickLabel', hidden_labels);
xlabel('隐藏层神经元数', 'FontSize', 11);
ylabel('MSE', 'FontSize', 11);
title('不同网络规模的MSE对比', 'FontSize', 13, 'FontWeight', 'bold');
legend('训练MSE', '测试MSE', 'Location', 'best');
grid on;

% 计算R²
y_mean = mean(y_test);
ss_tot = sum((y_test - y_mean).^2);
ss_res = sum((y_test - best_model.y_pred).^2);
r_squared = 1 - ss_res / ss_tot;

fprintf('\n【性能指标】\n');
fprintf('均方误差 MSE: %.6f\n', best_model.mse_test);
fprintf('均方根误差 RMSE: %.6f\n', sqrt(best_model.mse_test));
fprintf('决定系数 R²: %.6f\n', r_squared);
fprintf('最大绝对误差: %.6f\n', max(abs(error_best)));
