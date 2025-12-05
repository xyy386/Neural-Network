%% 实验1：Hebb学习规则实现
fprintf('实验1：Hebb学习规则实现\n');

%% 定义Hopfield网络类（面向对象实现）
fprintf('\n实现Hopfield网络类...\n');

% 创建简单的模式用于演示
fprintf('\n【示例】存储3个简单模式\n');

% 定义3个模式（5维向量）
P1 = [1; -1; 1; -1; 1];
P2 = [-1; -1; 1; 1; 1];
P3 = [1; 1; -1; -1; 1];

patterns = [P1, P2, P3];
n_neurons = size(patterns, 1);
n_patterns = size(patterns, 2);

fprintf('神经元数量: %d\n', n_neurons);
fprintf('存储模式数: %d\n', n_patterns);

fprintf('\n存储的模式:\n');
for i = 1:n_patterns
    fprintf('P%d: ', i);
    fprintf('%2d ', patterns(:, i));
    fprintf('\n');
end

%% 使用Hebb规则计算权重矩阵
fprintf('\n使用Hebb学习规则计算权重矩阵...\n');

W = zeros(n_neurons, n_neurons);

% Hebb规则：W = Σ p_k * p_k^T
for k = 1:n_patterns
    W = W + patterns(:, k) * patterns(:, k)';
end

% 零对角线（神经元不自连）
W = W - n_patterns * eye(n_neurons);

fprintf('权重矩阵 W:\n');
disp(W);

% 验证对称性
is_symmetric = isequal(W, W');
fprintf('权重矩阵是否对称: %s\n', char(string(is_symmetric)));
