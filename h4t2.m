%% 实验2：联想记忆功能实现
fprintf('实验2：联想记忆功能实现\n');

fprintf('\n【实验3.1】从完整模式恢复（测试记忆功能）\n');
fprintf('------------------------------------------------------------------------\n');

% 测试每个存储的模式
for i = 1:n_patterns
    input_pattern = patterns(:, i);
    fprintf('\n测试模式 P%d: ', i);
    fprintf('%2d ', input_pattern);
    fprintf('\n');
    
    % 网络演化
    current_state = input_pattern;
    max_iterations = 20;
    converged = false;
    
    for iter = 1:max_iterations
        old_state = current_state;
        
        % 异步更新（逐个神经元更新）
        for j = 1:n_neurons
            net_input = W(j, :) * current_state;
            current_state(j) = sign(net_input);
            if current_state(j) == 0
                current_state(j) = 1;  % 处理零输入的情况
            end
        end
        
        % 检查是否收敛
        if isequal(current_state, old_state)
            converged = true;
            fprintf('  迭代 %d 次后收敛\n', iter);
            break;
        end
    end
    
    if ~converged
        fprintf('  未在 %d 次迭代内收敛\n', max_iterations);
    end
    
    fprintf('  最终状态: ');
    fprintf('%2d ', current_state);
    fprintf('\n');
    
    % 检查是否正确恢复
    is_correct = isequal(current_state, input_pattern);
    fprintf('  恢复正确: %s\n', char(string(is_correct)));
end

%% 噪声恢复实验
fprintf('\n【实验3.2】从含噪声模式恢复（测试容错能力）\n');
fprintf('------------------------------------------------------------------------\n');

% 选择第一个模式进行测试
target_pattern = patterns(:, 1);
fprintf('目标模式 P1: ');
fprintf('%2d ', target_pattern);
fprintf('\n\n');

% 测试不同噪声水平
noise_levels = [0.2, 0.4, 0.6];  % 翻转比例

for noise_level = noise_levels
    fprintf('噪声水平: %.0f%% (翻转 %d 个位)\n', noise_level*100, round(noise_level*n_neurons));
    
    % 添加噪声（随机翻转一些位）
    noisy_pattern = target_pattern;
    n_flips = round(noise_level * n_neurons);
    flip_indices = randperm(n_neurons, n_flips);
    noisy_pattern(flip_indices) = -noisy_pattern(flip_indices);
    
    fprintf('  输入模式: ');
    fprintf('%2d ', noisy_pattern);
    fprintf('\n');
    
    % 计算Hamming距离
    hamming_dist_input = sum(noisy_pattern ~= target_pattern);
    fprintf('  初始Hamming距离: %d\n', hamming_dist_input);
    
    % 网络恢复
    current_state = noisy_pattern;
    max_iterations = 50;
    
    for iter = 1:max_iterations
        old_state = current_state;
        
        for j = 1:n_neurons
            net_input = W(j, :) * current_state;
            current_state(j) = sign(net_input);
            if current_state(j) == 0
                current_state(j) = 1;
            end
        end
        
        if isequal(current_state, old_state)
            break;
        end
    end
    
    fprintf('  恢复模式: ');
    fprintf('%2d ', current_state);
    fprintf('\n');
    
    % 检查恢复结果
    hamming_dist_output = sum(current_state ~= target_pattern);
    is_recovered = isequal(current_state, target_pattern);
    
    fprintf('  最终Hamming距离: %d\n', hamming_dist_output);
    fprintf('  完全恢复: %s\n', char(string(is_recovered)));
    fprintf('  迭代次数: %d\n\n', iter);
end