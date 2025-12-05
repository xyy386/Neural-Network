%% 实验3：网络记忆容量分析
fprintf('实验3：网络记忆容量分析\n');

fprintf('\n理论记忆容量: P_max ≈ 0.15 * N\n');
fprintf('其中 N 是神经元数量\n\n');

% 测试不同数量的模式
n_neurons_capacity = 50;
max_test_patterns = round(0.3 * n_neurons_capacity);  % 测试到30%

fprintf('实验设置:\n');
fprintf('神经元数量: %d\n', n_neurons_capacity);
fprintf('测试模式数范围: 1 ~ %d\n', max_test_patterns);
fprintf('每个配置重复测试: 10 次\n\n');

fprintf('正在进行记忆容量实验...\n');

n_repeats = 10;
success_rates = zeros(max_test_patterns, 1);

for n_patterns_test = 1:max_test_patterns
    success_count = 0;
    
    for repeat = 1:n_repeats
        % 生成随机模式
        patterns_test = sign(randn(n_neurons_capacity, n_patterns_test));
        patterns_test(patterns_test == 0) = 1;
        
        % 计算权重矩阵
        W_test = zeros(n_neurons_capacity, n_neurons_capacity);
        for k = 1:n_patterns_test
            W_test = W_test + patterns_test(:, k) * patterns_test(:, k)';
        end
        W_test = W_test - n_patterns_test * eye(n_neurons_capacity);
        
        % 测试每个模式是否能正确召回
        all_recalled = true;
        for k = 1:n_patterns_test
            current_state = patterns_test(:, k);
            
            % 异步更新
            for iter = 1:100
                old_state = current_state;
                for j = 1:n_neurons_capacity
                    net_input = W_test(j, :) * current_state;
                    current_state(j) = sign(net_input);
                    if current_state(j) == 0
                        current_state(j) = 1;
                    end
                end
                if isequal(current_state, old_state)
                    break;
                end
            end
            
            % 检查是否恢复到原模式
            if ~isequal(current_state, patterns_test(:, k))
                all_recalled = false;
                break;
            end
        end
        
        if all_recalled
            success_count = success_count + 1;
        end
    end
    
    success_rates(n_patterns_test) = success_count / n_repeats;
    
    % 显示进度
    if mod(n_patterns_test, 2) == 0
        fprintf('  测试 %d 个模式: 成功率 %.1f%%\n', ...
                n_patterns_test, success_rates(n_patterns_test) * 100);
    end
end

fprintf('\n记忆容量分析完成！\n');

% 找到成功率降到50%以下的点
capacity_threshold = find(success_rates < 0.5, 1);
if isempty(capacity_threshold)
    capacity_threshold = max_test_patterns;
end

fprintf('\n结果分析:\n');
fprintf('理论容量: %.1f 个模式\n', 0.15 * n_neurons_capacity);
fprintf('实验容量(50%%成功率): %d 个模式\n', capacity_threshold);
fprintf('容量比例: %.2f%%\n', capacity_threshold / n_neurons_capacity * 100);
