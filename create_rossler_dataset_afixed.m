function create_rossler_dataset_afixed(varargin)
    %% Configure integration and parameter grid.
    p = inputParser;
    addParameter(p, 'dt', 0.005, @(x) isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'end_time', 200, @(x) isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'a', 0.35, @(x) isscalar(x) && isfinite(x));
    addParameter(p, 'b_range', [0.1, 0.7], @(x) isnumeric(x) && numel(x) == 2 && all(isfinite(x)) && x(1) < x(2));
    addParameter(p, 'c_range', [1.5, 10], @(x) isnumeric(x) && numel(x) == 2 && all(isfinite(x)) && x(1) < x(2));
    addParameter(p, 'grid_resolution', 100, @(x) isscalar(x) && isfinite(x) && x >= 2 && x == fix(x));
    addParameter(p, 'folder_name', fullfile(fileparts(mfilename('fullpath')), 'rossler_grid_data_afixed'), @ischar);
    addParameter(p, 'batch_size', 500, @(x) isscalar(x) && isfinite(x) && x >= 1 && x == fix(x));
    addParameter(p, 'timeout', 3, @(x) isscalar(x) && isfinite(x) && x > 0);
    parse(p, varargin{:});
    dt = p.Results.dt;
    end_time = p.Results.end_time;
    a = p.Results.a;
    resolution = p.Results.grid_resolution;
    folder_name = p.Results.folder_name;
    batch_size = p.Results.batch_size;
    timeout = p.Results.timeout;
    t = 0:dt:end_time;
    initial_conditions = [0, 0, 0.1];
    b_vals = linspace(p.Results.b_range(1), p.Results.b_range(2), resolution);
    c_vals = linspace(p.Results.c_range(1), p.Results.c_range(2), resolution);
    total = resolution^2;
    indices = (1:total)';
    b_column = reshape(b_vals(floor((indices-1)/resolution)+1), [], 1);
    c_column = reshape(c_vals(mod(indices-1, resolution)+1), [], 1);
    grid_param_data = [indices, repmat(a, total, 1), b_column, c_column];
    is_training_grid = resolution == 100 && a == 0.35 && ...
        isequal(p.Results.b_range, [0.1, 0.7]) && isequal(p.Results.c_range, [1.5, 10]);
    if ~exist(folder_name, 'dir')
        mkdir(folder_name);
    end
    if isempty(gcp('nocreate'))
        parpool;
    end

    %% Generate and save complete training trajectories in parallel.
    success_flags = false(total, 1);
    for first = 1:batch_size:total
        last = min(first + batch_size - 1, total);
        batch_params = grid_param_data(first:last, :);
        batch_success = false(size(batch_params, 1), 1);
        parfor k = 1:size(batch_params, 1)
            idx = batch_params(k, 1);
            b = batch_params(k, 3);
            c = batch_params(k, 4);
            rossler_eq = @(~, s) [-s(2)-s(3); s(1)+a*s(2); b*s(1)+s(3)*(s(1)-c)];
            timer = tic;
            options = odeset('RelTol', 1e-4, 'AbsTol', 1e-5, 'MaxStep', 0.5, ...
                'OutputFcn', @(~, ~, flag) check_timeout(timer, flag, timeout));
            try
                [~, y_solution] = ode45(rossler_eq, t, initial_conditions, options);
                if size(y_solution, 1) ~= numel(t) || any(~isfinite(y_solution(:))) || max(abs(y_solution(:))) > 1e6
                    error('Invalid or incomplete trajectory.');
                end
                b_str = strrep(sprintf('%.4f', b), '.', '');
                c_str = strrep(sprintf('%.4f', c), '.', '');
                filename = sprintf('%04d_b%sc%s.mat', idx, b_str, c_str);
                save_trajectory(fullfile(folder_name, filename), y_solution, a, b, c, ...
                    dt, initial_conditions, idx, is_training_grid);
                batch_success(k) = true;
            catch ME
                fprintf('Training sample %04d failed: %s\n', idx, ME.message);
            end
        end
        success_flags(first:last) = batch_success;
        param_data = grid_param_data(success_flags, :);
        save(fullfile(folder_name, 'parameters.mat'), 'param_data', 'grid_param_data', 'success_flags');
        fprintf('Training: processed %d/%d, saved %d.\n', last, total, sum(success_flags));
    end
    save(fullfile(folder_name, 'time.mat'), 't', 'dt', 'end_time', 'initial_conditions');
    fprintf('Training generation finished: %d/%d complete trajectories.\n', sum(success_flags), total);
end

%% Worker helpers for saving and integration timeout.
function save_trajectory(path, y_solution, a, b, c, dt, initial_conditions, data_index, is_training_grid)
    save(path, 'y_solution', 'a', 'b', 'c', 'dt', 'initial_conditions', 'data_index', 'is_training_grid');
end

function status = check_timeout(timer, flag, limit)
    status = 0;
    if isempty(flag) && toc(timer) > limit
        error('TIMEOUT: integration exceeded %.1f seconds.', limit);
    end
end
