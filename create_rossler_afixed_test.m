function create_rossler_afixed_test(varargin)
    %% Configure integration, offsets, and input paths.
    base_dir = fileparts(mfilename('fullpath'));
    default_param_file = fullfile(base_dir, 'rossler_grid_data_afixed', 'parameters.mat');
    if ~isfile(default_param_file)
        default_param_file = fullfile(base_dir, 'parameters.mat');
    end
    p = inputParser;
    addParameter(p, 'dt', 0.005, @(x) isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'end_time', 200, @(x) isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'b_offset', 0.05, @(x) isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'c_offset', 0.05, @(x) isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'param_file', default_param_file, @ischar);
    addParameter(p, 'test_samples_file', fullfile(base_dir, 'test_samples.mat'), @ischar);
    addParameter(p, 'folder_name', fullfile(base_dir, 'rossler_test_data_afixed'), @ischar);
    parse(p, varargin{:});
    b_offset = p.Results.b_offset;
    c_offset = p.Results.c_offset;
    output_folder = p.Results.folder_name;
    dt = p.Results.dt;
    end_time = p.Results.end_time;
    t = 0:dt:end_time;
    initial_conditions = [0, 0, 1];
    rng('shuffle');
    random_state = rng;

    %% Select distinct test parameters outside the training grid.
    parameter_data = load(p.Results.param_file);
    param_data = parameter_data.param_data;
    if isfield(parameter_data, 'grid_param_data')
        grid_param_data = parameter_data.grid_param_data;
    else
        grid_param_data = param_data;
    end
    testData = load(p.Results.test_samples_file, 'test_samples');
    test_indices = testData.test_samples(:);
    assert(numel(test_indices) == 200 && numel(unique(test_indices)) == 200, ...
        'Expected 200 distinct test sample indices.');
    [found, rows] = ismember(test_indices, grid_param_data(:, 1));
    assert(all(found), 'Selected indices are missing from the parameter grid.');
    original_param_data = grid_param_data(rows, 1:4);
    test_param_data = original_param_data;
    parameter_offsets = zeros(numel(test_indices), 2);
    b_grid = unique(grid_param_data(:, 3));
    c_grid = unique(grid_param_data(:, 4));
    b_bounds = [min(b_grid), max(b_grid)];
    c_bounds = [min(c_grid), max(c_grid)];
    tolerance = 1e-10;
    directions = [-1, -1; -1, 1; 1, -1; 1, 1];
    for k = 1:numel(test_indices)
        accepted = false;
        for choice = randperm(4)
            offsets = directions(choice, :) .* [b_offset, c_offset];
            candidate = original_param_data(k, 3:4) + offsets;
            if candidate(1) < b_bounds(1) || candidate(1) > b_bounds(2) || ...
                    candidate(2) < c_bounds(1) || candidate(2) > c_bounds(2)
                continue;
            end
            on_training_grid = any(abs(b_grid - candidate(1)) <= tolerance) && ...
                               any(abs(c_grid - candidate(2)) <= tolerance);
            duplicate = any(all(abs(test_param_data(1:k-1, 3:4) - candidate) <= tolerance, 2));
            if on_training_grid || duplicate
                continue;
            end
            test_param_data(k, 3:4) = candidate;
            parameter_offsets(k, :) = offsets;
            accepted = true;
            break;
        end
        assert(accepted, 'No valid offset for sample %04d; change b_offset or c_offset.', test_indices(k));
    end
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    success_flags = false(numel(test_indices), 1);
    metadata_file = fullfile(output_folder, 'test_parameters.mat');
    save(metadata_file, 'original_param_data', 'test_param_data', 'parameter_offsets', ...
        'test_indices', 'b_offset', 'c_offset', 'random_state', 'initial_conditions', ...
        't', 'dt', 'end_time', 'success_flags');
    if isempty(gcp('nocreate'))
        parpool;
    end

    %% Generate and save complete test trajectories in parallel.
    options = odeset('RelTol', 1e-4, 'AbsTol', 1e-5, 'MaxStep', 0.5);
    parfor k = 1:numel(test_indices)
        idx = test_indices(k);
        a = test_param_data(k, 2);
        b = test_param_data(k, 3);
        c = test_param_data(k, 4);
        rossler_eq = @(~, s) [-s(2)-s(3); s(1)+a*s(2); b*s(1)+s(3)*(s(1)-c)];
        try
            [~, y_solution] = ode45(rossler_eq, t, initial_conditions, options);
            if size(y_solution, 1) ~= numel(t) || any(~isfinite(y_solution(:))) || max(abs(y_solution(:))) > 1e6
                error('Invalid or incomplete trajectory.');
            end
            filename = sprintf('%04d_test_new.mat', idx);
            save_trajectory(fullfile(output_folder, filename), y_solution, a, b, c, initial_conditions, dt, idx);
            success_flags(k) = true;
        catch ME
            fprintf('Test sample %04d failed: %s\n', idx, ME.message);
        end
    end
    save(metadata_file, 'success_flags', '-append');
    fprintf('Test generation finished: %d/%d complete trajectories.\n', sum(success_flags), numel(test_indices));
end

%% Save each test trajectory with its actual parameters.
function save_trajectory(path, y_solution, a, b, c, initial_conditions, dt, data_index)
    is_training_grid = false;
    save(path, 'y_solution', 'a', 'b', 'c', 'initial_conditions', 'dt', 'data_index', 'is_training_grid');
end
