"""Six-class Rössler classification using the preserved DCP LoopCorrection algorithm.
Source: https://bitbucket.org/pusuluri_krishna/deterministicchaosprospector/
"""
import ctypes as ct
import ctypes.util
import argparse
import csv
import json
from scipy.io import loadmat, savemat
import importlib.util
import os
from pathlib import Path
import re
import numpy as np


# Preserved DCP device functions; unused unsafe scratch storage is removed below.
CUDA_SOURCE_PATH = Path(__file__).with_name('rossler_classifier.cu')


def compute_cuda(params, dt=0.01, max_steps=6000000, start=999, end=1999, grid_mode=False):
    # Locate NVIDIA libraries from CUDA or an optional PyTorch installation.
    torch_spec = importlib.util.find_spec('torch')
    candidates = []
    if os.environ.get('NVRTC_PATH'):
        candidates.append(Path(os.environ['NVRTC_PATH']))
    if torch_spec:
        candidates.extend((Path(torch_spec.origin).parent / 'lib').glob('nvrtc64*.dll'))
        candidates.extend((Path(torch_spec.origin).parent / 'lib').glob('libnvrtc.so*'))
    if os.environ.get('CUDA_PATH'):
        candidates.extend((Path(os.environ['CUDA_PATH']) / 'bin').glob('nvrtc64*.dll'))
    library = str(candidates[0]) if candidates else ct.util.find_library('nvrtc')
    if not library:
        raise RuntimeError('NVRTC was not found. Install CUDA or set NVRTC_PATH to its library file.')
    dll_dir = Path(library).parent
    directory_handle = os.add_dll_directory(str(dll_dir)) if os.name == 'nt' and dll_dir.is_dir() else None
    builtin_paths = list(dll_dir.glob('nvrtc-builtins64*.dll'))
    builtins = ct.CDLL(str(builtin_paths[0])) if builtin_paths else None
    nvrtc = ct.CDLL(library)
    driver = ct.WinDLL('nvcuda.dll') if os.name == 'nt' else ct.CDLL('libcuda.so.1')
    fmad = True
    def check(result, operation):
        if result != 0:
            raise RuntimeError(f'{operation}: CUDA/NVRTC error {result}')
    source = CUDA_SOURCE_PATH.read_text(encoding='utf-8')
    source = re.sub(r'^#include.*$', '', source, flags=re.M)
    source = source.replace('double firstDerivativeCurrent[N_EQ1],firstDerivativePrevious;', 'double firstDerivativeCurrent[N_EQ1],firstDerivativePrevious=0;')
    source = source.replace('char s[200];unsigned si=0;', '')
    source = re.sub(r'^\s*s\[si\+\+\].*$', '', source, flags=re.M)
    source += '''
extern "C" __global__ void classify_reference(const double* parameters, double* output, int size) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=size) return;
    double state[3]={0.00000001,0,0};
    double params[3]={parameters[3*i],parameters[3*i+1],parameters[3*i+2]};
    GRID_RECONSTRUCTION
    output[i]=integrator_rk4(state,params,0.01,6000000,1,START_SYMBOL,END_SYMBOL);
}
'''
    grid_code = '''
    int bindex=(int)round((params[1]-0.1)/0.6*99);
    int cindex=(int)round((params[2]-1.5)/8.5*99);
    params[1]=0.1+bindex*((0.7-0.1)/99);
    params[2]=1.5+cindex*((10.0-1.5)/99);
    ''' if grid_mode else ''
    source = source.replace('GRID_RECONSTRUCTION', grid_code).replace('START_SYMBOL',str(start)).replace('END_SYMBOL',str(end))
    source = source.replace('params,0.01,6000000,1', f'params,{dt!r},{int(max_steps)},1')
    source = source.replace('#define MAX_KNEADING_LENGTH 2001', f'#define MAX_KNEADING_LENGTH {max(2001,end-start+2)}')
    program = ct.c_void_p()
    nvrtc.nvrtcCreateProgram.argtypes = [ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_char_p, ct.c_int, ct.c_void_p, ct.c_void_p]
    check(nvrtc.nvrtcCreateProgram(ct.byref(program), source.encode(), b'reference.cu', 0, None, None), 'create program')
    check(driver.cuInit(0), 'initialize driver')
    device = ct.c_int()
    check(driver.cuDeviceGet(ct.byref(device), 0), 'get device')
    major, minor = ct.c_int(), ct.c_int()
    check(driver.cuDeviceGetAttribute(ct.byref(major), 75, device), 'device capability')
    check(driver.cuDeviceGetAttribute(ct.byref(minor), 76, device), 'device capability')
    architecture = f'--gpu-architecture=compute_{major.value}{minor.value}'.encode()
    options = [architecture, b'--std=c++11', b'--fmad=true' if fmad else b'--fmad=false']
    option_array = (ct.c_char_p * len(options))(*options)
    nvrtc.nvrtcCompileProgram.argtypes = [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_char_p)]
    result = nvrtc.nvrtcCompileProgram(program, len(options), option_array)
    if result:
        length=ct.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(program,ct.byref(length))
        log=ct.create_string_buffer(length.value)
        nvrtc.nvrtcGetProgramLog(program,log)
        raise RuntimeError(log.value.decode())
    length = ct.c_size_t()
    nvrtc.nvrtcGetPTXSize.argtypes = [ct.c_void_p, ct.POINTER(ct.c_size_t)]
    nvrtc.nvrtcGetPTX.argtypes = [ct.c_void_p, ct.c_void_p]
    check(nvrtc.nvrtcGetPTXSize(program, ct.byref(length)), 'PTX size')
    ptx = ct.create_string_buffer(length.value)
    check(nvrtc.nvrtcGetPTX(program, ptx), 'get PTX')
    nvrtc.nvrtcDestroyProgram(ct.byref(program))
    check(driver.cuInit(0), 'initialize driver')
    device = ct.c_int()
    check(driver.cuDeviceGet(ct.byref(device), 0), 'get device')
    context = ct.c_void_p()
    driver.cuCtxCreate_v2.argtypes = [ct.POINTER(ct.c_void_p), ct.c_uint, ct.c_int]
    check(driver.cuCtxCreate_v2(ct.byref(context), 0, device), 'create context')
    module, function = ct.c_void_p(), ct.c_void_p()
    driver.cuModuleLoadData.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p]
    driver.cuModuleGetFunction.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p]
    driver.cuMemAlloc_v2.argtypes = [ct.POINTER(ct.c_uint64), ct.c_size_t]
    driver.cuMemcpyHtoD_v2.argtypes = [ct.c_uint64, ct.c_void_p, ct.c_size_t]
    driver.cuMemcpyDtoH_v2.argtypes = [ct.c_void_p, ct.c_uint64, ct.c_size_t]
    driver.cuMemFree_v2.argtypes = [ct.c_uint64]
    driver.cuCtxDestroy_v2.argtypes = [ct.c_void_p]
    driver.cuModuleUnload.argtypes = [ct.c_void_p]
    check(driver.cuModuleLoadData(ct.byref(module), ptx), 'load PTX')
    check(driver.cuModuleGetFunction(ct.byref(function), module, b'classify_reference'), 'get kernel')
    params = np.ascontiguousarray(params, dtype=np.float64)
    output = np.empty(len(params), dtype=np.float64)
    d_params, d_output = ct.c_uint64(), ct.c_uint64()
    check(driver.cuMemAlloc_v2(ct.byref(d_params), params.nbytes), 'allocate parameters')
    check(driver.cuMemAlloc_v2(ct.byref(d_output), output.nbytes), 'allocate output')
    check(driver.cuMemcpyHtoD_v2(d_params, params.ctypes.data, params.nbytes), 'copy parameters')
    size = ct.c_int(len(params))
    arguments = (ct.c_void_p * 3)(ct.cast(ct.byref(d_params), ct.c_void_p), ct.cast(ct.byref(d_output), ct.c_void_p), ct.cast(ct.byref(size), ct.c_void_p))
    driver.cuLaunchKernel.argtypes = [ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_uint, ct.c_uint, ct.c_uint, ct.c_uint, ct.c_uint, ct.c_void_p, ct.c_void_p, ct.c_void_p]
    try:
        check(driver.cuLaunchKernel(function, (len(params)+127)//128, 1, 1, 128, 1, 1, 0, None, arguments, None), 'launch kernel')
        check(driver.cuCtxSynchronize(), 'synchronize')
        check(driver.cuMemcpyDtoH_v2(output.ctypes.data, d_output, output.nbytes), 'copy output')
    finally:
        driver.cuMemFree_v2(d_params)
        driver.cuMemFree_v2(d_output)
        driver.cuModuleUnload(module)
        driver.cuCtxDestroy_v2(context)
        if directory_handle is not None:
            directory_handle.close()
    return output


def classify_k(values, tolerance=1e-5):
    # Map the recovered periodic signatures to A-E; other valid signatures are F.
    prototypes = np.array([0.0, -2 / 3, -2 / 7, -18 / 31, -8 / 15])
    values = np.asarray(values, dtype=float)
    distances = abs(values[:, None] - prototypes)
    labels = distances.argmin(axis=1) + 1
    labels[distances.min(axis=1) > tolerance] = 6
    labels[(values <= -1.1) | ~np.isfinite(values)] = 0
    return labels


def _validate_legacy_grid(params):
    b_index = np.rint((params[:, 1] - 0.1) / 0.6 * 99)
    c_index = np.rint((params[:, 2] - 1.5) / 8.5 * 99)
    valid = (np.all(abs(params[:, 0] - 0.35) < 1e-10)
             and np.all((0 <= b_index) & (b_index <= 99))
             and np.all((0 <= c_index) & (c_index <= 99))
             and np.all(abs(params[:, 1] - (0.1 + b_index * (0.6 / 99))) < 1e-10)
             and np.all(abs(params[:, 2] - (1.5 + c_index * (8.5 / 99))) < 1e-10))
    if not valid:
        raise ValueError('Legacy grid mode only supports the original 100x100 training grid.')


def _symbol_value(sequence):
    # Preserve the historical period and K normalization for trajectory mode.
    sequence = np.asarray(sequence, dtype=np.uint8)
    n = len(sequence)
    for period in range(1, n // 2):
        stop = ((n - 1) // period) * period
        if np.all(sequence[:stop].reshape(-1, period) == sequence[:period]):
            best = best_inverse = 0.0
            shift = inverse_shift = 0
            for i in range(period):
                value = inverse = 0.0
                for bit in sequence[i:i + period]:
                    value = 3 * value + int(bit)
                    inverse = 3 * inverse + 1 - int(bit)
                if best == 0 or value < best:
                    best, shift = value, i
                if best_inverse == 0 or inverse < best_inverse:
                    best_inverse, inverse_shift = inverse, i
            total = 0.0
            for i in range(n):
                bit = int(sequence[shift + period - 1 - i % period]) if best < best_inverse else 1 - int(sequence[inverse_shift + period - 1 - i % period])
                total += bit * 2.0 ** (i - n)
            return -total
    count, length, i, k, maximum = 1, 1, 0, 1, 1
    while True:
        if sequence[i + k - 1] != sequence[length + k - 1]:
            maximum = max(maximum, k)
            i += 1
            if i == length:
                count += 1
                length += maximum
                if length + 1 > n:
                    break
                i, k, maximum = 0, 1, 1
            else:
                k = 1
        else:
            k += 1
            if length + k > n:
                count += 1
                break
    return count / n


def classify_rossler(data, parameters=None, mode='system', legacy_grid=None, return_details=False,
                     integration_dt=0.01, max_steps=6000000, start=999, end=1999):
    """Return class 1-6 for a MAT path, sample dictionary, or N-by-3 states.

    Supply [a,b,c] for arrays without metadata. System mode recomputes a long
    orbit from the generating parameters. Trajectory mode uses only supplied
    states and requires at least 2000 detected symbols. return_details adds K.
    """
    if not np.isfinite(integration_dt) or integration_dt <= 0 or max_steps < 2 or not 0 <= start < end:
        raise ValueError('Invalid integration settings or symbol window.')
    # Read one sample and its generating parameters.
    sample_path = None
    if isinstance(data, (str, Path)):
        sample_path = Path(data)
        sample = loadmat(sample_path)
    elif isinstance(data, dict):
        sample = data
    else:
        sample = {'y_solution': data}
    states = np.asarray(sample.get('y_solution', sample.get('states', [])), dtype=float)
    if states.ndim != 2 or states.shape[1] != 3 or len(states) < 2 or not np.all(np.isfinite(states)):
        raise ValueError('Expected a finite N-by-3 raw state trajectory in y_solution or states.')
    recovered_from_table = False
    if parameters is None:
        if all(key in sample for key in ('a', 'b', 'c')):
            parameters = [float(np.asarray(sample[key]).squeeze()) for key in ('a', 'b', 'c')]
        elif 'parameters' in sample:
            parameters = sample['parameters']
        elif sample_path is not None and (sample_path.parent / 'parameters.mat').is_file():
            match = re.match(r'(\d+)_', sample_path.name)
            if match:
                table = loadmat(sample_path.parent / 'parameters.mat')['param_data']
                row = table[table[:, 0] == int(match.group(1))]
                if len(row) == 1:
                    parameters = row[0, 1:4]
                    recovered_from_table = True
    if parameters is None:
        raise ValueError('Missing system parameters: store a,b,c in the MAT file or pass parameters=[a,b,c].')
    if isinstance(parameters, dict):
        parameters = [parameters[key] for key in ('a', 'b', 'c')]
    params = np.asarray(parameters, dtype=float).reshape(-1)
    if params.shape != (3,) or not np.all(np.isfinite(params)) or params[0] <= 0:
        raise ValueError('Expected finite [a,b,c] with a > 0.')
    if legacy_grid is None:
        legacy_grid = bool(np.asarray(sample.get('is_training_grid', False)).squeeze())
        if recovered_from_table:
            try:
                _validate_legacy_grid(params[None, :])
                legacy_grid = True
            except ValueError:
                legacy_grid = False

    # Classify the generating system or the supplied long trajectory.
    symbol_count = None
    if mode == 'system':
        if legacy_grid:
            _validate_legacy_grid(params[None, :])
        value = float(compute_cuda(params[None, :], integration_dt, max_steps, start, end, bool(legacy_grid))[0])
    elif mode == 'trajectory':
        a, b, c = params
        dx = -states[:, 1] - states[:, 2]
        dy = states[:, 0] + a * states[:, 1]
        crossings = np.flatnonzero(dx[:-1] * dx[1:] < 0) + 1
        directions = dy[crossings] > 0
        previous = np.r_[False, directions[:-1]]
        peaks = crossings[(dx[crossings - 1] > 0) & (directions != previous)]
        symbols = (states[peaks, 2] > 0.12 * (c - a * b) / a).astype(np.uint8)
        symbol_count = len(symbols)
        if symbol_count < end + 1:
            raise ValueError(f'Only {symbol_count} symbols detected; trajectory mode needs {end + 1}. Use a longer trajectory or mode="system".')
        value = float(_symbol_value(symbols[start:end + 1]))
    else:
        raise ValueError('mode must be "system" or "trajectory".')
    label = int(classify_k([value])[0])
    if label == 0:
        raise RuntimeError('The system diverged or did not yield enough symbols within the integration limit.')
    if return_details:
        return {'class_id': label, 'label': 'ABCDEF'[label - 1], 'K': value,
                'parameters': params.tolist(), 'mode': mode, 'legacy_grid': bool(legacy_grid),
                'symbol_count': symbol_count}
    return label


def main():
    # Configure batch or single-system classification.
    base_dir = Path(__file__).resolve().parent
    default_parameters = base_dir / 'rossler_grid_data_afixed' / 'parameters.mat'
    if not default_parameters.is_file():
        default_parameters = base_dir / 'parameters.mat'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameters', type=Path, default=default_parameters)
    parser.add_argument('--single', nargs=3, type=float, metavar=('A', 'B', 'C'))
    parser.add_argument('--data', type=Path, help='Classify one MAT sample containing raw states and parameters.')
    parser.add_argument('--mode', choices=['system', 'trajectory'], default='system')
    parser.add_argument('--output', type=Path, default=base_dir / 'rossler_classification')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--legacy-grid', action='store_true', help='Reproduce CUDA construction of the original 100x100 training grid.')
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--max-steps', type=int, default=6000000)
    parser.add_argument('--start', type=int, default=999)
    parser.add_argument('--end', type=int, default=1999)
    args = parser.parse_args()
    if args.dt <= 0 or not np.isfinite(args.dt) or args.max_steps < 2 or not 0 <= args.start < args.end:
        parser.error('Invalid integration settings or symbol window.')
    if args.data:
        result = classify_rossler(args.data, mode=args.mode, legacy_grid=True if args.legacy_grid else None,
                                 return_details=True, integration_dt=args.dt, max_steps=args.max_steps, start=args.start, end=args.end)
        print(json.dumps(result, indent=2))
        return
    if args.single:
        param_data = np.array([[1, *args.single]], dtype=float)
    else:
        contents = loadmat(args.parameters)
        key = 'test_param_data' if 'test_param_data' in contents else 'param_data'
        param_data = np.asarray(contents[key][:, :4], dtype=float)
    params = param_data[:, 1:4]
    if len(params) == 0 or not np.all(np.isfinite(params)) or np.any(params[:, 0] <= 0):
        parser.error('Expected finite [index,a,b,c] rows with a > 0.')
    if args.legacy_grid:
        b_index = np.rint((params[:, 1] - 0.1) / 0.6 * 99)
        c_index = np.rint((params[:, 2] - 1.5) / 8.5 * 99)
        on_grid = (np.all(abs(params[:, 0] - 0.35) < 1e-10)
                   and np.all((0 <= b_index) & (b_index <= 99))
                   and np.all((0 <= c_index) & (c_index <= 99))
                   and np.all(abs(params[:, 1] - (0.1 + b_index * (0.6 / 99))) < 1e-10)
                   and np.all(abs(params[:, 2] - (1.5 + c_index * (8.5 / 99))) < 1e-10))
        if not on_grid:
            parser.error('--legacy-grid is restricted to the original training grid; omit it for shifted test parameters.')

    # Calculate K independently, then write the six-class results.
    values = compute_cuda(params, args.dt, args.max_steps, args.start, args.end, args.legacy_grid)
    labels = classify_k(values)
    output_data = np.column_stack((param_data, labels))
    args.output.mkdir(parents=True, exist_ok=True)
    savemat(args.output / 'parameters_with_labels.mat', {'param_data': output_data, 'k_values': values[:, None]})
    letters = np.array(['UNRESOLVED', 'A', 'B', 'C', 'D', 'E', 'F'])
    with (args.output / 'classification.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['index', 'a', 'b', 'c', 'K', 'class', 'label'])
        for row, value, label in zip(param_data, values, labels):
            writer.writerow([int(row[0]), *row[1:], value, int(label), letters[label]])
    report = {'systems': len(params), 'class_counts': {letters[c]: int(sum(labels == c)) for c in range(7)},
              'legacy_grid': args.legacy_grid, 'dt': args.dt, 'max_steps': args.max_steps,
              'symbol_start': args.start, 'symbol_end': args.end, 'fmad': True, 'K_tolerance': 1e-5,
              'algorithm': 'DCP LoopCorrection; A-E periodic signatures; F other valid signatures',
              'source': 'https://bitbucket.org/pusuluri_krishna/deterministicchaosprospector/'}

    # Compare with reference labels only after the independent calculation.
    if args.reference:
        reference = loadmat(args.reference)['param_data']
        mapping = {int(row[0]): row for row in reference}
        try:
            expected = np.array([mapping[int(row[0])] for row in param_data])
        except KeyError as error:
            raise ValueError(f'Reference is missing sample {error.args[0]}.') from error
        if not np.allclose(expected[:, 1:4], params, atol=1e-10, rtol=0):
            raise ValueError('Reference parameters differ from the classified parameters.')
        matches = labels == expected[:, 4]
        report.update(matched=int(sum(matches)), mismatched=int(sum(~matches)), agreement=float(np.mean(matches)),
                      confusion=[[int(sum((expected[:, 4] == a) & (labels == b))) for b in range(7)] for a in range(1, 7)])
        with (args.output / 'mismatches.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['index', 'expected_class', 'predicted_class', 'K'])
            for row, expect, label, value, match in zip(param_data, expected[:, 4], labels, values, matches):
                if not match:
                    writer.writerow([int(row[0]), int(expect), int(label), value])
    (args.output / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if np.any(labels == 0):
        raise SystemExit('Some systems diverged or did not provide enough symbols; see UNRESOLVED rows.')


if __name__ == '__main__':
    main()


