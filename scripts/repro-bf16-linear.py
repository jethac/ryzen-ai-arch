"""Real-NPU diagnostic: exact integer-valued Gemm, no checkpoints required.

Run through run-sdk.sh. A nonzero exit indicates incorrect output or CPU fallback.
Artifacts include ONNX, inputs, CPU reference, NPU output and numerical diagnostics.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--m', type=int, default=32)
p.add_argument('--k', type=int, default=64)
p.add_argument('--n', type=int, default=64)
p.add_argument('--keep-artifacts', action='store_true')
p.add_argument('--tp-size', type=int, choices=[1, 2])
p.add_argument('--overlay-api', action='store_true', help='Diagnostic: use the newer L1/L2 overlay API')
a = p.parse_args()
root = Path(a.root)
root.mkdir(parents=True, exist_ok=True)
x = (np.arange(a.m*a.k).reshape(a.m, a.k) % 16 + 1).astype(np.float32)
w = np.zeros((a.n, a.k), dtype=np.float32)
w[np.arange(a.n), np.arange(a.n) % a.k] = 1
b = np.zeros(a.n, dtype=np.float32)
model = helper.make_model(helper.make_graph(
    [helper.make_node('Gemm', ['x', 'w', 'b'], ['y'], transB=1)], 'exact_gemm',
    [helper.make_tensor_value_info('x', TensorProto.FLOAT, [a.m, a.k])],
    [helper.make_tensor_value_info('y', TensorProto.FLOAT, [a.m, a.n])],
    [numpy_helper.from_array(w, 'w'), numpy_helper.from_array(b, 'b')]),
    opset_imports=[helper.make_opsetid('', 17)], ir_version=9)
onnx.checker.check_model(model)
onnx.save(model, root/'model.onnx')
config = {'passes': [{'name': 'init', 'plugin': 'vaip-pass_init'},
                     {'name': 'vaiml_partition', 'plugin': 'vaip-pass_vaiml_partition',
                      'vaiml_config': {'optimize_level': 1, 'preferred_data_storage': 'auto'}}],
          'target': 'VAIML', 'targets': [{'name': 'VAIML', 'pass': ['init', 'vaiml_partition']}]}
if a.keep_artifacts:
    config['passes'][1]['vaiml_config'].update(keep_outputs=True, enable_temp_file_io_in_mem=False)
if a.tp_size:
    config['passes'][1]['vaiml_config']['tp_size'] = a.tp_size
if a.overlay_api:
    config['passes'][1]['vaiml_config']['backend_args'] = 'use-L1L2-connections-overlay-api=true'
(root/'config.json').write_text(json.dumps(config))
cpu = ort.InferenceSession(str(root/'model.onnx'), providers=['CPUExecutionProvider'])
expected = x @ w.T + b
reference = cpu.run(None, {'x': x})[0]
assert np.array_equal(reference, expected), 'CPU export mismatch'
options = ort.SessionOptions()
options.log_severity_level = 1
npu = ort.InferenceSession(str(root/'model.onnx'), sess_options=options,
    providers=['VitisAIExecutionProvider'], provider_options=[{
        'config_file': str(root/'config.json'), 'cache_dir': str(root/'cache'),
        'enable_cache_file_io_in_mem': '0'}])
assert 'VitisAIExecutionProvider' in npu.get_providers(), 'CPU fallback'
actual = npu.run(None, {'x': x})[0]
contexts = list((root/'cache').glob('*/context.json'))
assert any(meta.get('device') == 'VAIML' for path in contexts
           for meta in json.loads(path.read_text()).get('metaDef', [])), 'No VAIML NPU partition recorded'
np.savez(root/'tensors.npz', x=x, w=w, b=b, expected=expected, actual=actual)
report = dict(shape=list(actual.shape), identical=bool(np.array_equal(actual, expected)),
              relative_l2=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)),
              zero_columns=np.flatnonzero(np.all(actual == 0, axis=0)).tolist(),
              first_expected=expected[0].tolist(), first_actual=actual[0].tolist(),
              loaded_xrt=sorted(set(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                    if 'libxrt' in line)))
(root/'result.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k: report[k] for k in ('shape', 'identical', 'relative_l2', 'loaded_xrt')} |
                 {'zero_column_count': len(report['zero_columns'])}), flush=True)
raise SystemExit(0 if report['identical'] else 2)
