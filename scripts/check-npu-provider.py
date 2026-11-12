"""Run AMD's supplied quicktest, rejecting silent CPU-only fallback."""
import os
import runpy
import argparse
import json
import numpy as np
import onnxruntime as ort
parser = argparse.ArgumentParser()
parser.add_argument('--compare-cpu', action='store_true', help='Compare the vendor model against ONNX Runtime CPU')
args = parser.parse_args()
original = ort.InferenceSession
def checked_session(*args, **kwargs):
    session = original(*args, **kwargs)
    if 'VitisAIExecutionProvider' not in session.get_providers():
        raise RuntimeError('Vitis AI initialization failed: refusing CPU-only fallback')
    print('Active providers:', session.get_providers(), flush=True)
    return session
ort.InferenceSession = checked_session
np.random.seed(42)
result = runpy.run_path(os.environ['RYZEN_AI_INSTALLATION_PATH'] + '/quicktest/quicktest.py', run_name='__main__')
# Provider presence alone does not prove NPU placement: retain and inspect the vendor's partition log too.
if args.compare_cpu:
    cpu = original(result['model'], providers=['CPUExecutionProvider'])
    reference = cpu.run(None, {'input': result['input_data']})
    for index, (expected, actual) in enumerate(zip(reference, result['outputs'])):
        print(json.dumps(dict(output=index, shape=list(expected.shape),
                              relative_l2=float(np.linalg.norm(actual-expected) / max(np.linalg.norm(expected), 1e-12)),
                              max_absolute_error=float(np.max(np.abs(actual-expected))),
                              identical=bool(np.array_equal(actual, expected)))), flush=True)
