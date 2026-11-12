"""Run AMD's supplied quicktest, rejecting silent CPU-only fallback."""
import os
import runpy
import onnxruntime as ort
original = ort.InferenceSession
def checked_session(*args, **kwargs):
    session = original(*args, **kwargs)
    if 'VitisAIExecutionProvider' not in session.get_providers():
        raise RuntimeError('Vitis AI initialization failed: refusing CPU-only fallback')
    print('Active providers:', session.get_providers(), flush=True)
    return session
ort.InferenceSession = checked_session
runpy.run_path(os.environ['RYZEN_AI_INSTALLATION_PATH'] + '/quicktest/quicktest.py', run_name='__main__')
# Provider presence alone does not prove NPU placement: retain and inspect the vendor's partition log too.
