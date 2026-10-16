"""Assemble the deployable site in sp/dist: app + Python code + models + the slice of Pyodide and
onnxruntime-web it needs. Everything self-hosted (same origin) so cross-origin isolation is simple
and the app works offline after first use."""
import json, os, shutil
from pathlib import Path

SP = Path(__file__).resolve().parent
DIST = SP / 'dist'
PYO = SP / 'dl' / 'pyodide'
ORT = SP / 'web' / 'node_modules' / 'onnxruntime-web' / 'dist'
NEED = ['numpy', 'scipy', 'scikit-image', 'opencv-python']


def deps(lock, names):
    seen, stack = set(), list(names)
    while stack:
        n = stack.pop().lower().replace('_', '-')
        if n in seen: continue
        seen.add(n); stack += lock['packages'][n]['depends']
    return seen


def main():
    shutil.rmtree(DIST, ignore_errors=True)
    shutil.copytree(SP / 'app', DIST)
    (DIST / '.nojekyll').write_text('')
    (DIST / 'models').mkdir(exist_ok=True)
    for m in ['birefnet_lite_1120_dc_w16.onnx', 'isnet_w16.onnx']:
        shutil.copy(SP / 'export' / m, DIST / 'models' / m)
    # pyodide core + only the wheels we load
    p = DIST / 'pyodide'; p.mkdir()
    for f in ['pyodide.js', 'pyodide.asm.js', 'pyodide.asm.wasm', 'python_stdlib.zip', 'pyodide-lock.json']:
        shutil.copy(PYO / f, p / f)
    lock = json.load(open(PYO / 'pyodide-lock.json'))
    pk = deps(lock, NEED)
    for n in sorted(pk):
        shutil.copy(PYO / lock['packages'][n]['file_name'], p / lock['packages'][n]['file_name'])
    # onnxruntime-web: the webgpu bundle + both wasm variants it may pick
    o = DIST / 'ort'; o.mkdir()
    for f in ['ort.webgpu.min.js', 'ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm',
              'ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm',
              'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']:
        if (ORT / f).exists(): shutil.copy(ORT / f, o / f)
    t = DIST / 'tools' / 'export'; t.mkdir(parents=True)
    shutil.copy(SP / 'build.py', DIST / 'tools' / 'build.py')
    for f in ['load_birefnet.py', 'lowmem.py', 'export_dc.py', 'fp16_store.py', 'split_export.py', 'deform_portable.py', 'chunked_conv.py']:
        shutil.copy(SP / 'export' / f, t / f)
    tot = sum(f.stat().st_size for f in DIST.rglob('*') if f.is_file())
    big = max((f for f in DIST.rglob('*') if f.is_file()), key=lambda f: f.stat().st_size)
    print(f'dist: {tot / 1e6:.0f} MB, {len(pk)} python packages, largest file {big.name} {big.stat().st_size / 1e6:.0f} MB')


if __name__ == '__main__':
    main()
