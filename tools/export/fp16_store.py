"""Store large float32 weights as float16 + a Cast back to float32 at load.
Halves the download; the maths still runs in float32 (identical ops, ~1e-3 weight rounding)."""
import sys
import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto


def convert(src, dst, min_elems=1024):
    m = onnx.load(src)
    g = m.graph
    new_inits, casts, n = [], [], 0
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT and np.prod(init.dims) >= min_elems:
            a = numpy_helper.to_array(init)
            h = numpy_helper.from_array(a.astype(np.float16), init.name + '__f16')
            new_inits.append(h)
            casts.append(helper.make_node('Cast', [h.name], [init.name], to=TensorProto.FLOAT,
                                          name=init.name + '__cast'))
            n += 1
        else:
            new_inits.append(init)
    del g.initializer[:]
    g.initializer.extend(new_inits)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print(f'{src} -> {dst}: {n} weights to fp16')


if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2])
