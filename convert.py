import os
import numpy as np
from plyfile import PlyData, PlyElement

import os
import time
import numpy as np
from plyfile import PlyData, PlyElement


def convert_fast(data, path, binary=True):
    """
    Fast Gaussian .ply export.

    data layout:
      0:3   xyz
      3:51  SH, where 3:6 -> f_dc, 6:51 -> f_rest
      51    opacity
      52:55 scale
      55:59 rotation
    """
    t0 = time.time()

    print(f"[DEBUG] saving to: {path}")
    print(f"[DEBUG] data shape: {data.shape}")

    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)

    # 保证是 CPU numpy float32 contiguous
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()

    data = np.asarray(data, dtype=np.float32)
    data = np.ascontiguousarray(data)

    if data.ndim != 2 or data.shape[1] < 59:
        raise ValueError(f"Expected data shape [N, >=59], got {data.shape}")

    n = data.shape[0]

    names = (
        ["x", "y", "z"]
        + [f"f_dc_{i}" for i in range(3)]
        + [f"f_rest_{i}" for i in range(45)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )

    dtype_full = [(name, "f4") for name in names]
    elements = np.empty(n, dtype=dtype_full)

    # 关键：逐字段 vectorized 赋值，避免 list(map(tuple, ...))
    elements["x"] = data[:, 0]
    elements["y"] = data[:, 1]
    elements["z"] = data[:, 2]

    for i in range(3):
        elements[f"f_dc_{i}"] = data[:, 3 + i]

    for i in range(45):
        elements[f"f_rest_{i}"] = data[:, 6 + i]

    elements["opacity"] = data[:, 51]

    for i in range(3):
        elements[f"scale_{i}"] = data[:, 52 + i]

    for i in range(4):
        elements[f"rot_{i}"] = data[:, 55 + i]

    print(f"[DEBUG] structured array ready: {time.time() - t0:.2f}s")

    el = PlyElement.describe(elements, "vertex")
    PlyData([el], text=not binary).write(path)

    print(f"[DEBUG] write done: {path}")
    print(f"[DEBUG] total time: {time.time() - t0:.2f}s")
    print(f"[DEBUG] file size: {os.path.getsize(path) / 1024 / 1024:.2f} MB")



def convert(data, path):
    try:
        print(f"[DEBUG] saving to: {path}")
        print(f"[DEBUG] data shape: {data.shape}")

        parent = os.path.dirname(path)
        os.makedirs(parent, exist_ok=True)
        print(f"[DEBUG] parent exists: {os.path.exists(parent)}")

        xyz = data[:, :3]
        normals = np.zeros_like(xyz)
        f_dc = data[:, 3:6]
        f_rest = data[:, 6:51]
        opacities = data[:, 51:52]
        scale = data[:, 52:55]
        rotation = data[:, 55:59]

        def construct_list_of_attributes():
            l = ['x', 'y', 'z']
            for i in range(3):
                l.append(f'f_dc_{i}')
            for i in range(45):
                l.append(f'f_rest_{i}')
            l.append('opacity')
            for i in range(3):
                l.append(f'scale_{i}')
            for i in range(4):
                l.append(f'rot_{i}')
            return l

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, f_dc, f_rest, opacities, scale, rotation), axis=1
        )

        print(f"[DEBUG] attributes shape: {attributes.shape}")
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        print(f"[DEBUG] write done: {path}")
        print(f"[DEBUG] file exists after write: {os.path.exists(path)}")
        if os.path.exists(path):
            print(f"[DEBUG] file size: {os.path.getsize(path)}")

    except Exception as e:
        print(f"[ERROR] failed to save {path}: {e}")
        import traceback
        traceback.print_exc()

def convert_compact(data, path):
    try:
        print(f"[DEBUG] saving to: {path}")
        print(f"[DEBUG] data shape: {data.shape}")

        parent = os.path.dirname(path)
        os.makedirs(parent, exist_ok=True)
        print(f"[DEBUG] parent exists: {os.path.exists(parent)}")

        xyz = data[:, :3]
        f_dc = data[:, 3:6]
        opacities = data[:, 6:7]
        scale = data[:, 7:10]
        rotation = data[:, 10:14]

        def construct_list_of_attributes():
            l = ['x', 'y', 'z']
            for i in range(3):
                l.append(f'f_dc_{i}')
            l.append('opacity')
            for i in range(3):
                l.append(f'scale_{i}')
            for i in range(4):
                l.append(f'rot_{i}')
            return l

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, f_dc, opacities, scale, rotation), axis=1
        )

        print(f"[DEBUG] attributes shape: {attributes.shape}")
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        print(f"[DEBUG] write done: {path}")
        print(f"[DEBUG] file exists after write: {os.path.exists(path)}")
        if os.path.exists(path):
            print(f"[DEBUG] file size: {os.path.getsize(path)}")

    except Exception as e:
        print(f"[ERROR] failed to save {path}: {e}")
        import traceback
        traceback.print_exc()
