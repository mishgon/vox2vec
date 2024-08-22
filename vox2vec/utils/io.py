import json
import numpy as np
from gzip import GzipFile


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_numpy(value, path, *, allow_pickle: bool = True, fix_imports: bool = True, compression: int = None,
               timestamp: int = None):
    if compression is not None:
        with GzipFile(path, 'wb', compresslevel=compression, mtime=timestamp) as file:
            return save_numpy(value, file, allow_pickle=allow_pickle, fix_imports=fix_imports)

    np.save(path, value, allow_pickle=allow_pickle, fix_imports=fix_imports)


def load_numpy(path, *, allow_pickle: bool = True, fix_imports: bool = True, decompress: bool = False):
    if decompress:
        with GzipFile(path, 'rb') as file:
            return load_numpy(file, allow_pickle=allow_pickle, fix_imports=fix_imports)

    return np.load(path, allow_pickle=allow_pickle, fix_imports=fix_imports)
