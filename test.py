"""
@author supermantx
@date 2024/7/15 16:43
"""
import numpy as np
import MinkowskiEngine as ME


def print_title(s, data):
    print('='*20, s, '='*20)
    print(data)

cb0 = 100 * np.random.uniform(0, 1, (10, 3))
feat0 = np.ones((10, 1), dtype=np.float32)
cb1 = 100 * np.random.uniform(0, 1, (6, 3))
feat1 = np.ones((6, 1), dtype=np.float32)
coords, feats = ME.utils.sparse_collate([cb0, cb1], [feat0, feat1])
print_title('coords', coords)
print_title('feats', feats)
input = ME.SparseTensor(feats, coordinates=coords)
print_title('input', input)