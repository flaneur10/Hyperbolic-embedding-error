
from hype.graph import eval_reconstruction, load_adjacency_matrix
import argparse
import numpy as np
import torch
import os
from tqdm import tqdm
import timeit
from hype import MANIFOLDS, MODELS

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('file', help='Path to checkpoint')
parser.add_argument('-workers', default=1, type=int, help='Number of workers')
parser.add_argument('-sample', type=int, help='Sample size')
parser.add_argument('-quiet', action='store_true', default=False)
args = parser.parse_args()

chkpnt = torch.load(args.file)
dset = chkpnt['conf']['dset']
if not os.path.exists(dset):
    raise ValueError("Can't find dset!")

format = 'hdf5' if dset.endswith('.h5') else 'csv'
dset = load_adjacency_matrix(dset, format, objects=chkpnt['objects'])

sample_size = args.sample or len(dset['ids'])
sample = np.random.choice(len(dset['ids']), size=sample_size, replace=False)

adj = {}

for i in sample:
    end = dset['offsets'][i + 1] if i + 1 < len(dset['offsets']) \
        else len(dset['neighbors'])
    adj[dset['ids'][i]] = set(dset['neighbors'][dset['offsets'][i]:end])
manifold = MANIFOLDS[chkpnt['conf']['manifold']]()

manifold = MANIFOLDS[chkpnt['conf']['manifold']]()
model = MODELS[chkpnt['conf']['model']](
    manifold,
    dim=chkpnt['conf']['dim'],
    size=chkpnt['embeddings'].size(0),
    sparse=chkpnt['conf']['sparse']
)
model.load_state_dict(chkpnt['model'])

lt = chkpnt['embeddings']
if not isinstance(lt, torch.Tensor):
    lt = torch.from_numpy(lt).cuda()

#i hope to calculate the distance matrix between words

objects = np.insert(np.array(list(dset['ids'])), 13, 13)
dists_matrix = torch.tensor([])
for object in tqdm(objects):
    dists = model.energy(model.lt.weight[None, object], model.lt.weight)
    dists_matrix = torch.cat((dists_matrix, dists.unsqueeze(0)), 0)

(evals, evecs) = torch.linalg.eig(dists_matrix)

U, S, Vh = torch.svd(dists_matrix)
estimated_rank = torch.sum(S > 1).item()

print(f"应用双曲余弦函数后的张量秩估计为：{estimated_rank}")

dists_matrix_cosh = torch.cosh(dists_matrix)
U, S, Vh = torch.svd(dists_matrix_cosh)
estimated_rank = torch.sum(S > 1).item()

print(f"应用双曲余弦函数后的张量秩估计为：{estimated_rank}")

print(dists_matrix_cosh)

