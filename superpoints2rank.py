import numpy as np
import pickle
import os
import torch
from pathlib import Path
from torch import nn
from copy import deepcopy
import pandas as pd
import shutil
import time

def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]

class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))

def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))

class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1

def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

class SuperGlue(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        assert self.config['weights'] in ['indoor', 'outdoor']
        path = Path(__file__).parent
        path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
        self.load_state_dict(torch.load(str(path), weights_only = True))

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }

        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, data['scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return {
            'matches0': indices0, # use -1 for invalid match
            'matches1': indices1, # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }

def ranking_score(matches, match_confidence):
    return np.sum(np.multiply(matches, match_confidence)).astype(np.float32)

def load_pickle(path):
    with open(path, 'rb') as file:
        loaded = pickle.load(file)
    return loaded

def process_superpoints(file, input_dir, output_dir):

    # Отримуємо ім'я файлу з його шляху
    main_file = os.path.basename(file)
    # Створюємо новий шлях для файлу в каталозі ./data
    new_file_path = os.path.join(input_dir, main_file)
    # Копіюємо файл
    shutil.copy(os.path.join(input_dir, file), new_file_path)

    # score for each image to query image
    score_dict = {}

    # Create the output directories if they do not exist already.
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    all_file_name = os.listdir(input_dir)
    total_file_num = len(all_file_name)

    pairs = [(main_file, file_name) for file_name in all_file_name if file_name.endswith('.pickle')]

    config = {
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }

    superglue = SuperGlue(config).eval().to('cpu')

    for i, pair in enumerate(pairs):
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)

        # Handle --cache logic.
        do_match = True

        if not (do_match):
            continue

        superpoints_0 = load_pickle(str(input_dir / name0))
        superpoints_1 = load_pickle(str(input_dir / name1))

        superpoints_0 = {k + '0': v for k, v in superpoints_0.items()}
        superpoints_1 = {k + '1': v for k, v in superpoints_1.items()}

        if superpoints_0 is None or superpoints_1 is None:
            exit(1)

        if do_match:
            # Perform the matching.
            # change np to torch tensor
            dummy_data = {'image0': np.zeros((1, 1, 1000, 1000)),
                          'image1': np.zeros((1, 1, 1000, 1000))}

            data = {**dummy_data, **superpoints_0, **superpoints_1}
            for k in data:
                if isinstance(data[k], (list, tuple)):
                    data[k] = torch.stack(data[k])

            # convert to ndarray to be able to save .npz
            pred = superglue(data)
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

            kpts0, kpts1 = superpoints_0['keypoints0'][0].cpu().numpy(), superpoints_1['keypoints1'][0].cpu().numpy()
            matches, conf = pred['matches0'], pred['matching_scores0']

            # Write the matches to disk.
            out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                           'matches': matches, 'match_confidence': conf}

            # save score to score dict
            score_dict[stem1] = ranking_score(matches, conf)

            # save full score to calculate %
            if name0 == name1:
                full_score = score_dict[stem1]

            # save to .npz file
            np.savez(str(matches_path), **out_matches)

    ranked_images = {k: v for k, v in sorted(score_dict.items(), reverse=True, key=lambda x: x[1])}

    ranked_images_percentage = {k: f'{((v / full_score) * 100):.3f}%' for k, v in ranked_images.items()}

    # write ranked image .csv
    df = pd.DataFrame.from_dict(ranked_images_percentage, orient='index', columns=['score'])
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'image'}, inplace=True)
    df.to_csv(str(output_dir / 'ranking_score.csv'), index=True)

    # Після завершення роботи видаляємо копію файлу
    if os.path.exists(new_file_path):
        os.remove(new_file_path)

def superpoints2rank (input_dir, output_base_dir):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    torch.set_grad_enabled(False)

    # Знаходимо всі файли .pickle в input_dir
    pickle_files = [f for f in os.listdir(input_dir) if f.endswith('.pickle')]

    input_dir = os.path.dirname(input_dir.rstrip('/'))
    # Запускаємо process_superpoints для кожного файлу
    for pickle_file in pickle_files:
        file_path = f'frame_superpoints/{pickle_file}'
        output_dir, _ = os.path.splitext(os.path.join(output_base_dir, pickle_file))
        process_superpoints(file_path, input_dir, output_dir)



