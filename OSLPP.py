import math
from sklearn.decomposition import PCA
import scipy
import numpy as np
import scipy.io
import scipy.linalg
from dataclasses import dataclass

def _load_tensors(domain):
    mapping = {
        'art': 'Art',
        'clipart': 'Clipart',
        'product': 'Product',
        'real_world': 'RealWorld'
    }
    mat = scipy.io.loadmat(f'mats/OfficeHome-{mapping[domain]}-resnet50-noft.mat')
    features, labels = mat['resnet50_features'], mat['labels']
    features, labels = features[:,:,0,0], labels[0]
    assert len(features) == len(labels)
    # features, labels = torch.tensor(features), torch.tensor(labels)
    # features = torch.load(f'./data_handling/features/OH_{domain}_features.pt')
    # labels = torch.load(f'./data_handling/features/OH_{domain}_labels.pt')
    return features, labels

def create_datasets(source, target, num_src_classes, num_total_classes):
    src_features, src_labels = _load_tensors(source)
    idxs = src_labels < num_src_classes
    src_features, src_labels = src_features[idxs], src_labels[idxs]

    tgt_features, tgt_labels = _load_tensors(target)
    idxs = tgt_labels < num_total_classes
    tgt_features, tgt_labels = tgt_features[idxs], tgt_labels[idxs]
    tgt_labels[tgt_labels >= num_src_classes] = num_src_classes

    assert (np.unique(src_labels) == np.arange(0, num_src_classes)).all()
    assert (np.unique(tgt_labels) == np.arange(0, num_src_classes+1)).all()
    assert len(src_features) == len(src_labels)
    assert len(tgt_features) == len(tgt_labels)

    return (src_features, src_labels), (tgt_features, tgt_labels)

def get_l2_norm(features:np.ndarray): return np.sqrt(np.square(features).sum(axis=1)).reshape((-1,1))

def get_l2_normalized(features:np.ndarray): return features / get_l2_norm(features)

def get_PCA(features, dim):
    result = PCA(n_components=dim).fit_transform(features)
    assert len(features) == len(result)
    return result

def get_W(labels,):
    W = (labels.reshape(-1,1) == labels).astype(np.int)
    negative_one_idxs = np.where(labels == -1)[0]
    W[:,negative_one_idxs] = 0
    W[negative_one_idxs,:] = 0
    return W

def get_D(W): return np.eye(len(W), dtype=np.int) * W.sum(axis=1)

def fix_numerical_assymetry(M): return (M + M.transpose()) * 0.5

def get_projection_matrix(features, labels, proj_dim):
    N, d = features.shape
    X = features.transpose()
    
    W = get_W(labels)
    D = get_D(W)
    L = D - W

    A = fix_numerical_assymetry(np.matmul(np.matmul(X, D), X.transpose()))
    B = fix_numerical_assymetry(np.matmul(np.matmul(X, L), X.transpose()) + np.eye(d))
    assert (A.transpose() == A).all() and (B.transpose() == B).all()

    w, v = scipy.linalg.eigh(A, B)
    assert w[0] < w[-1]
    w, v = w[-proj_dim:], v[:, -proj_dim:]
    assert np.abs(np.matmul(A, v) - w * np.matmul(B, v)).max() < 1e-5

    w = np.flip(w)
    v = np.flip(v, axis=1)

    for i in range(v.shape[1]):
        if v[0,i] < 0:
            v[:,i] *= -1
    return v

def project_features(P, features):
    # P: pca_dim x proj_dim
    # features: N x pca_dim
    # result: N x proj_dim
    return np.matmul(P.transpose(), features.transpose()).transpose()

def get_centroids(features, labels): 
    centroids = np.stack([features[labels == c].mean(axis=0) for c in np.unique(labels)], axis=0)
    centroids = get_l2_normalized(centroids)
    return centroids

def get_dist(f, features):
    return get_l2_norm(f - features)

def get_closed_set_pseudo_labels(features_S, labels_S, features_T):
    centroids = get_centroids(features_S, labels_S)
    dists = np.stack([get_dist(f, centroids)[:,0] for f in features_T], axis=0)
    pseudo_labels = np.argmin(dists, axis=1)
    pseudo_probs = np.exp(-dists[np.arange(len(dists)), pseudo_labels]) / np.exp(-dists).sum(axis=1)
    return pseudo_labels, pseudo_probs

def select_initial_rejected(pseudo_probs, n_r):
    is_rejected = np.zeros((len(pseudo_probs),), dtype=np.int)
    is_rejected[np.argsort(pseudo_probs)[:n_r]] = 1
    return is_rejected

def select_closed_set_pseudo_labels(pseudo_labels, pseudo_probs, t, T):
    if t >= T: t = T - 1
    selected = np.zeros_like(pseudo_labels)
    for c in np.unique(pseudo_labels):
        idxs = np.where(pseudo_labels == c)[0]
        Nc = len(idxs)
        if Nc > 0:
            class_probs = pseudo_probs[idxs]
            class_probs = np.sort(class_probs)
            threshold = class_probs[math.floor(Nc*(1-t/(T-1)))]
            idxs2 = idxs[pseudo_probs[idxs] > threshold]
            assert (selected[idxs2] == 0).all()
            selected[idxs2] = 1
    return selected    

def update_rejected(selected, rejected, features_T):
    unlabeled = (selected == 0) * (rejected == 0)
    new_is_rejected = rejected.copy()
    for idx in np.where(unlabeled)[0]:
        dist_to_selected = get_dist(features_T[idx], features_T[selected == 1]).min()
        dist_to_rejected = get_dist(features_T[idx], features_T[rejected == 1]).min()
        if dist_to_rejected < dist_to_selected:
            new_is_rejected[idx] = 1
    return new_is_rejected

def evaluate(predicted, labels, num_src_classes):
    acc_unk = (predicted[labels == num_src_classes] == labels[labels == num_src_classes]).mean()
    accs = [(predicted[labels == c] == labels[labels == c]).mean() for c in range(num_src_classes)]
    acc_common = np.array(accs).mean()
    hos = 2 * acc_unk * acc_common / (acc_unk + acc_common)
    _os = np.array(accs+[acc_unk]).mean()
    return f'OS={_os*100:.2f} OS*={acc_common*100:.2f} unk={acc_unk*100:.2f} HOS={hos*100:.2f}'

@dataclass
class Params:
    pca_dim: int # = 512
    proj_dim: int # = 128
    T: int # = 10
    n_r: int #  = 1200
    dataset: str # = 'OfficeHome'
    source: str # = 'art'
    target: str # = 'clipart'
    num_src_classes: int # = 25
    num_total_classes: int # = 65

def do_l2_normalization(feats_S, feats_T):
    feats_S, feats_T = get_l2_normalized(feats_S), get_l2_normalized(feats_T)
    assert np.abs(get_l2_norm(feats_S) - 1.).max() < 1e-5
    assert np.abs(get_l2_norm(feats_T) - 1.).max() < 1e-5
    return feats_S, feats_T

def do_pca(feats_S, feats_T, pca_dim):
    feats = np.concatenate([feats_S, feats_T], axis=0)
    feats = get_PCA(feats, pca_dim)
    feats_S, feats_T = feats[:len(feats_S)], feats[len(feats_S):]
    print('data shapes: ', feats_S.shape, feats_T.shape)
    return feats_S, feats_T

def center_and_l2_normalize(zs_S, zs_T):
    # center
    zs_mean = np.concatenate((zs_S, zs_T), axis=0).mean(axis=0).reshape((1,-1))
    zs_S = zs_S - zs_mean
    zs_T = zs_T - zs_mean
    # l2 normalize
    zs_S, zs_T = do_l2_normalization(zs_S, zs_T)
    return zs_S, zs_T

def main(params:Params):
    (feats_S, lbls_S), (feats_T, lbls_T) = create_datasets(params.source, params.target, params.num_src_classes, params.num_total_classes)
    print(len(lbls_S), len(lbls_T))

    # l2 normalization and pca
    feats_S, feats_T = do_l2_normalization(feats_S, feats_T)
    feats_S, feats_T = do_pca(feats_S, feats_T, params.pca_dim)
    feats_S, feats_T = do_l2_normalization(feats_S, feats_T)

    # initial
    feats_all = np.concatenate((feats_S, feats_T), axis=0)
    pseudo_labels = -np.ones_like(lbls_T)
    rejected = np.zeros_like(pseudo_labels)

    # iterations
    for t in range(1, params.T+1):
        P = get_projection_matrix(feats_all, np.concatenate((lbls_S, pseudo_labels), axis=0), params.proj_dim)
        proj_S, proj_T = project_features(P, feats_S), project_features(P, feats_T)
        proj_S, proj_T = center_and_l2_normalize(proj_S, proj_T)

        pseudo_labels, pseudo_probs = get_closed_set_pseudo_labels(proj_S, lbls_S, proj_T)
        selected = select_closed_set_pseudo_labels(pseudo_labels, pseudo_probs, t, params.T)
        selected = selected * (1-rejected)

        if t == 2:
            print(f'selecting initial rejected samples at t={t}')
            rejected = select_initial_rejected(pseudo_probs, params.n_r)
        if t >= 2:
            rejected = update_rejected(selected, rejected, proj_T)
        selected = selected * (1-rejected)

        pseudo_labels[selected == 0] = -1
        pseudo_labels[rejected == 1] = -2

    # final pseudo labels
    pseudo_labels[pseudo_labels == -2] = params.num_src_classes
    assert (pseudo_labels != -1).all()

    # evaluation
    print(evaluate(pseudo_labels, lbls_T, params.num_src_classes))


if __name__ == '__main__':
    params = Params(pca_dim=512, proj_dim=128, T=10, n_r=1200, 
                  dataset='OfficeHome', source='art', target='clipart',
                  num_src_classes=25, num_total_classes=65)
    main(params)
