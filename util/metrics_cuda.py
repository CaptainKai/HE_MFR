import torch
import numpy as np

def find_thresholds_by_FAR(score_vec, label_vec, FARs=None, epsilon=1e-8):
    assert len(score_vec.shape) == 1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == np.bool
    
    score_vec = torch.from_numpy(score_vec).cuda()
    label_vec = torch.from_numpy(label_vec).cuda()

    score_neg = score_vec[~label_vec]
    score_neg = score_neg.sort(descending=True)[0]
    num_neg = len(score_neg)

    assert num_neg >= 1

    if FARs is None:
        thresholds = torch.unique(score_neg)
        thresholds = torch.cat([thresholds[0].unsqueeze(0) + epsilon, thresholds, thresholds[-1].unsqueeze(0) - epsilon])
    else:
        FARs = torch.tensor(FARs)
        num_false_alarms = torch.round(num_neg * FARs).to(torch.int32)

        thresholds = []
        for num_false_alarm in num_false_alarms:
            if num_false_alarm == 0:
                threshold = score_neg[0] + epsilon
            else:
                threshold = score_neg[num_false_alarm - 1]
            thresholds.append(threshold)
        thresholds = torch.tensor(thresholds)

    return thresholds.cpu().numpy()


def ROC(score_vec, label_vec, thresholds=None, FARs=None, get_false_indices=False):
    ''' Compute Receiver operating characteristic (ROC) with a score and label vector.
    '''
    assert score_vec.ndim == 1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == np.bool
    
    score_vec = torch.from_numpy(score_vec).cuda()
    label_vec = torch.from_numpy(label_vec).cuda()

    if thresholds is None:
        score_vec_np = score_vec.cpu().numpy()
        label_vec_np = label_vec.cpu().numpy()
        thresholds = find_thresholds_by_FAR(score_vec_np, label_vec_np, FARs=FARs)

    assert len(thresholds.shape) == 1 
    if thresholds.size > 10000:
        print('Number of thresholds (%d) is very large, computation may take a long time!' % thresholds.size)

    TARs = np.zeros(thresholds.shape[0])
    FARs = np.zeros(thresholds.shape[0])
    false_accept_indices = []
    false_reject_indices = []
    for i, threshold in enumerate(thresholds):
        accept = score_vec >= threshold
        TARs[i] = accept[label_vec].float().mean().item()
        FARs[i] = accept[~label_vec].float().mean().item()
        if get_false_indices:
            false_accept_indices.append(torch.nonzero(accept & (~label_vec)).flatten().cpu().numpy())
            false_reject_indices.append(torch.nonzero((~accept) & label_vec).flatten().cpu().numpy())
        if thresholds.size > 10000 and i % 10000 == 0:
            print("%dth threshold computation done!" % i)

    if get_false_indices:
        return TARs, FARs, thresholds, false_accept_indices, false_reject_indices
    else:
        return TARs, FARs, thresholds

def ROC_by_mat(score_mat, label_mat, thresholds=None, FARs=None, get_false_indices=False, triu_k=None):
    ''' Compute ROC using a pairwise score matrix and a corresponding label matrix.
        A wapper of ROC function.
    '''
    assert score_mat.ndim == 2
    assert score_mat.shape == label_mat.shape
    assert label_mat.dtype == np.bool

    # Move data to GPU
    score_mat = torch.from_numpy(score_mat).cuda()
    label_mat = torch.from_numpy(label_mat).cuda()

    # Convert into vectors
    m, n = score_mat.shape
    if triu_k is not None:
        assert m == n, "If using triu for ROC, the score matrix must be a square matrix!"
        triu_indices = torch.triu_indices(m, n, offset=triu_k)
        score_vec = score_mat[triu_indices[0], triu_indices[1]]
        label_vec = label_mat[triu_indices[0], triu_indices[1]]
    else:
        score_vec = score_mat.flatten()
        label_vec = label_mat.flatten()

    # Convert to numpy arrays
    score_vec = score_vec.cpu().numpy()
    label_vec = label_vec.cpu().numpy()

    # Compute ROC
    if get_false_indices:
        TARs, FARs, thresholds, false_accept_indices, false_reject_indices = \
            ROC(score_vec, label_vec, thresholds, FARs, True)
    else:
        TARs, FARs, thresholds = ROC(score_vec, label_vec, thresholds, FARs, False)

    # Convert false accept/reject indices into [row, col] indices
    if get_false_indices:
        rows, cols = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
        rc = np.stack([rows, cols], axis=2)
        if triu_k is not None:
            rc = rc[triu_indices[0], triu_indices[1], :]
        else:
            rc = rc.reshape([-1, 2])

        for i in range(len(FARs)):
            false_accept_indices[i] = rc[false_accept_indices[i]]
            false_reject_indices[i] = rc[false_reject_indices[i]]
        return TARs, FARs, thresholds, false_accept_indices, false_reject_indices
    else:
        return TARs, FARs, thresholds
