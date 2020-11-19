import dill
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score


class RenameUnpickler(dill.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "pyHGT.data" or module == 'data':
            renamed_module = "GPT_GNN.data"
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


def check_walk_length(walk, walk_length):
    min_walk_len = min([len(i) for k, v in walk.items() for i in v])
    print('min walk length:', min_walk_len)
    assert min_walk_len == walk_length


def index_mapping_dict(nodes):
    inx_dict = {v: k for k, v in enumerate(sorted(nodes))}
    inx_dict_inv = {v: k for k, v in inx_dict.items()}
    return inx_dict, inx_dict_inv


def early_stopping(log_value, best_value, stopping_step, flag_step=3):
    # early stopping strategy:
    if best_value is None or log_value > best_value:
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1
    if stopping_step >= flag_step:
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop


def ranklist_by_sort(u_score, all_items, pos_items):
    sorted_index = np.argsort(-u_score)
    sorted_score = u_score[sorted_index]
    sorted_items = all_items[sorted_index]
    r = []
    for i in sorted_items:
        if i in pos_items:
            r.append(1)
        else:
            r.append(0)
    a = auc(ground_truth=r, prediction=sorted_score)
    return r, a


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def auc(ground_truth, prediction):
    '''
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    '''

    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception as e:
        print(e)
        res = 0
    return res