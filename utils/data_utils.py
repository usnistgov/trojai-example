import os
import numpy as np

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectFromModel

def _vectorize(x, y):
    vectorizer = DictVectorizer()
    x = vectorizer.fit_transform(x)
    y = np.asarray(y)
    return x, y, vectorizer


def load_drebin_dataset(selected, data_dir, **kwargs):
    """ Vectorize and load the Drebin dataset.

    :param selected: (bool) if true return feature subset selected with Lasso
    :return:
    """

    if selected:
        x_train_file = os.path.join(data_dir, 'x_train_sel.npy')
        y_train_file = os.path.join(data_dir, 'y_train_sel.npy')
        i_train_file = os.path.join(data_dir, 'i_train_sel.npy')
        x_test_file = os.path.join(data_dir, 'x_test_sel.npy')
        y_test_file = os.path.join(data_dir, 'y_test_sel.npy')
        i_test_file = os.path.join(data_dir, 'i_test_sel.npy')
        s_feat_file = os.path.join(data_dir, 's_feat_sel.npy')

    else:
        x_train_file = os.path.join(data_dir, 'x_train.npy')
        y_train_file = os.path.join(data_dir, 'y_train.npy')
        i_train_file = os.path.join(data_dir, 'i_train.npy')
        x_test_file = os.path.join(data_dir, 'x_test.npy')
        y_test_file = os.path.join(data_dir, 'y_test.npy')
        i_test_file = os.path.join(data_dir, 'i_test.npy')
    vec_file = os.path.join(data_dir, 'vectorizer.pkl')

    # First check if the processed files are already available,
    # load them directly if available.
    if os.path.isfile(x_train_file) and os.path.isfile(y_train_file) and \
            os.path.isfile(i_train_file) and os.path.isfile(x_test_file) and \
            os.path.isfile(y_test_file) and os.path.isfile(i_test_file) and \
            os.path.isfile(vec_file):

        if selected:
            x_train = np.load(x_train_file, allow_pickle=True)
            x_test = np.load(x_test_file, allow_pickle=True)

        else:
            x_train = np.load(x_train_file, allow_pickle=True).item()
            x_test = np.load(x_test_file, allow_pickle=True).item()

        y_train = np.load(y_train_file, allow_pickle=True)
        y_test = np.load(y_test_file, allow_pickle=True)

        return x_train, y_train, x_test, y_test

    print('Could not find Drebin processed data files - vectorizing')

    d_dir = os.path.join(data_dir, 'feature_vectors')
    d_classes = os.path.join(data_dir, 'sha256_family.csv')

    d_all_sha = sorted(os.listdir(d_dir))
    d_mw_sha = sorted(pd.read_csv(d_classes)['sha256'])

    d_x_raw = []
    d_y_raw = []

    for fn in d_all_sha:
        cls = 1 if fn in d_mw_sha else 0

        with open(os.path.join(d_dir, fn)) as f:
            ls = f.readlines()
            ls = [l.strip() for l in ls if l.strip()]

            d_x_raw.append(dict(zip(ls, [1] * len(ls))))
            d_y_raw.append(cls)

    assert len(d_x_raw) == len(d_y_raw)
    assert len(d_x_raw) == 129013

    d_x, d_y, vectorizer = _vectorize(d_x_raw, d_y_raw)

    d_train_idxs, d_test_idxs = train_test_split(
        range(d_x.shape[0]),
        stratify=d_y,
        test_size=0.33,
        random_state=42
    )

    x_train = d_x[d_train_idxs]
    x_test = d_x[d_test_idxs]
    y_train = d_y[d_train_idxs]
    y_test = d_y[d_test_idxs]

    if selected:
        sel_ = SelectFromModel(LogisticRegression(
            C=1,
            penalty='l1',
            solver='liblinear',
            random_state=42
        ))
        sel_.fit(x_train, y_train)
        f_sel = sel_.get_support()
        n_f_sel = sum(f_sel)
        # noinspection PyTypeChecker
        print('Num features selected: {}'.format(n_f_sel))

        x_train = x_train[:, f_sel].toarray()
        x_test = x_test[:, f_sel].toarray()
        assert x_train.shape[1] == n_f_sel
        assert x_test.shape[1] == n_f_sel
        np.save(s_feat_file, f_sel)

    np.save(x_train_file, x_train)
    np.save(y_train_file, y_train)
    np.save(i_train_file, d_train_idxs)
    np.save(x_test_file, x_test)
    np.save(y_test_file, y_test)
    np.save(i_test_file, d_test_idxs)
    #joblib.dump(vectorizer, vec_file)

    return x_train, y_train, x_test, y_test