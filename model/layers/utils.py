import numpy as np


def score_model(y_hat, y, kkc_labels):
    kkc_labels = kkc_labels.numpy()
    new_y = np.zeros(shape=y.shape, dtype=int)
    gmm_map = { -1: -1}
    for i in kkc_labels:
        y_hat_label = y_hat[(y == i).nonzero()]
        new_label = np.argmax(np.bincount(y_hat_label)) 

        new_y[(y_hat == i).nonzero()] = new_label
        gmm_map[i] = new_label
        
    return new_y, gmm_map