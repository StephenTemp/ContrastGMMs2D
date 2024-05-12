'''
--------------------------------------------------------------------------------
GMM_block imports
--------------------------------------------------------------------------------
'''
from model.layers.utils import score_model
from model.layers.gmm import GaussianMixture
from scipy.spatial.distance import mahalanobis

import matplotlib.pyplot as plt
import numpy as np

'''
--------------------------------------------------------------------------------
GMM_block class
--------------------------------------------------------------------------------
'''
class GMM_block:
    gmm_map = None
    model = None
    KKCs = None
    label_map = None
    threshold = None

    # initialize GMM_block
    def __init__(self, KKCs, class_map, latent_dims):
        self.KKCs = KKCs
        self.label_map = class_map
        self.num_feats = latent_dims

        # default threshold is 80%
        self.threshold = 0.8
        self.dists = []

    def gmm_loss(self, X):
        gmm = GaussianMixture(n_components=len(self.KKCs), n_features=3)
        gmm.fit(X)
        return gmm.log_likelihood
    
    # provided features and y, train a GMM and set
    # the mahalanobis threshold
    def train_GMM(self, X, y, verbose=True, conf=80):
        gmm = GaussianMixture(n_components=len(self.KKCs), n_features=3)
        gmm.fit(X)
        
        y = y.numpy().astype(int).flatten()
        self.model = gmm

        y_hat = gmm.predict(X).numpy()
        y_hat, y_map = score_model(y_hat, y, self.KKCs)
        self.gmm_map = y_map

        # OPTIONAL: show the learned Gaussians
        if verbose == True: 
            self.display_GMM(y_pred=y_hat, y=y, X_feats=X)
            
            gmm_acc = np.sum(y_hat == y) / len(y)
            print("Acc [Train]: ", gmm_acc)

        # get all diatances between points and corresponding 
        # cluster means
        y_dists = self.get_distances(X, y_hat)
        self.dists = y_dists

        # compute the [conf] percentile of distance distribution
        threshold = np.percentile(y_dists, q=conf)
        y_hat[y_dists > threshold] = -1
        self.threshold = threshold
        
        if verbose == True: 
            self.display_GMM(y_pred=y_hat, y=y, X_feats=X.numpy())


    def predict(self, X, threshold=0.80):
        if self.model == None: 
            gmm = GaussianMixture(n_components=len(self.KKCs), n_features=3)
            gmm.fit(X)
            self.model = gmm
        model = self.model

        y_hat = np.array([model.predict(X).numpy()])
        y_dists = self.get_distances(X, y_hat)

        # compute the [conf] percentile of distance distribution
        conf = np.percentile(self.dists, q=threshold)
        y_hat[(y_dists > conf).nonzero()] = -1


        for i, label in enumerate(y_hat):
            y_hat[i] = self.gmm_map[label]
        return y_hat
    
    def get_distances(self, X, y_hat):
        gmm = self.model

        y_dists = np.zeros(shape=y_hat.shape)
        gmm_means = gmm.mu.numpy()[0]
        gmm_cov = gmm.var.numpy()[0]

        for i, x_i in enumerate(X):
            x_mean = gmm_means[y_hat[i]]
            x_cov = gmm_cov[y_hat[i]]

            y_dists[i] = mahalanobis(x_i.numpy(), x_mean, x_cov)

        return y_dists
    '''
    --------------------------------------------------------------------------------
        Helper Functions
    '''

    def display_GMM(self, y_pred, y, X_feats):
        y_unique = np.unique(y_pred)
        labels = self.label_map
        # means = self.model.means_

        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        for y_i in y_unique:
            ax.scatter(X_feats[y_pred == y_i, 0], X_feats[y_pred == y_i, 1], X_feats[y_pred == y_i, 2], label=labels[y_i], marker='o')     
            # ax.scatter(means[:, 0], means[:, 1], means[:, 2], marker='x', color='red')
        ax.set_axis_off()

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        for y_i in y_unique:
            ax.scatter(X_feats[y == y_i, 0], X_feats[y == y_i, 1], X_feats[y == y_i, 2], label=labels[y_i], marker='o')     
            # ax.scatter(means[:, 0], means[:, 1], means[:, 2], marker='x', color='red')
        ax.set_axis_off()
        
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        _, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        #fig.legend(set(labels), loc="center")
        plt.show()
        return None