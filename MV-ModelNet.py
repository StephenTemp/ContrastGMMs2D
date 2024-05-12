'''
MV-ModelNet.py: Running a contrastive network on ModelNet40 dataset using a 
                Multi-View CNN approach
'''

# IMPORTS 
# ---------------------------
from model.NovelNetwork import NovelNetwork
from model.layers.SupConLoss import SupConLoss
from model.layers.MVNet import MV_CNN
import data.utils

import torch
import numpy as np
from collections import OrderedDict
# ---------------------------

# CONSTANTS
# ---------------------------
BATCH_SIZE = 64
KKC = ["airplane", "toilet", "guitar"]  # Classes seen during training and test time (KKC)
UUC = ["car"]                           # Classes not seen during training time (UUC)
ALL_CLASSES = KKC + UUC                 # All classes
PERC_VAL = 0.20                         # Percent of data for validation 
# ---------------------------

def MV_ModelNet(LATENT_DIMS=3):
    print("Collecting ModelNet data . . .\n")
    MODELNET = data.utils.get_ModelNet(KKC=KKC, ALL=ALL_CLASSES, BATCH_SIZE=BATCH_SIZE)
    print(". . . done!")

    # Show examples from MNIST
    data.utils.showImg2d(MODELNET['TRAIN'])
    # Set the device (CUDA compatibility needs to be added to NovelNetwork.py)
    USE_GPU = True
    dtype = torch.float32 # we will be using float throughout this tutorial
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')


    # Constant to control how frequently we print train loss
    print_every = 100
    print('using device:', device)

    # Define the layers
    layers = MV_CNN(LATENT_DIMS)

    # hyperparameters for our model
    args = {
    'print_every' : 100,
    'feat_layer'  : 'fc',
    'feat_sample' : 50,
    'dist_metric' : 'mahalanobis',
    'epoch' : 10,
    'lr' : 5e-4,
    'train_model' : False,
    'saved_weights' : "model/saved/mvcnn_weights_lambda_p25",
    "lambda" : 0.15
    }

    # Run the model
    # ------------------------------------------------
    new_model = NovelNetwork(layers, known_labels=np.array(KKC), criterion=SupConLoss)

    if args['train_model'] == True:
        new_model.train(MODELNET['TRAIN'], MODELNET['VAL'], args, print_info=True)
    else: 
        new_model._modules['model'].load_state_dict(torch.load(args["saved_weights"]))
        # Then train the GMM seperately
        X_feats, y = new_model.GMM_batch(MODELNET['TRAIN'], train=True)
        train_acc = new_model.gmm.train_GMM(X_feats, y)

    # ------------------------------------------------

    # Evaluate the model on the test set
    test_thresholds = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.5]
    best_acc = { 0.5 : 0 }
    best_threshold = None
    for threshold in test_thresholds:
        accs = new_model.check_accuracy(MODELNET['VAL'], threshold=threshold)
        print("Threshold: ", threshold, " | ", accs)
        mean_acc = accs[0.5]
        if mean_acc > best_acc[0.5]:
            best_acc = accs
            best_threshold = threshold

    print('Best Mean Accuracy: ', best_acc)
    print('Best Threshold: ', best_threshold)

    test = 5


if __name__ == "__main__":
    MV_ModelNet()