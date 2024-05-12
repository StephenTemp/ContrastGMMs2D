# IMPORTS
# ------------------------------------------
from data.utils import showImg2d
from model.layers.utils import score_model

import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F

from model.layers.GMM_block import GMM_block

import matplotlib.pyplot as plt
import numpy as np
# ------------------------------------------
IS_UCC = 1
IS_KKC = 0

class NovelNetwork(torch.nn.Module):
    # instance variables
    # ------------------
    model = None            # Architecture (arbitrary)
    gmm = None              # GMM model for neural outputs 
    device = None           # for GPU

    criterion = None        # Loss Criterion (network-dependent)
    feat_layer = None       # Feature map to extract from model
    known_labels = None     # Specify which labels are KKC
    threshold = None        # Distance threshold to classify as 'novel'
    dist_metric = None      # Distance metric (default: mahalanobis)

    # ----------------------------------------------------
    # Nearest-Class Mean    # (this is another baseline)
    NearCM = None
    NearCM_delta = None
    
    def __init__(self, model, known_labels, criterion, use_gpu=False):
        super().__init__()
        self.criterion = criterion
        self.known_labels = torch.tensor([i for i in range(0, len(known_labels))])
        
        self.class_map = { -1 : "novel" }
        for i, label in enumerate(known_labels): self.class_map[i] = label

        if use_gpu:
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        self.model = model
        self.gmm = GMM_block(KKCs=self.known_labels, class_map=self.class_map, latent_dims=3)
    
    # FUNCTION: extract_feats( [], str ) => []
    # SUMMARY: provided a target layer and input, return feats from
    #          the target layer
    def extract_feats(self, input, target_layer):
        model = self._modules['model']
        cur_feats = input
        for cur_layer in model._modules:
            cur_feats = model._modules[cur_layer](cur_feats)
            if cur_layer == target_layer: return cur_feats
        
        raise ValueError("[Feature Extraction] Request layer not found!")

    def to_novel(self, input, label_kkc=False):
        input = input.to(dtype=int)
        bools = torch.tensor([x not in self.known_labels for x in input])

        if not label_kkc: input[bools] = IS_UCC
        else: return bools.to(dtype=int)

    def predict(self, input, is_train=True, threshold=0.80):
        model = self._modules['model']
        gmm = self.gmm

        scores = model(input)
        if is_train == False: scores = torch.mean(scores, axis=0)[None, :]
        return gmm.predict(scores, threshold=threshold), scores
    
    # FUNCTION: augment( x ) => x'
    # SUMMARY: provided some samples, augment them with respect to 
    #          color, rotation, etc.
    def augment(self, x):
        transforms = torch.nn.Sequential(
            torchvision.transforms.RandomCrop(x.shape[2]),
            torchvision.transforms.ColorJitter(),
        )

        trans = torch.jit.script(transforms)
        aug_x = trans(x)
        return aug_x

    # FUNCTION: train( y, {} ) => void
    # SUMMARY: train the network, then fit a GMM to a sample of the 
    #          training data features
    def train(self, train_data, val_data, args, print_info=False):
        self.feat_layer = args['feat_layer']
        criterion = self.criterion()
        if 'dist_metric' in args:
            self.dist_metric = args['dist_metric']
        else: self.dist_metric = 'mahalanobis'

        print_every = args['print_every']
        epochs = args['epoch']
        lr = args['lr']
        lambda_const = args['lambda']

        # train neural network
        # ---------------------
        model = self._modules['model']
        device = self.device
        print("Using device: ", device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model = model.to(device=self.device)  # move the model parameters to CPU/GPU

        cur_gmm = GMM_block(KKCs=self.known_labels, class_map=self.class_map, latent_dims=3)
        for _ in range(epochs):
            for t, (x, y) in enumerate(train_data):
                model.train()  # put model to training mode
                x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=torch.long)
                
                # Data Augmentation
                aug_x = self.augment(x)

                scores = model(x)
                scores_aug = model(aug_x)

                gmm_loss = cur_gmm.gmm_loss(scores)

                feats = torch.zeros(size=(scores.shape[0], 2, scores.shape[1]))
                feats[:, 0] = scores
                feats[:, 1] = scores_aug

                loss = (1 - lambda_const) * criterion(feats, labels=y) + (lambda_const) * gmm_loss

                # Zero out all of the gradients for the variables which the optimizer
                # will update.
                optimizer.zero_grad()

                # This is the backwards pass: compute the gradient of the loss with
                # respect to each  parameter of the model.
                loss.backward()

                # Actually update the parameters of the model using the gradients
                # computed by the backwards pass.
                optimizer.step()

                if t % print_every == 0 and print_info==True:
                    print('Iteration %d, loss = %.4f' % (t, loss.item()))
                    #self.check_accuracy(val_data, model)
                    print()
        
        # Save the trained model
        torch.save(model.state_dict(), "model/saved/mvcnn_weights")
        # run coarse search over gaussian mixture model
        # ---------------------------------------------
        print(train_data)
        X_feats, y = self.GMM_batch(train_data, train=True)

        train_acc = cur_gmm.train_GMM(X_feats, y)
        self.gmm = cur_gmm
        
        # Set the mahalanobis threshold 
        # ------------------------------
        return train_acc
    

    # FUNCTION: GMM_batch
    # SUMMARY: Use the network to produce embeddings, then max-pool the embeddings 
    #          per-object for GMM processing
    # ----------------------------------------------------------------------------
    def GMM_batch(self, loader, train=False, verbose=False):

        X_feats = torch.tensor([])
        X = torch.tensor([])
        y = torch.tensor([], dtype=torch.float32)
        for i in range(0, 10):
            cur_batch = iter(loader)
            X_batch, y_batch = next(cur_batch)

            if verbose: showImg2d(cur_batch)

            X_batch = X_batch.to(self.device, torch.float32)
            
            if not train:
                X_flat = torch.flatten(X_batch, end_dim=1)
                scores = self._modules['model'](X_flat)
                scores, _ = torch.mean(scores, dim=0)
                scores = torch.unsqueeze(scores, 0) # add dummy dimension
            else: 
                scores = self._modules['model'](X_batch)
            
            
            y = torch.cat((y.cpu(), y_batch.cpu()), axis=0)
            if i == 0: X_feats = X_feats.reshape((0, scores.shape[1]))
            X_feats = torch.cat((X_feats, scores.detach().cpu()), axis=0)

        return X_feats, y


    def check_accuracy(self, loader, get_wrong=False, OWL=False, threshold=0.8):
        model = self._modules['model']
        gmm = self.gmm
        device = self.device

        if loader.dataset.train: print('Checking accuracy on validation set')
        else: print('Checking accuracy on test set')   
        num_correct = 0
        num_samples = 0
        
        all_samples, all_preds, all_y = [], [], []
        
        wrong_imgs = []
        wrong_labels = []
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for t, (x, y) in enumerate(loader):
                x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=torch.long)

                preds, x_new = self.predict(x[0], is_train=False, threshold=threshold)
                all_samples.append(x_new.numpy()[0])
                all_preds.append(preds[0])
                all_y.append(y.numpy()[0])

                num_correct += (preds == y.numpy()).sum()
                num_samples += preds.shape[0]
            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
         
        if get_wrong == True: 
            return wrong_imgs, np.array(wrong_labels)

        # convert to numpy arrays for cleanliness...
        all_preds = np.array(all_preds)
        all_y = np.array(all_y).flatten()
        all_samples = np.array(all_samples)

        # Novelty Detection
        if OWL == False:
            all_y[all_y > max(self.known_labels.numpy())] = -1
            novel_acc = self.novel_eval(all_preds, all_y)

            plt.figure(3)
            plt.plot([0, 0.25, 0.50, 0.75, 1], list(novel_acc.values()))
            plt.title("Novelty Accuracy by Lambda Constant")
           #plt.show()

        # display outputs
        gmm.display_GMM(all_preds, all_y, all_samples)
        return novel_acc

    def novel_eval(self, y_hat, y):
        all_acc = {}
        novel_lambdas = [0, 0.25, 0.50, 0.75, 1]

        # compute novel accuracy
        y_novel = y[(y == -1).nonzero()]
        y_hat_novel = y_hat[(y == -1).nonzero()]
        novel_acc = np.sum(y_novel == y_hat_novel)/len(y_novel)

        # compute KKC accuracy
        y_KKC = y[(y != -1).nonzero()]
        y_hat_KKC = y_hat[(y != -1).nonzero()]
        KKC_acc = np.sum(y_KKC == y_hat_KKC)/len(y_KKC)

        for lamb in novel_lambdas: all_acc[lamb] = lamb * KKC_acc + (1 - lamb) * novel_acc     
        return all_acc