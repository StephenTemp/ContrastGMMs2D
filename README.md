# Contrastive-GMMs for Open-World Learning
**Summary**: leverage Gaussian mixture models to clarify semi-supervised contrastive embeddings in open-set conditions.We present a light-weight, deep contrastive network which utilizes a Gaussian mixture model to identify class instances unseen at train time. Our model compares favorably with the state-of-the-art on _ModelNet40_, and we perform an ablation study to determine the impact of our newly-introduced Gaussian loss term on both the learned embeddings and model performance. Last, we discuss the potential left un-analyzed in our empirical work here, and provide direction for future work


## How to Run
A sample of _ModelNet40_ is included--unzipped--in the ``data`` folder. To run the model on the following classes (which includes ```airplane, bed, car, guitar, and toilet```). Run the ```MV-ModelNet.py``` file to reproduce our results reported in our paper. 

## Results
The Open-World Learning literature is nearly vacant with respect to 3D data; however, we adapt many of its conventions for analyzing known and unknown data classes. For instance, we implement an accuracy metric to scale priority given to novel instances vs. known instances; that is, our reported accuracy (NA) is a mixture of the novel (N) and known (K) accuracy metrics:

$$NA = \lambda_r(ACC_{K}) + (1 - \lambda_r)(ACC_{N})$$

In practice, however, we scale the Gaussian term with a constant $\delta$ to identify an ideal trade-off between the Gaussian and contrastive objectives.

| δ    | λ = 0 | λ = 0.25 | λ = 0.50 | λ = 0.75 | λ = 1 |
| -----| ----- | -------- | -------- | -------- | ----- |
| 0    | 1     | 0.85     | 0.71     | 0.58     | 0.43  |
| 0.25 | 0.7   | 0.72     | 0.74     | 0.76     | 0.78  |
| 0.5  | 0.7   | 0.7      | 0.71     | 0.71     | 0.72  |

We examine the impact of the GMM loss term both qualitatively and quantitatively: **Figure 2** (below) shows our model embeddings with and without the GMM loss term. Embeddings trained without the Gaussian loss term are, expectedly, less Gaussian, which we confirm empirically in the **Table above**. Both findings suggest a trade-off between the Gaussian and contrastive objectives, where the proper δ is selected empirically over the validation set, optimizing for $λ = 0.5$.

![Results](plots/embeddings.png)

We compare our model results to that of (a) confidence
thresholding, a simple but general open-set classification
method, and (b) Clip2Point, a recent adaptation of the zero-shot method CLIP to 3D [5, 7]. We implement thresholding from scratch, training another MV-CNN with an identical architecture but capped with a softmax layer. We then
threshold the output probabilities $p(i)$ by some $t$ such that if $p(i) < t$ we label the instance as being novel.

![Results](plots/exp-plot.jpeg)

The figure above (**Figure 3** in the report) shows the results of our experiment, which are compelling since we can achieve comparable accuracy to Clip2Point; however, we re-emphasize that these results cover a limited space of the Open-World problem. We observed instances of only one novel class and thus do not share the burden of reconciling rival novel classes as Clip2Point does, since this was out of the scope of this paper.