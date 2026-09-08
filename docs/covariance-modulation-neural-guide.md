# What the neural comparisons learn

Both models receive a batch B x T x M of raw recordings and predict one scalar
alpha per recording. They receive neither the hidden states nor group assignments
nor the simulator's factor loadings. There are10,20 or40 labelled recordings in a
fit, not T times as many labelled examples. Many time points help estimate what
happened within a recording; they do not replace independent labelled systems.

## Aligned-channel encoder:94,017 parameters

    raw channel sequences
      -> shared temporal CNN
      -> channel attention at each aligned temporal position
      -> temporal CNN
      -> mean/max pooling over channels and time
      -> small regression head

The shared CNN uses the same filters on every channel. Two layers map1->32->64
features with kernels7/5 and strides2/2. Each token initially summarizes about15
input time points. Attention compares channels at the same temporal position:

    Attention(H) = softmax(Q K^T / sqrt(d)) V

Q,K,V are learned linear transforms of the token features. The attention weights
are data-dependent, so a channel can use information from other channels. They
are not estimates of a causal graph. Two4-head attention blocks are followed by a
kernel3 temporal convolution, giving an effective local receptive field of about23
original samples before global pooling. The final head maps128->32->1.

No channel-position embeddings are used: channel identities are exchangeable in
this experiment. Shared processing is permutation-equivariant; pooling makes the
final prediction invariant. Temporal alignment is retained until channels have
interacted. Pooling each channel to a scalar first could erase precisely the
dependence the experiment is trying to measure.

This model can combine many channels in one interaction step. It does not perform
full attention across all M*T time-channel tokens, which would be much more costly.

## Pair-relation encoder:6,993 parameters

    every ordered pair of aligned raw channel sequences
      -> shared two-channel temporal CNN
      -> mean/SD pooling over time, separately for each pair
      -> mean/SD pooling over pairs
      -> small regression head

For each pair (i,j), the CNN takes two input sequences together. Layers2->16->32
use kernels7/5 and strides4/4, again a local receptive field of about23 samples.
Time pooling yields64 features per pair; pair pooling yields128 per recording;
the head maps128->32->1. Both orientations are included so the result is invariant
to channel relabelling even if the two-input CNN is not symmetric by itself.

This is a more constrained relational inductive bias. It can learn nonlinear
pairwise features directly from raw aligned observations, but it cannot generally
retain which edges meet at the same node after pooling. Its pair count scales as
M(M-1); sharing weights keeps the parameter count independent of M and T.
It is inspired by [relation networks](https://arxiv.org/abs/1706.01427), not a
claimed reproduction of a published time-series model.

## Why these comparisons are informative

In this generator each channel alone is white Gaussian noise for every alpha.
The changing signal lies in the joint distribution. A neural model must learn
useful joint functions from its input; SPI-SPI starts with a library of such
functions and learns a small readout. This difference in prior structure could
matter with few labels. It does not give z information unavailable from raw data.

Neither neural model is a foundation model, pretrained transformer, or exhaustive
test of neural learning. A win over both is evidence about two specific learners
under a specified budget. Fourth-moment, covariance and windowed-covariance
references are essential because they test whether much simpler known statistics
already expose the target effectively.

![Raw pair geometry](../results/covariance_modulation_260909/raw-pair-intuition.png)

This illustration holds c=.1 and d=.5 fixed, comparing alpha=0 and .9 with20,000
iid-state observations (seed260909). Each channel has the same Gaussian law and
population correlation is .1, but the fourth cross-cumulant changes from0 to.405.
It shows raw pair geometry, not a two-channel z experiment. Reproduce with
`python -m scripts.plot_covariance_modulation_intuition`.

## Training and checks

Both minimize mean squared error using AdamW. The same two learning rates and two
weight decays are compared by two-fold validation wholly inside the label budget.
Corrected early stopping uses unclipped validation MAE; the selected median best epoch determines the
final refit on all available source labels. Test recordings are not used for
preprocessing, tuning or selecting the stopping epoch. Predictions are clipped to
the target's[0,1] range at evaluation, as for statistical readouts.

The pilot runs three independent source cohorts per state process and nested
10/20/40-label subsets. Models are trained at M16/T1000 and tested at that size and
M8/T500. Shape compatibility is guaranteed by the architecture; good transfer is
not. Report selected fits reaching the epoch cap, training errors and paired test
results before interpreting weak neural performance. Successful memorization of
four training records is an optimization sanity check, not evidence of generalization.

## Gadi GPU scout (2026-09-09)

Live queues include gpuvolta, dgxa100 and gpuhopper; visibility does not by itself
verify account-specific access or driver compatibility. The current environment
imports PyTorch2.13.0+cu130 and contains kernels for sm75/80/86/90/100/120, not
sm70. The V100 gpuvolta option therefore needs a compatible PyTorch build; an
approved newer-GPU queue would still need an actual CUDA smoke test. No GPU job
was submitted and no environment changed. Local MPS already completes these
small fits efficiently, so moving active fits has no demonstrated benefit.

The first runs used clipped validation MAE; this could hide progress below zero.
A reproduced fold confirmed the issue. Corrected configurations use an unclipped
stopping signal, keep clipping for reported prediction error, and allow up to600
epochs. Original scores are retained as diagnostics and are not the final neural
comparison. A supplementary pointwise pair encoder (4753parameters) tests whether
instantaneous nonlinear features are better suited to this particular target.
