Isaac Alexis Lopez Paredes

WORK ON TUTORIALS

TUTORIAL 1 \- SUPERVISED LINEAR REGRESSION

Date: 12/03/25  
Hours: 11-13

Work progression: Installed astroML. Ran the notebook.   
Innovations: Fixed some commands in the code that didn’t allow to show the ellipses correctly

TUTORIAL 2 \- NAIVE BAYES

Date: 19/03/25  
Hours: 11-13  
Work progression: 

* Checked the labels in the documentation to know what the features mean  
* Did scatter plots and histograms using the table in the archive to explore some features in the dataset  
* I first thought of classifying planets as rocky or gaseous but then decided to classify them as “Earth-like” or “Jupiter-like”  
* Implemented the NB pipeline to classify the planets

Innovations:  
Implemented the mentioned classification (point 11 in the modified notebook in my repository):   
I found that the model correctly works for classifying Earth-like planets, however performs as a random classifier in the case of Jupiter-like planets. Then I plotted the feature distribution and found that pl\_orbper is actually bimodal for Jupiter-like planets, which breaks the assumption of Gaussian features of the model. So a possible solution could be to use a different model. Actually, when comparing with the logistic regression the model performs slightly better. 

TUTORIAL 3 \- NEURAL NETWORKS

Date: 26/03/25  
Hours: 11-3

Work progression:

* Ran the notebook on gradient descent, activation functions, loss functions, and neural networks applied to real NASA exoplanet data which:  
    * Explores the dataset from the NASA Exoplanet Archive (pscomppars table, \~4300 planets after cleaning) and inspected distributions of pl\_eqt, pl\_orbsmax, st\_teff, and st\_rad  
    * Implements single-neuron gradient descent from scratch in NumPy, comparing sigmoid, tanh, ReLU, and leaky ReLU activations against MSE, MAE, and Huber losses on a toy 1D dataset  
    * Trains a shallow PyTorch MLP (ExoplanetRegressor) to predict equilibrium temperature (pl\_eqt) and a binary classifier (ExoplanetClassifier) for a temperate rocky proxy label  
* Completed all 8 exercises in Part V of the notebook:  
  * Exercise 1: Changed the regression target to log10(pl\_rade), removing it from input features to avoid leakage; noted that stellar features are less physically motivated for radius prediction  
  * Exercise 2: Added nn.Dropout layers at rates 0, 0.2, and 0.4; compared training vs. validation curves to observe the regularisation effect  
  * Exercise 3: Built a configurable deep MLP (1 to 6 hidden layers) and tabulated RMSE and R² to check whether depth alone improves performance on this dataset  
  * Exercise 4: Ablation study removing st\_teff, log10\_st\_rad, and log10\_st\_mass one at a time; tracked RMSE increase to rank stellar feature importance  
  * Exercise 5: Replaced the simple temperature/radius proxy with an insolation-based habitable zone criterion using Stefan-Boltzmann and the Kopparapu et al. 2013 conservative HZ limits (0.36–1.11 S\_Earth), with the R\_p ≤ 1.8 R\_Earth rocky criterion  
  * Exercise 6: Investigated the \~99:1 class imbalance; compared unweighted vs. class-weighted BCE loss; showed that accuracy is a misleading metric and that F1/recall are more informative  
  * Exercise 7: Compared learning rates 1e-4, 1e-3, and 1e-2 on the regression task; plotted training and validation curves to observe convergence speed vs. stability trade-off  
  * Exercise 8: Implemented permutation feature importance by shuffling each feature on the test set and measuring ΔMSE; verified that log10\_pl\_orbsmax and st\_teff dominate, consistent with T\_eq ∝ T\_eff \* sqrt(R\_star / a)

Innovations:  
For Exercise 5, derived the insolation flux from first principles using the Stefan-Boltzmann approximation for stellar luminosity rather than using the tabulated pl\_eqt values, making the habitability criterion physically grounded and independent of the equilibrium temperature column. For Exercise 6, used BCEWithLogitsLoss(pos\_weight=...) with the exact imbalance ratio as the weight.


TUTORIAL 4 \- SUPPORT VECTOR MACHINES

Date: 02/04/25  
Hours: 11-13

Work progression:

* Ran the Hot Jupiter vs Non-Hot Jupiter SVM notebook on NASA Exoplanet Archive data (~6100 planets from pscomppars)  
* The target label is defined operationally as P < 10 days and R\_p > 8 R\_Earth  
* Ran two pre-built experiments: Experiment A (all features including pl\_orbper and pl\_rade) and Experiment B (removing those defining features) to study definition leakage  
* Completed the missing exercises:  
  * Exercise 1: Tested a stricter definition (P < 5 days and R\_p > 8 R\_Earth) — fewer positives (597 vs 739), coefficients sharpen on orbital period and radius  
  * Exercise 2: Trained on only stellar features (st\_teff, st\_rad, st\_mass, st\_met, st\_logg) — performance drops significantly, confirming that orbital architecture carries most of the signal  
  * Exercise 3: Explicit side-by-side comparison of Linear SVM vs RBF SVM across both experiments — in Experiment A both perform similarly (nearly linear boundary); in Experiment B the RBF kernel outperforms linear because indirect features create curved decision boundaries  
  * Exercise 4: Explained physically why pl\_orbsmax and pl\_eqt remain predictive without pl\_orbper/pl\_rade — Kepler's 3rd law links period to semi-major axis, and T\_eq is set by irradiation at the planet's orbit (T\_eq ∝ T\_star \* sqrt(R\_star / a))  
  * Exercise 5: Discussed whether the classifier is discovering astrophysics or recovering a human-made definition — concluded it is both: Experiment A is largely definition leakage, Experiment B captures real physical correlations (orbital architecture, stellar metallicity, irradiation)

Innovations:  
For Exercise 5, articulated the distinction between label leakage (Experiment A recovers the step-function threshold) and genuine physical signal (Experiment B achieves ROC AUC ~0.95 through orbital and stellar proxies), and proposed what a rigorous study would require: defining labels from independent measurements (e.g., dynamical mass from RV rather than radius from transit) to avoid feature overlap with the classification criterion.

TUTORIAL 5 \- RANDOM FOREST

Date: 09/04/25  
Hours: 11-13

Work progression:

* Ran the Random Forest classifier notebook on NASA Exoplanet Archive data (~6100 confirmed planets from pscomppars), predicting a 3-class radius label (rocky / sub-Neptune / giant)  
* The classification baseline achieved ~97% balanced accuracy and macro-F1 with 5-fold CV; permutation importance showed pl\_bmasse and pl\_dens as the dominant features  
* Completed the optional extension (section 13): switched from classification to regression using RandomForestRegressor  
  * Primary regression target: log10(pl\_bmasse). pl\_rade was re-added to the feature set (no longer a leakage risk since the target is not the radius class). pl\_dens was excluded to avoid algebraic leakage (density encodes mass/radius³). Cross-validated and evaluated on held-out test set with R², RMSE, and MAE. Diagnostic plots include predicted vs true and residual plots. Permutation importance confirms pl\_rade dominates via the mass-radius relation  
  * Secondary regression target: pl\_eqt (equilibrium temperature). pl\_insol was excluded (direct insolation proxy for T\_eq). The model recovers the radiative equilibrium scaling T\_eq ∝ T\_eff \* sqrt(R\_star / a), with pl\_orbsmax and st\_teff as the top features  
* Added a summary comparison table contrasting the classification and regression framings across metrics, leakage risks, interpretability tools, and scientific insight

Innovations:  
For the mass regression, identified and excluded pl\_dens as an algebraic proxy for leakage (density ∝ mass/radius³) rather than including it naively. For the T\_eq regression, excluded pl\_insol for the same reason (S ∝ L/a² and T\_eq ∝ S^0.25 make them algebraically coupled). This makes the feature importance physically interpretable: the model's top features match the expected terms in the analytical scaling laws, which can be used as a sanity check that the RF is not exploiting hidden shortcuts.

TUTORIAL 6 \- CUDA FOR ML

Date: 16/04/25  
Hours: 11-13

Work progression:

* Ran the CUDA for ML notebook covering host/device model, GPU timing with synchronisation, the CUDA execution hierarchy (threads/blocks/grids/warps), memory hierarchy, arithmetic intensity, softmax stabilisation, attention scaling, memory layout and profiling  
* Completed these exercises:  
  * Exercise 7: Built a 4-layer MLP (512→1024→1024→256→10), ran the PyTorch profiler on one forward pass; aten::mm dominates CUDA time, ReLU activations are fast and elementwise; discussed implications for attention models  
  * Exercise 8: Designed a custom experiment with a 3D tensor permuted to be non-contiguous; measured sum and matmul latency for contiguous vs non-contiguous layouts; discussed memory coalescing, warp divergence, and the implicit copy triggered by reshape on non-contiguous tensors

Innovations:  
For Exercise 8, used a 3D permute (not just a 2D transpose as in the notebook) to expose the coalescing problem more clearly, and combined a reduction (sum) with a matmul after reshape to show two distinct failure modes: strided-access overhead and implicit copy overhead respectively.

TUTORIAL 7 \- CUDA AND SPECTRA

Date: 23/04/25  
Hours: 11-13

Work progression:

* Ran the MUSCLES/Mega-MUSCLES real spectra notebook which downloads panchromatic SEDs (X-ray to IR) for 7 host stars from MAST using astroquery, reads FITS products, computes UV-band proxy integrals (UV-C 100–280 nm, UV-B 280–315 nm, UV-A 315–400 nm), resamples all spectra to a common 2048-point log-wavelength grid, and trains a GPU-accelerated dense autoencoder with augmentation to learn latent spectral representations  
* Completed all 6 exercises:  
  * Exercise 1: Extended the target slug list to 11 stars (adding hd-40307, hd-85512, hd-97658, proxima-cen); recomputed and replotted UV-band fractions to check robustness of earlier patterns across a broader stellar type range  
  * Exercise 2: Downloaded the const-res-sed (non-adapted) product for TRAPPIST-1 alongside the adapt-const-res-sed; compared UV-C/UV-B/UV-A fractions and plotted both spectra side-by-side to assess whether adaptive downsampling materially changes the astrobiology proxies  
  * Exercise 3: Defined a lightweight autoencoder (LightAE) that handles variable input dimensions; retrained separately on UV-only (\<400 nm), optical (400–700 nm), and near-IR (700–2500 nm) sub-ranges; compared 2-D PCA projections of the latent vectors for each range to identify which spectral window drives star-to-star separation  
  * Exercise 4: Swept latent dimensions d in {2, 4, 8, 16}; trained a separate DenseAutoencoder for each; plotted training loss curves and 2-D latent projections; identified the smallest d that still separates stars physically  
  * Exercise 5: Computed PCA reconstruction MSE as a function of n\_components (2, 4, 8, 16, 32, 64); compared against the autoencoder MSE; plotted the joint comparison to assess whether a nonlinear GPU model adds value over a linear PCA baseline with this dataset size  
  * Exercise 6: Wrote a scientific paragraph on TRAPPIST-1 discussing its UV-soft SED (negligible UV-C relative to warmer M dwarfs), prebiotic photochemistry implications, hydrodynamic atmospheric escape during the $\gtrsim 1$ Gyr pre-main-sequence phase, and JWST priority for planets 1e and 1f

Innovations:  
For Exercise 3, introduced a LightAE class with architecture that adapts to the variable input dimension of each wavelength sub-range (hidden layer = min(512, input\_dim/4)), avoiding the need to redesign the network for each range. For Exercise 5, showed that with only 7 real spectra PCA already captures nearly all variance linearly, so the autoencoder's advantage is limited to potential recovery of nonlinear spectral manifolds — a distinction that only becomes meaningful with larger heterogeneous datasets. The Exercise 6 paragraph explicitly connects the MUSCLES EUV/X-ray quiescent flux to the pre-main-sequence atmospheric erosion problem, distinguishing historical irradiation from the current-epoch measurement.

TUTORIAL 8 \- GRADIENT DESCENT

Date: 30/04/25  
Hours: 11-13

Work progression:

* Ran two complete demo notebooks: NASA\_Exoplanet\_Archive\_Gradients\_GD\_ML\_Demo.ipynb (derives and implements linear and logistic regression from scratch with explicit gradient expressions, gradient checking via finite differences, and comparison with sklearn) and NASA\_Exoplanet\_Archive\_ML\_Demo.ipynb (end-to-end ML pipeline using RandomForestRegressor and RandomForestClassifier on the NASA Exoplanet Archive pscomppars table, including feature engineering, data cleaning, confusion matrix, feature importance, and selection-effect analysis by discovery method)  

Innovations:  
Both notebooks are pedagogically complete. The gradients notebook explicitly derives $\nabla_w L = \frac{2}{N} X^T(\hat{y} - y)$ and $\nabla_w L = \frac{1}{N} X^T(p - y)$ analytically and verifies them with finite differences, giving a concrete numerical check on the chain-rule derivation.

TUTORIAL 9 \- PRINCIPAL COMPONENT ANALYSIS

Date: 07/05/25  
Hours: 11-13

Work progression:

* Ran the Principal Component Analysis notebook (Python Data Science Handbook chapter by Jake VanderPlas), covering PCA as dimensionality reduction, visualization of high-dimensional data (digits dataset, 64-D to 2-D), PCA for noise filtering, and eigenfaces on the Labeled Faces in the Wild dataset (LFW, 150 components preserving \>90% variance)  

Innovations:  
The eigenfaces example is a clear demonstration that PCA finds basis functions that reconstruct global image structure from \~5% of the original pixel count, directly connecting explained variance ratio to the quality of inverse-transform reconstruction.

TUTORIAL 10 \- CONVOLUTIONAL NEURAL NETWORKS

Date: 14/05/25  
Hours: 11-13

Work progression:

* Ran the CNN tutorial (John Wu, LSSTC DSFP Session 19) on predicting galaxy gas-phase metallicity $Z = 12 + \log(\mathrm{O/H})$ from SDSS $gri$ images using fastai; covered convolution mechanics, activation functions, batch normalisation, pooling, fully connected layers, forward/backward pass, SGD, Adam, and learning rate schedules  
* Exercises 1, 2, 3, and 5 already had hidden answers in \<details\> / @title cells; completed the missing Exercise 4:  
  * Exercise 4: Switched from xresnet18 to xresnet34 (more residual blocks, higher capacity) and raised the peak LR to 2e-2 with fit\_one\_cycle; the larger model converges faster and reaches valid RMSE \< 0.09 in 8 epochs; also trained a bonus DeepMerge classifier on HST+JWST Illustris simulated galaxy images using xresnet18 with CrossEntropyLossFlat, achieving \>70% classification accuracy in 10 epochs

Innovations:  
For Exercise 4, the key insight is that the one-cycle policy combined with a model with more depth (xresnet34 vs xresnet18) converges in fewer epochs because the larger model can fit the non-linear morphology-metallicity relation more efficiently at higher peak learning rates without diverging, whereas a shallower model needs more epochs to reach the same loss level.

TUTORIAL 11 \- VISION TRANSFORMER

Date: 21/05/25  
Hours: 11-13

Work progression:

* Ran the Vision Transformer notebook (Phillip Lippe, UvA DL course tutorial 15) implementing a full ViT from scratch in PyTorch Lightning: patch embedding ($32 \times 32$ CIFAR-10 images split into 64 patches of $4 \times 4$), Pre-LN attention blocks with nn.MultiheadAttention, learnable positional encodings, CLS token, and MLP head; trained on CIFAR-10 achieving \~75% test accuracy vs \~90% for CNN baselines  

Innovations:  
The comparison with ResNet in TensorBoard makes the inductive bias gap concrete: the ResNet matches the ViT's best validation accuracy after 5 epochs while the ViT needs 50k+ iterations to achieve the same. The notebook explains this through the absence of translation invariance and local connectivity priors in the ViT, which must be learned from scratch from classification labels alone — a bottleneck that disappears with large-scale pre-training.
