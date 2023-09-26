# Towards a Unified Analysis of Kernel-based Methods Under Covariate Shift (NIPS 2023)
A Python code for "Towards a Unified Analysis of Kernel-based Methods Under Covariate Shift".



## File Description
KLIEP_importance_estimation.py is devoted to accomplishing the  KLIEP algorithm for estimating the importance ratio.

United_function_tools.py contains the function tools used in our experiment, including four parts: KRR estimation, KQR estimation, KLR estimation, and KSVM estimation.

Kernel_covariate_shift_experiments.ipynb is our main code that performs the experiments. In addition to the KRR model for the 1-dimensional bounded case that we put in the first, we only present the code for varying the regularization parameter $\lambda$ for brevity. For various combinations of $\tau$ and $r$ for the KQR model, we only present the case of $\tau=0.5$ and $r=1$. For the real data studies, we only present the code for performing on the Raisin dataset. As for the multi-source datasets, we randomly choose one covariate to exist shift and the splitting procedure is the same as the Raisin dataset.

