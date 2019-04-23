#%%

import numpy as np
import pandas as pd
from scipy.fftpack import fftshift
import matplotlib.pyplot as plt
from scipy import stats
from hmmlearn import hmm
from MHMM import hmm2

#%%
import numpy as np
from hmmlearn import hmm
np.random.seed(42)

model = hmm.GaussianHMM(n_components=3, covariance_type="full")
model.startprob_ = np.array([0.6, 0.3, 0.1])
model.transmat_ = np.array([[0.7, 0.2, 0.1],
                             [0.3, 0.5, 0.2],
                             [0.3, 0.3, 0.4]])
model.means_ = np.array([[0.0], [3.0], [5.0]])
model.covars_ = np.tile(np.identity(1), (3, 1, 1))
X, Z = model.sample(100)

#%%
# train
model = hmm2(3)
model.set_emmision_model("gaussian", 1)
print(model.fit(X.reshape(1,-1)))