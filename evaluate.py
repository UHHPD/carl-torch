import os
<<<<<<< HEAD
=======
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
>>>>>>> d4a15a9 (Initial commit for CARL-TORCH)
import sys
import logging
import numpy as np
from arg_handler import arg_handler_eval
from ml import RatioEstimator
from ml.utils.loading import Loader

#################################################
opts = arg_handler_eval()
nominal  = opts.nominal
variation = opts.variation
n = opts.nentries
p = opts.datapath
global_name = opts.global_name
features = opts.features.split(",")
weightFeature = opts.weightFeature
treename = opts.treename
model = opts.model
binning = opts.binning
normalise = opts.normalise
raw_weight = opts.raw_weight
scale_method = opts.scale_method
<<<<<<< HEAD
=======
batch_size = opts.batch_size
>>>>>>> d4a15a9 (Initial commit for CARL-TORCH)
#################################################


logger = logging.getLogger(__name__)
if os.path.exists('data/'+global_name+'/X_train_'+str(n)+'.npy') and os.path.exists('data/'+global_name+'/metaData_'+str(n)+'.pkl'):
    logger.info(" Doing evaluation of model trained with datasets: [{}, {}], with {} events.".format(nominal, variation, n))
else:
    logger.info(" No data set directory of the form {}.".format('data/'+global_name+'/X_train_'+str(n)+'.npy'))
    logger.info(" No datasets available for evaluation of model trained with datasets: [{},{}] with {} events.".format(nominal, variation, n))
    logger.info("ABORTING")
    sys.exit()

loading = Loader()
carl = RatioEstimator()
carl.scaling_method = scale_method
if model:
    carl.load(model, global_name=global_name, nentries=n)
else:
    carl.load('models/'+global_name+'_carl_'+str(n), global_name=global_name, nentries=n)
evaluate = ['train','val']
raw_w = "raw_" if raw_weight else ""
for i in evaluate:
    logger.info("Running evaluation for {}".format(i))
<<<<<<< HEAD
    r_hat, s_hat = carl.evaluate(x='data/'+global_name+'/X0_'+i+'_'+str(n)+'.npy')
    logger.info("s_hat = {}".format(s_hat))
    logger.info("r_hat = {}".format(r_hat))
=======
    r_hat, s_hat = carl.evaluate(x='data/'+global_name+'/X0_'+i+'_'+str(n)+'.npy', batch_size = batch_size)
    logger.info("s_hat for Default = {}".format(s_hat))
    logger.info("r_hat for Default = {}".format(r_hat))
>>>>>>> d4a15a9 (Initial commit for CARL-TORCH)
    w = 1./r_hat   # I thought r_hat = p_{1}(x) / p_{0}(x) ???
    # Correct nan's and inf's to 1.0 corrective weights as they are useless in this instance. Warning
    # to screen should already be printed
    if opts.weight_protection:
        w = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)
<<<<<<< HEAD
    
=======
>>>>>>> d4a15a9 (Initial commit for CARL-TORCH)
    # Weight clipping if requested by user
    if opts.weight_threshold < 100:
        carl_w_clipping = np.percentile(w, opts.weight_threshold)
        w[w > carl_w_clipping] = carl_w_clipping

<<<<<<< HEAD
    print("w = {}".format(w))
=======
    # Save s_hat, r_hat, and w as numpy arrays
    np.save(f"data/{global_name}/w_CARL_{i}_{n}.npy", w) # These are the CARL weights!

    print("w = {}".format(w))
    
>>>>>>> d4a15a9 (Initial commit for CARL-TORCH)
    print("<evaluate.py::__init__>::   Loading Result for {}".format(i))
    loading.load_result(
        x0=f'data/{global_name}/X0_{i}_{n}.npy',
        x1=f'data/{global_name}/X1_{i}_{n}.npy',
        w0=f'data/{global_name}/w0_{i}_{raw_w}{n}.npy',
        w1=f'data/{global_name}/w1_{i}_{raw_w}{n}.npy',
        metaData=f'data/{global_name}/metaData_{n}.pkl',
        weights=w,
        features=features,
        #weightFeature=weightFeature,
        label=i,
        plot=True,
        nentries=n,
        #TreeName=treename,
        #pathA=p+nominal+".root",
        #pathB=p+variation+".root",
        global_name=global_name,
        plot_ROC=opts.plot_ROC,
        plot_obs_ROC=opts.plot_obs_ROC,
        ext_binning = binning,
        normalise = normalise,
        scaling=scale_method,
        plot_resampledRatio=opts.plot_resampledRatio,
    )
# Evaluate performance
print("<evaluate.py::__init__>::   Evaluate Performance of Model")
carl.evaluate_performance(x='data/'+global_name+'/X_val_'+str(n)+'.npy',
<<<<<<< HEAD
                          y='data/'+global_name+'/y_val_'+str(n)+'.npy')
=======
                          y='data/'+global_name+'/y_val_'+str(n)+'.npy',
                          batch_size = batch_size)
>>>>>>> d4a15a9 (Initial commit for CARL-TORCH)
