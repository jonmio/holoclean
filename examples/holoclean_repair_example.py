import sys
sys.path.append('../')
import holoclean
from detect import NullDetector, ViolationDetector
from repair.featurize import *
import random
import numpy as np

seeds = [random.randint(0, 10000) for _ in range(30)]
# weight_decays = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
# learning_rates = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

# [10E-3, 10E3][10E-7, 10E-2]


# weight_decays = [0.00000001, 0.01] linspace 10
# learning_rates = [0.001, 500] linspace 15
#
weight_decays = np.logspace(-5, 0, 10)
learning_rates = np.logspace(-6, 0, 10)


for seed in seeds:
    for weight_decay in weight_decays:
        for learning_rate in learning_rates:
            # 1. Setup a HoloClean session.
            hc = holoclean.HoloClean(
                db_name='holo_jon',
                domain_thresh_1=0,
                domain_thresh_2=0,
                weak_label_thresh=0.99,
                max_domain=10000,
                cor_strength=0.6,
                nb_cor_strength=0.8,
                epochs=10,
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                threads=1,
                batch_size=1,
                verbose=False,
                timeout=3*60000,
                feature_norm=False,
                weight_norm=False,
                print_fw=True,
                seed=seed
            ).session

            # 2. Load training data and denial constraints.
            hc.load_data('food', '~/food5k/food5k-transformed.csv')
            hc.load_dcs('~/food5k/food5k_constraints.txt')
            hc.ds.set_constraints(hc.get_dcs())

            # 3. Detect erroneous cells using these two detectors.
            detectors = [NullDetector(), ViolationDetector()]
            hc.detect_errors(detectors)

            # 4. Repair errors utilizing the defined features.
            hc.setup_domain()
            featurizers = [
                InitAttrFeaturizer(),
                OccurAttrFeaturizer(),
                FreqFeaturizer(),
                ConstraintFeaturizer(),
            ]

            hc.repair_errors(featurizers)

            # 5. Evaluate the correctness of the results.
            hc.evaluate(fpath='~/food5k/food5k-transformed_clean_flat.csv',
                        tid_col='tid',
                        attr_col='attribute',
                        val_col='correct_val')
