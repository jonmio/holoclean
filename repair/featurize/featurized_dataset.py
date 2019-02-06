from collections import namedtuple
import logging

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dataset import AuxTables, CellStatus

FeatInfo = namedtuple('FeatInfo', ['name', 'size', 'learnable', 'init_weight', 'feature_names'])
Example = namedtuple('Example', ['X', 'Y', 'var_mask'])


class FeaturizedDataset:
    def __init__(self, dataset, env, featurizers):
        self.ds = dataset
        self.env = env
        self.total_vars, self.classes = self.ds.get_domain_info()
        self.processes = self.env['threads']
        for f in featurizers:
            f.setup_featurizer(self.ds, self.processes, self.env['batch_size'])
        self.featurizers = featurizers
        self.featurizer_info = [FeatInfo(featurizer.name,
                                         featurizer.num_features(),
                                         featurizer.learnable,
                                         featurizer.init_weight,
                                         featurizer.feature_names())
                                for featurizer in self.featurizers]

        # TODO: remove after we validate it is not needed.
        self.in_features = sum(featurizer.size for featurizer in self.featurizer_info)
        logging.debug("generating weak labels...")
        self.weak_labels, self.is_clean = self.generate_weak_labels()
        logging.debug("DONE generating weak labels.")
        logging.debug("generating mask...")
        self.var_class_mask, self.var_to_domsize = self.generate_var_mask()
        logging.debug("DONE generating mask.")

    def generate_weak_labels(self):
        """
        generate_weak_labels returns a tensor where for each VID we have the
        domain index of the initial value.

        :return: Torch.Tensor of size (# of variables) X 1 where tensor[i][0]
            contains the domain index of the initial value for the i-th
            variable/VID.
        """
        # Trains with clean cells AND cells that have been weak labelled.
        query = 'SELECT _vid_, weak_label_idx, fixed, (t2._cid_ IS NULL) AS clean ' \
                'FROM {} AS t1 LEFT JOIN {} AS t2 ON t1._cid_ = t2._cid_ ' \
                'WHERE t2._cid_ is NULL ' \
                '   OR t1.fixed != {};'.format(AuxTables.cell_domain.name,
                                               AuxTables.dk_cells.name,
                                               CellStatus.NOT_SET.value)
        res = self.ds.engine.execute_query(query)
        if len(res) == 0:
            raise Exception("No weak labels available. Reduce pruning threshold.")
        labels = -1 * torch.ones(self.total_vars, 1).type(torch.LongTensor)
        is_clean = torch.zeros(self.total_vars, 1).type(torch.LongTensor)
        for tuple in tqdm(res):
            vid = int(tuple[0])
            label = int(tuple[1])
            fixed = int(tuple[2])
            clean = int(tuple[3])
            labels[vid] = label
            is_clean[vid] = clean
        return labels, is_clean

    def generate_var_mask(self):
        """
        generate_var_mask returns a mask tensor where invalid domain indexes
        for a given variable/VID has value -10e6.

        An invalid domain index is possible since domain indexes are expanded
        to the maximum domain size of a given VID: e.g. if a variable A has
        10 unique values and variable B has 6 unique values, then the last
        4 domain indexes (index 6-9) of variable B are invalid.

        :return: Torch.Tensor of size (# of variables) X (max domain)
            where tensor[i][j] = 0 iff the value corresponding to domain index 'j'
            is valid for the i-th VID and tensor[i][j] = -10e6 otherwise.
        """
        var_to_domsize = {}
        query = 'SELECT _vid_, domain_size FROM %s' % AuxTables.cell_domain.name
        res = self.ds.engine.execute_query(query)
        mask = torch.zeros(self.total_vars,self.classes)
        for tuple in tqdm(res):
            vid = int(tuple[0])
            max_class = int(tuple[1])
            mask[vid, max_class:] = -10e6
            var_to_domsize[vid] = max_class
        return mask, var_to_domsize

    def get_training_dataset(self):
        """
        Constructs a TorchFeaturizedDataset for the training dataset.

        :return: An instance of TorchFeaturizedDataset for the training data.
        """
        train_idx = (self.weak_labels != -1).nonzero()[:,0]
        return TorchFeaturizedDataset(
            vids=train_idx,
            featurizers=self.featurizers,
            Y=self.weak_labels,
            var_mask=self.var_class_mask,
            batch_size=self.env['batch_size'],
            feature_norm =self.env['feature_norm']
        )

    def get_infer_dataset(self):
        """
        Constructs a TorchFeaturizedDataset for the inference dataset.

        :return: An instance of TorchFeaturizedDataset for the inference data.
        """
        infer_idx = (self.is_clean == 0).nonzero()[:, 0]
        return TorchFeaturizedDataset(
            vids=infer_idx,
            featurizers=self.featurizers,
            Y=self.weak_labels,
            var_mask=self.var_class_mask,
            batch_size=self.env['batch_size'],
            feature_norm =self.env['feature_norm']
        ), infer_idx


"""
Implements interface for torch.utils.data.Dataset.

An instance of TorchFeaturizedDataset can be created for training and inference
data. It can be used with Torch DataLoader to iterate over batches of the data.
"""
class TorchFeaturizedDataset(torch.utils.data.Dataset):

    def __init__(self, vids, featurizers, Y, var_mask, batch_size, feature_norm):
        """
        Initializes a TorchFeaturizedDataset.

        :param vids: List[str] with vids of the cells to include in the dataset.
        :param featurizers: List[Featurizer] used to featurize the data.
        :param Y: Torch.tensor(num_cells, 1) where tensor[i][0] contains the domain
            index of the initial value for the cell with vid i.
        :param var_mask: Torch.tensor(num_cells, max_domain) where tensor[i][j] = 0
            iff the value corresponding to domain index j is valid for the cell with
            vid i, otherwise tensor[i][j] = -10e6.
        :param batch_size: An int storing the number of examples to include per batch.
        :param feature_norm: A boolean indicating if features should be normalized.
        """
        self.vids = vids
        self.featurizers = featurizers
        self.Y = Y
        self.var_mask = var_mask
        self.batch_size = batch_size
        self.feature_norm = feature_norm
        self.num_examples = len(self.vids)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        """
        Get the featurized tensor for a cell.

        :param idx: An int storing the index of the cell to be featurized in
            self.vids.
        :return: A torch.Tensor(max_domain, num_features) holding the featurized
            values of the cell.
        """
        X = torch.cat([featurizer.gen_feat_tensor(self.vids[idx]) for featurizer in self.featurizers], dim=1)
        if self.feature_norm:
            # normalize within each cell the features
            X = F.normalize(X, p=2, dim=0)

        Y = self.Y[self.vids[idx]]
        var_mask = self.var_mask[self.vids[idx]]
        return Example(X, Y, var_mask)
