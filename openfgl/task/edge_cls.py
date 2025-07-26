import torch
import torch.nn as nn
from torch_geometric.loader import LinkNeighborLoader
import os
from os import path as osp
import pickle
import numpy as np
from typing import Tuple

from openfgl.task.base import BaseTask
from openfgl.utils.basic_utils import extract_floats, idx_to_mask_tensor, mask_tensor_to_idx
from openfgl.utils.task_utils import load_edge_attributed_default_model
from openfgl.utils.wanb import wandb_run
from openfgl.utils.metrics import compute_supervised_metrics
from openfgl.utils.privacy_utils import clip_gradients, add_noise
from openfgl.data.processing import processing


class EdgeClsTask(BaseTask):
    """
    Task class for edge classification in a federated learning setup.

    Attributes:
        client_id (int): ID of the client.
        data_dir (str): Directory containing the data.
        args (Namespace): Arguments containing model and training configurations.
        device (torch.device): Device to run the computations on.
        data (object): Data specific to the task.
        model (torch.nn.Module): Model to be trained.
        optim (torch.optim.Optimizer): Optimizer for the model.
        train_mask (torch.Tensor): Mask for the training set.
        val_mask (torch.Tensor): Mask for the validation set.
        test_mask (torch.Tensor): Mask for the test set.
        splitted_data (dict): Dictionary containing split data and DataLoaders.
        processed_data (object): Processed data for training.
    """
    def __init__(self, args, client_id, data, data_dir, device):
        super(EdgeClsTask, self).__init__(args, client_id, data, data_dir, device)

    def train(self):
        """
        Train the model on the processed data.
        """
        if self.args.use_batch_loading:
            print("training with batch loading")
            self.train_with_batch_loading()
        else:
            self.train_without_batch_loading()
    
    def train_with_batch_loading(self):
        """
        Train the model using batch loading with LinkNeighborLoader.
        """
        data = self.processed_data["data"]  # Use processed data for training
        train_mask = self.processed_data["train_mask"] 

        self.model.train()

        # Get the indices of training edges
        train_edge_indices = train_mask.nonzero(as_tuple=False).squeeze(1)
        
        # TODO we could use 'edge_label_time' here when we want to inductive learning. 
        # Requires time_attr to be set too. See doc.
        # Also we still need to make sure train_mask is set for indutive learning.
        loader = LinkNeighborLoader(
            data,
            edge_label_index=data.edge_index[:, train_edge_indices],
            edge_label=data.y[train_edge_indices],
            batch_size=self.args.batch_size,
            shuffle=True,
            num_neighbors=self.args.num_neighbors,
            num_workers=0,  # Single-threaded to avoid issues
        )

        for _ in range(self.args.num_epochs):
            
            for batchidx, batch in enumerate(loader):
                if batchidx > self.args.iterations_per_epoch:
                    break

                # Move batch to device
                batch = batch.to(self.device)

                # reset gradients
                self.optim.zero_grad()

                # not all the target edges will be sampled in the batch graph, but we only want those.
                # TODO it is more common to put this in the model, but we do it here so that we can easier use node models.
                mask_logits = torch.isin(batch.e_id, train_edge_indices[batch.input_id])
                mask_labels = torch.isin(train_edge_indices[batch.input_id], batch.e_id)
                assert mask_logits.sum() == mask_labels.sum(), "mask_logits sum should match mask_labels sum"

                # if more than 10% of the edges are not in the mask, raise an error
                if mask_logits.sum() < 0.9 * batch.input_id.shape[0]:
                    raise ValueError("More than 10% of the edges are not in the mask. This should not happen, as input_id should be the edges in the batch.")

                # node_embeddings and edge_logits for full batch graph
                node_embeddings, edge_logits = self.model.forward(batch)

                # logits here are the logits for all edges in the graph, so we need to filter them using the mask
                labeled_logits = edge_logits[mask_logits]  # Get logits for the edges in the batch

                # TODO use self.loss_fn instead of default_loss_fn
                loss_train = self.default_loss_fn(labeled_logits, batch.edge_label[mask_labels])

                if self.args.dp_mech != "no_dp":
                    # clip the gradient of each sample in this batch
                    clip_gradients(self.model, loss_train, loss_train.shape[0], self.args.dp_mech, self.args.grad_clip)
                else:
                    loss_train.backward()

                if self.step_preprocess is not None:
                    self.step_preprocess()

                self.optim.step()

                print(f"[client {self.client_id}] Epoch: {_+1}/{self.args.num_epochs}, Batch: {batchidx+1}/{len(loader)}, Loss: {loss_train.item():.4f}")

                if self.args.dp_mech != "no_dp":
                    # add noise to parameters
                    add_noise(self.args, self.model, loss_train.shape[0])

    def train_without_batch_loading(self):
        """
        Train the model on the processed data.
        """
        splitted_data = self.processed_data # use processed_data to train model

        self.model.train()
        for _ in range(self.args.num_epochs):
            self.optim.zero_grad()

            # the node_embeddings are in fact node logits
            node_embeddings, edge_logits = self.model.forward(splitted_data["data"]) 

            # logits here are the logits for all edges in the graph, so we need to filter them using the mask

            loss_train = self.loss_fn(node_embeddings, edge_logits, splitted_data["data"].y, splitted_data["train_mask"])
            if self.args.dp_mech != "no_dp":
                # clip the gradient of each sample in this batch
                clip_gradients(self.model, loss_train, loss_train.shape[0], self.args.dp_mech, self.args.grad_clip)
            else:
                loss_train.backward()

            if self.step_preprocess is not None:
                # TODO we need to schedule learning rate
                self.step_preprocess()

            self.optim.step()
            if self.args.dp_mech != "no_dp":
                # add noise to parameters
                add_noise(self.args, self.model, loss_train.shape[0])

    def evaluate(self, mute=False):
        """
        Evaluate the model on all splits (train, validation, test).

        Returns:
            dict: Evaluation metrics 
        """
        if self.args.use_batch_loading:
            evals = self.evaluate_with_batch_loading(mute)
        else:
            evals = self.evaluate_without_batch_loading(mute)

        # log to wandb
        # TODO need a way to add the round 
        wandb_run.log( evals )
        return evals
    
    def evaluate_with_batch_loading(self, mute=False):
        """
        Evaluate the model using batch loading with LinkNeighborLoader.

        Returns:
            dict: Evaluation metrics 
        """
        splitted_data = self.processed_data

        if self.override_evaluate is not None:
            return self.override_evaluate(splitted_data)
    
        eval_output = {}
        self.model.eval()
        with torch.no_grad():
            split_examples = 0
            for split in ["train", "val", "test"]:
                split_mask = splitted_data[f"{split}_mask"]
                data = splitted_data["data"]
                split_edge_indices = split_mask.nonzero(as_tuple=False).squeeze(1)


                eval_output[f"loss_{split}"] = 0


                loader = LinkNeighborLoader(
                    data,
                    edge_label_index=data.edge_index[:, split_edge_indices],
                    edge_label=data.y[split_edge_indices],
                    batch_size=self.args.batch_size,
                    shuffle=True,
                    num_neighbors=self.args.num_neighbors,
                    num_workers=0,  # Single-threaded to avoid issues
                )
                for batchidx, batch in enumerate(loader):
                    if batchidx > self.args.iterations_per_epoch:
                        break
                    # Move batch to device
                    batch = batch.to(self.device)

                    # node_embeddings and edge_logits for full batch graph
                    node_embeddings, edge_logits = self.model.forward(batch)

                    # logits here are the logits for all edges in the graph, so we need to filter them using the mask
                    mask_logits = torch.isin(batch.e_id, split_edge_indices[batch.input_id])
                    mask_labels = torch.isin(split_edge_indices[batch.input_id], batch.e_id)
                    labeled_logits = edge_logits[mask_logits]
                    labeled_labels = batch.edge_label[mask_labels]

                    loss = self.default_loss_fn(labeled_logits, labeled_labels)

                    eval_output[f"loss_{split}"] += loss.item()

                    # update the metrics proportional to number of samples
                    num_samples = labeled_logits.shape[0]
                    split_examples += num_samples

                    metrics = compute_supervised_metrics(
                        metrics=self.args.metrics,
                        logits=labeled_logits,
                        labels=labeled_labels,
                        suffix=split,
                    )

                    for key, value in metrics.items():
                        if key in eval_output:
                            eval_output[key] += value * num_samples
                        else:
                            eval_output[key] = value * num_samples

                # eval_output[f"loss_{split}"] /= len(loader)
                for key, value in eval_output.items():
                    if key.startswith("loss_"):
                        continue
                    eval_output[key] /= split_examples

        info = ""
        for key, val in eval_output.items():
            try:
                info += f"\t{key}: {val:.4f}"
            except:
                continue

        prefix = f"[client_{self.client_id}]" if self.client_id is not None else "[server]"
        if not mute:
            print(prefix+info)
        return eval_output

    def evaluate_without_batch_loading(self, mute=False):
        """
        Evaluate the model on all splits (train, validation, test).

        Returns:
            dict: Evaluation metrics 
        """
        splitted_data = self.processed_data

        if self.override_evaluate is not None:
            return self.override_evaluate(splitted_data)

        eval_output = {}
        self.model.eval()
        with torch.no_grad():
            node_embedding, logits = self.model.forward(splitted_data["data"])
            loss_train = self.loss_fn(node_embedding, logits, splitted_data["data"].y, splitted_data["train_mask"])
            loss_val = self.loss_fn(node_embedding, logits, splitted_data["data"].y, splitted_data["val_mask"])
            loss_test = self.loss_fn(node_embedding, logits, splitted_data["data"].y, splitted_data["test_mask"])

            
        eval_output["embedding"] = node_embedding
        eval_output["logits"] = logits
        eval_output["loss_train"] = loss_train
        eval_output["loss_val"]   = loss_val
        eval_output["loss_test"]  = loss_test
            
            
        metric_train = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["train_mask"]], labels=splitted_data["data"].y[splitted_data["train_mask"]], suffix="train")
        metric_val = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["val_mask"]], labels=splitted_data["data"].y[splitted_data["val_mask"]], suffix="val")
        metric_test = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["test_mask"]], labels=splitted_data["data"].y[splitted_data["test_mask"]], suffix="test")
        eval_output = {**eval_output, **metric_train, **metric_val, **metric_test}
            
        info = ""
        for key, val in eval_output.items():
            try:
                info += f"\t{key}: {val:.4f}"
            except:
                continue

        prefix = f"[client {self.client_id}]" if self.client_id is not None else "[server]"
        if not mute:
            print(prefix+info)
        return eval_output

    def loss_fn(self, embedding, logits, label, mask):
        """
        Calculate the loss for the model.

        Args:
            embedding (torch.Tensor): Embeddings from the model.
            logits (torch.Tensor): Logits from the model.
            label (torch.Tensor): Ground truth labels.
            mask (torch.Tensor): Mask to filter the logits and labels.

        Returns:
            torch.Tensor: Calculated loss.
        """
        return self.default_loss_fn(logits[mask], label[mask])

    @property
    def default_model(self):
        """
        Get the default model for node and edge level tasks.

        Returns:
            torch.nn.Module: Default model.
        """
        model = load_edge_attributed_default_model(
            self.args,
            input_dim_node=self.num_feats[0],
            input_dim_edge=self.num_feats[1],
            output_dim=self.num_global_classes,
            client_id=self.client_id
        )
        print(f"loaded model: {model}")
        return model

    @property
    def default_optim(self):
        """
        Get the default optimizer for the task.

        Returns:
            torch.optim.Optimizer: Default optimizer.
        """
        if self.args.optim == "adam":
            from torch.optim import Adam
            return Adam

    @property
    def num_samples(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.data.y.shape[0]

    @property
    def num_feats(self) -> Tuple[int, int]:
        """
        Get the number of node and edge features in the dataset.

        Returns:
            int: Number of features.
        """
        print(self.data.keys())
        return self.data.x.shape[1], self.data.edge_attr.shape[1]

    @property
    def num_global_classes(self):
        """
        Get the number of global classes in the dataset.

        Returns:
            int: Number of global classes.
        """
        return self.data.num_global_classes

    @property
    def default_loss_fn(self):
        """
        Get the default loss function for the task.

        Returns:
            function: Default loss function.
        """
        if self.args.dp_mech != "no_dp":
            return nn.CrossEntropyLoss(reduction="none")
        else:
            # TODO parametrize this
            # return nn.BCEWithLogitsLoss(weight=torch.tensor([1,7], device=self.device))
            return nn.CrossEntropyLoss(weight=torch.tensor([1, 7], device=self.device, dtype=torch.float32))

    @property
    def default_train_val_test_split(self):
        """
        Get the default train/validation/test split based on the dataset.

        Returns:
            tuple: Default train/validation/test split ratios.
        """
        if self.client_id is None:
            return None

        if len(self.args.dataset) > 1:
            name = self.args.dataset[self.client_id]
        else:
            name = self.args.dataset[0]

        if 'AML' in name:
            return 0.6, 0.2, 0.2
        else:
            return 0.8, 0.1, 0.1


    @property
    def train_val_test_path(self):
        """
        Get the path to the train/validation/test split file.

        Returns:
            str: Path to the split file.
        """

        if self.args.train_val_test == "default_split":
            return osp.join(self.data_dir, f"edge_cls", "default_split")
        else:
            split_dir = f"split_{self.args.train_val_test}"
            return osp.join(self.data_dir, f"edge_cls", split_dir)


    def load_train_val_test_split(self):
        """
        Load the train/validation/test split from a file.
        """
        # here we do the split
        # if no file, we call the local_subgraph_train_val_test_split method
        # see node_cls

        if self.client_id is None and len(self.args.dataset) == 1: # server
            glb_train = []
            glb_val = []
            glb_test = []
            
            for client_id in range(self.args.num_clients):
                glb_train_path = osp.join(self.train_val_test_path, f"glb_train_{client_id}.pkl")
                glb_val_path = osp.join(self.train_val_test_path, f"glb_val_{client_id}.pkl")
                glb_test_path = osp.join(self.train_val_test_path, f"glb_test_{client_id}.pkl")
                
                with open(glb_train_path, 'rb') as file:
                    glb_train_data = pickle.load(file)
                    glb_train += glb_train_data
                    
                with open(glb_val_path, 'rb') as file:
                    glb_val_data = pickle.load(file)
                    glb_val += glb_val_data
                    
                with open(glb_test_path, 'rb') as file:
                    glb_test_data = pickle.load(file)
                    glb_test += glb_test_data
                
            train_mask = idx_to_mask_tensor(glb_train, self.num_samples).bool()
            val_mask = idx_to_mask_tensor(glb_val, self.num_samples).bool()
            test_mask = idx_to_mask_tensor(glb_test, self.num_samples).bool()
            
        else: # client
            train_path = osp.join(self.train_val_test_path, f"train_{self.client_id}.pt")
            val_path = osp.join(self.train_val_test_path, f"val_{self.client_id}.pt")
            test_path = osp.join(self.train_val_test_path, f"test_{self.client_id}.pt")
            glb_train_path = osp.join(self.train_val_test_path, f"glb_train_{self.client_id}.pkl")
            glb_val_path = osp.join(self.train_val_test_path, f"glb_val_{self.client_id}.pkl")
            glb_test_path = osp.join(self.train_val_test_path, f"glb_test_{self.client_id}.pkl")
            
            # if the file exists, we load it
            if osp.exists(train_path) and osp.exists(val_path) and osp.exists(test_path)\
                and osp.exists(glb_train_path) and osp.exists(glb_val_path) and osp.exists(glb_test_path): 
                train_mask = torch.load(train_path)
                val_mask = torch.load(val_path)
                test_mask = torch.load(test_path)
            # otherwise, we do the split
            else:
                train_mask, val_mask, test_mask = self.local_subgraph_train_val_test_split(self.data, self.args.train_val_test)
                
                if not osp.exists(self.train_val_test_path):
                    os.makedirs(self.train_val_test_path)
                    
                torch.save(train_mask, train_path)
                torch.save(val_mask, val_path)
                torch.save(test_mask, test_path)
                
                # map the splits id (here indices of the edges) to index in the global edge list
                if len(self.args.dataset) == 1:
                    # map edges to global
                    glb_train_id = []
                    glb_val_id = []
                    glb_test_id = []
                    for id_train in train_mask.nonzero():
                        glb_train_id.append(
                            self.data.global_edge_map[id_train.item()]
                        )
                    for id_val in val_mask.nonzero():
                        glb_val_id.append(self.data.global_edge_map[id_val.item()])
                    for id_test in test_mask.nonzero():
                        glb_test_id.append(self.data.global_edge_map[id_test.item()])

                    with open(glb_train_path, 'wb') as file:
                        pickle.dump(glb_train_id, file)
                    with open(glb_val_path, 'wb') as file:
                        pickle.dump(glb_val_id, file)
                    with open(glb_test_path, 'wb') as file:
                        pickle.dump(glb_test_id, file)

        self.train_mask = train_mask.to(self.device)
        self.val_mask = val_mask.to(self.device)
        self.test_mask = test_mask.to(self.device)
        

        self.splitted_data = {
            "data": self.data,
            "train_mask": self.train_mask,
            "val_mask": self.val_mask,
            "test_mask": self.test_mask
        }
        
        # processing
        self.processed_data = processing(args=self.args, splitted_data=self.splitted_data, processed_dir=self.data_dir, client_id=self.client_id)


    def local_subgraph_train_val_test_split(self, local_subgraph, split: str, shuffle=True):
        """
        Split the local subgraph into train, validation, and test sets based on the edges.

        Args:
            local_subgraph (object): Local subgraph to be split.
            split (str or tuple): Split ratios or default split identifier.
            shuffle (bool, optional): If True, shuffle the subgraph before splitting. Defaults to True.

        Returns:
            tuple: Masks for the train, validation, and test sets.
        """
        num_edges = local_subgraph.y.shape[0]
        if split == "default_split":
            train_, val_, test_ = self.default_train_val_test_split
        else:
            if isinstance(split, str):
               train_, val_, test_ = extract_floats(split)
            elif isinstance(split, iterable) and len(split) == 3:
                train_, val_, test_ = split
                assert train_ + val_ + test_ <= 1, "The sum of train, val, and test split should be <= 1"
            else:
                raise ValueError(f"Invalid split: {split}. Expected a string or a tuple of three floats.")
        
        # make zeroes
        train_mask = idx_to_mask_tensor([], num_edges)
        val_mask = idx_to_mask_tensor([], num_edges)
        test_mask = idx_to_mask_tensor([], num_edges)

        # stratified split 
        # TODO this does not make a lot of sense for AML, we need to do a split based on timestamps, instead of transductive
        for class_i in range(local_subgraph.num_global_classes):
            class_i_edge_mask = local_subgraph.y == class_i
            num_class_i_edges = class_i_edge_mask.sum()
            
            class_i_edge_list = mask_tensor_to_idx(class_i_edge_mask)
            if shuffle:
                np.random.shuffle(class_i_edge_list)
            train_mask += idx_to_mask_tensor(class_i_edge_list[:int(train_ * num_class_i_edges)], num_edges)
            val_mask += idx_to_mask_tensor(class_i_edge_list[int(train_ * num_class_i_edges) : int((train_+val_) * num_class_i_edges)], num_edges)
            test_mask += idx_to_mask_tensor(class_i_edge_list[int((train_+val_) * num_class_i_edges): min(num_class_i_edges, int((train_+val_+test_) * num_class_i_edges))], num_edges)
        
        
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

        return train_mask, val_mask, test_mask
    