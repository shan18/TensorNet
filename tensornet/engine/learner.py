import math
import time
import torch
from copy import deepcopy

from tensornet.engine.ops.regularizer import l1
from tensornet.engine.ops.checkpoint import ModelCheckpoint
from tensornet.engine.ops.tensorboard import TensorBoard
from tensornet.data.processing import InfiniteDataLoader
from tensornet.utils.progress_bar import ProgressBar


class Learner:
    """Model Trainer and Validator.

    Args:
        train_loader (torch.utils.data.DataLoader): Training data loader.
        optimizer (torch.optim): Optimizer for the model.
        criterion (torch.nn): Loss Function.
        device (:obj:`str` or :obj:`torch.device`, optional): Device where the data
            will be loaded. (default='cpu')
        epochs (:obj:`int`, optional): Numbers of epochs/iterations to train the model for.
            (default: 1)
        l1_factor (:obj:`float`, optional): L1 regularization factor. (default: 0)
        val_loader (:obj:`torch.utils.data.DataLoader`, optional): Validation data loader.
        callbacks (:obj:`list`, optional): List of callbacks to be used during training.
        metrics (:obj:`list`, optional): List of names of the metrics for model
            evaluation.

            *Note*: If the model has multiple outputs, then this will be a nested list
            where each individual sub-list will specify the metrics which are to be used for
            evaluating each output respectively. In such cases, the model checkpoint will
            consider only the metric of the first output for saving checkpoints.
        activate_loss_logits (:obj:`bool`, optional): If True, the logits will first pass
            through the `activate_logits` function before going to the criterion.
            (default: False)
        record_train (:obj:`bool`, optional): If False, metrics will be calculated only
            during validation. (default: True)
    """

    def __init__(
        self, train_loader, optimizer, criterion, device='cpu',
        epochs=1, l1_factor=0.0, val_loader=None, callbacks=None, metrics=None,
        activate_loss_logits=False, record_train=True
    ):
        self.model = None
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.device = device
        self.epochs = epochs
        self.val_loader = val_loader
        self.l1_factor = l1_factor
        self.activate_loss_logits = activate_loss_logits
        self.record_train = record_train

        self.lr_schedulers = {
            'step_lr': None,
            'lr_plateau': None,
            'one_cycle_policy': None,
            'cyclic_lr': None,
        }
        self.checkpoint = None
        self.summary_writer = None
        if callbacks is not None:
            self._setup_callbacks(callbacks)

        # Training
        self.train_losses = []  # Change in loss
        self.train_metrics = []  # Change in evaluation metric

        self.val_losses = []  # Change in loss
        self.val_metrics = []  # Change in evaluation metric

        # Set evaluation metrics
        self.metrics = []
        if metrics:
            self._setup_metrics(metrics)

    def _setup_callbacks(self, callbacks):
        """Extract callbacks passed to the class.

        Args:
            callbacks (list): List of callbacks.
        """
        for callback in callbacks:
            if isinstance(callback, torch.optim.lr_scheduler.StepLR):
                self.lr_schedulers['step_lr'] = callback
            elif isinstance(callback, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_schedulers['lr_plateau'] = callback
            elif isinstance(callback, torch.optim.lr_scheduler.OneCycleLR):
                self.lr_schedulers['one_cycle_policy'] = callback
            elif isinstance(callback, ModelCheckpoint):
                if callback.monitor.startswith('train_'):
                    if self.record_train:
                        self.checkpoint = callback
                    else:
                        raise ValueError(
                            'Cannot use checkpoint for a training metric if record_train is set to False'
                        )
                else:
                    self.checkpoint = callback
            elif isinstance(callback, TensorBoard):
                self.summary_writer = callback
            elif isinstance(callback, torch.optim.lr_scheduler.CyclicLR):
                self.lr_schedulers['cyclic_lr'] = callback

    def set_model(self, model):
        """Assign model to learner.

        Args:
            model (torch.nn.Module): Model Instance.
        """
        self.model = model
        if self.summary_writer is not None:
            self.summary_writer.write_model(self.model)

    def _accuracy(self, label, prediction, idx=0):
        """Calculate accuracy.

        Args:
            label (torch.Tensor): Ground truth.
            prediction (torch.Tensor): Prediction.
        """
        self.metrics[idx]['accuracy']['sum'] += prediction.eq(
            label.view_as(prediction)
        ).sum().item()
        self.metrics[idx]['accuracy']['num_steps'] += len(label)
        self.metrics[idx]['accuracy']['value'] = round(
            100 * self.metrics[idx]['accuracy']['sum'] / self.metrics[idx]['accuracy']['num_steps'], 2
        )

    def _iou(self, label, prediction, idx=0):
        """Calculate Intersection over Union.

        Args:
            label (torch.Tensor): Ground truth.
            prediction (torch.Tensor): Prediction.
        """
        # Remove 1 channel dimension
        label = label.squeeze(1)
        prediction = prediction.squeeze(1)

        intersection = (prediction * label).sum(2).sum(1)
        union = (prediction + label).sum(2).sum(1) - intersection

        # epsilon is added to avoid 0/0
        epsilon = 1e-6
        iou = (intersection + epsilon) / (union + epsilon)

        self.metrics[idx]['iou']['sum'] += iou.sum().item()
        self.metrics[idx]['iou']['num_steps'] += label.size(0)
        self.metrics[idx]['iou']['value'] = round(
            self.metrics[idx]['iou']['sum'] / self.metrics[idx]['iou']['num_steps'], 3
        )

    def _pred_label_diff(self, label, prediction, rel=False):
        """Calculate the difference between label and prediction.

        Args:
            label (torch.Tensor): Ground truth.
            prediction (torch.Tensor): Prediction.
            rel (:obj:`bool`, optional): If True, return the relative
                difference. (default: False)

        Returns:
            Difference between label and prediction
        """
        # For numerical stability
        valid_labels = label > 0.0001
        _label = label[valid_labels]
        _prediction = prediction[valid_labels]
        valid_element_count = _label.size(0)

        if valid_element_count > 0:
            diff = torch.abs(_label - _prediction)
            if rel:
                diff = torch.div(diff, _label)

            return diff, valid_element_count

    def _rmse(self, label, prediction, idx=0):
        """Calculate Root Mean Square Error.

        Args:
            label (torch.Tensor): Ground truth.
            prediction (torch.Tensor): Prediction.
        """
        diff = self._pred_label_diff(label, prediction)
        rmse = 0
        if diff is not None:
            rmse = math.sqrt(torch.sum(torch.pow(diff[0], 2)) / diff[1])

        self.metrics[idx]['rmse']['num_steps'] += label.size(0)
        self.metrics[idx]['rmse']['sum'] += rmse * label.size(0)
        self.metrics[idx]['rmse']['value'] = round(
            self.metrics[idx]['rmse']['sum'] / self.metrics[idx]['rmse']['num_steps'], 3
        )

    def _mae(self, label, prediction, idx=0):
        """Calculate Mean Average Error.

        Args:
            label (torch.Tensor): Ground truth.
            prediction (torch.Tensor): Prediction.
        """
        diff = self._pred_label_diff(label, prediction)
        mae = 0
        if diff is not None:
            mae = torch.sum(diff[0]).item() / diff[1]

        self.metrics[idx]['mae']['num_steps'] += label.size(0)
        self.metrics[idx]['mae']['sum'] += mae * label.size(0)
        self.metrics[idx]['mae']['value'] = round(
            self.metrics[idx]['mae']['sum'] / self.metrics[idx]['mae']['num_steps'], 3
        )

    def _abs_rel(self, label, prediction, idx=0):
        """Calculate Absolute Relative Error.

        Args:
            label (torch.Tensor): Ground truth.
            prediction (torch.Tensor): Prediction.
        """
        diff = self._pred_label_diff(label, prediction, rel=True)
        abs_rel = 0
        if diff is not None:
            abs_rel = torch.sum(diff[0]).item() / diff[1]

        self.metrics[idx]['abs_rel']['num_steps'] += label.size(0)
        self.metrics[idx]['abs_rel']['sum'] += abs_rel * label.size(0)
        self.metrics[idx]['abs_rel']['value'] = round(
            self.metrics[idx]['abs_rel']['sum'] / self.metrics[idx]['abs_rel']['num_steps'], 3
        )

    def _setup_metrics(self, metrics):
        """Validate the evaluation metrics passed to the class.

        Args:
            metrics (:obj:`list` or :obj:`dict`): Metrics.
        """

        if not isinstance(metrics[0], (list, tuple)):
            metrics = [metrics]

        for idx, metric_list in enumerate(metrics):
            metric_dict = {}
            for metric in metric_list:
                metric_info = {'value': 0, 'sum': 0, 'num_steps': 0}
                if metric == 'accuracy':
                    metric_info['func'] = self._accuracy
                elif metric == 'rmse':
                    metric_info['func'] = self._rmse
                elif metric == 'mae':
                    metric_info['func'] = self._mae
                elif metric == 'abs_rel':
                    metric_info['func'] = self._abs_rel
                elif metric == 'iou':
                    metric_info['func'] = self._iou

                if 'func' in metric_info:
                    metric_dict[metric] = metric_info

            if metric_dict:
                self.metrics.append(metric_dict)
                self.train_metrics.append({
                    x: [] for x in metric_dict.keys()
                })
                self.val_metrics.append({
                    x: [] for x in metric_dict.keys()
                })

    def _calculate_metrics(self, labels, predictions):
        """Update evaluation metric values.

        Args:
            label (:obj:`torch.Tensor` or :obj:`dict`): Ground truth.
            prediction (:obj:`torch.Tensor` or :obj:`dict`): Prediction.
        """
        predictions = self.activate_logits(predictions)

        if not isinstance(labels, (list, tuple)):
            labels = [labels]
            predictions = [predictions]

        for idx, (label, prediction) in enumerate(zip(labels, predictions)):
            # If predictions are one-hot encoded
            if label.size() != prediction.size():
                prediction = prediction.argmax(dim=1, keepdim=True) * 1.0

            if idx < len(self.metrics):
                for metric in self.metrics[idx]:
                    self.metrics[idx][metric]['func'](
                        label, prediction, idx=idx
                    )

    def _reset_metrics(self):
        """Reset metric params."""
        for idx in range(len(self.metrics)):
            for metric in self.metrics[idx]:
                self.metrics[idx][metric]['value'] = 0
                self.metrics[idx][metric]['sum'] = 0
                self.metrics[idx][metric]['num_steps'] = 0

    def _get_pbar_values(self, loss):
        """Create progress bar description.

        Args:
            loss (float): Loss value.
        """
        pbar_values = [('loss', round(loss, 2))]
        if self.metrics and self.record_train:
            for idx in range(len(self.metrics)):
                for metric, info in self.metrics[idx].items():
                    metric_name = metric
                    if len(self.metrics) > 1:
                        metric_name = f'{idx} - {metric}'
                    pbar_values.append((metric_name, info['value']))
        return pbar_values

    def update_training_history(self, loss):
        """Update the training history.

        Args:
            loss (float): Loss value.
        """
        self.train_losses.append(loss)
        if self.record_train:
            for idx in range(len(self.metrics)):
                for metric in self.metrics[idx]:
                    self.train_metrics[idx][metric].append(
                        self.metrics[idx][metric]['value']
                    )

    def reset_history(self):
        """Reset the training history"""
        self.train_losses = []
        self.val_losses = []
        for idx in range(len(self.metrics)):
            for metric in self.metrics[idx]:
                self.train_metrics[idx][metric] = []
                self.val_metrics[idx][metric] = []
        self._reset_metrics()

    def activate_logits(self, logits):
        """Apply activation function to the logits if needed.
        After this the logits will be sent for calculation of
        loss or evaluation metrics.

        Args:
            logits (torch.Tensor): Model output

        Returns:
            (*torch.Tensor*): activated logits
        """
        return logits

    def calculate_criterion(self, logits, targets, train=True):
        """Calculate loss.

        Args:
            logits (torch.Tensor): Prediction.
            targets (torch.Tensor): Ground truth.
            train (:obj:`bool`, optional): If True, loss is sent to the
                L1 regularization function. (default: True)

        Returns:
            (*torch.Tensor*): loss value
        """
        if self.activate_loss_logits:
            logits = self.activate_logits(logits)
        if train:
            return l1(self.model, self.criterion(logits, targets), self.l1_factor)
        return self.criterion(logits, targets)

    def fetch_data(self, data):
        """Fetch data from loader and load it to GPU.

        Args:
            data (:obj:`tuple` or :obj:`list`): List containing inputs and targets.

        Returns:
            inputs and targets loaded to GPU.
        """
        return data[0].to(self.device), data[1].to(self.device)

    def train_batch(self, data):
        """Train the model on a batch of data.

        Args:
            data (:obj:`tuple` or :obj:`list`): Input and target data for the model.

        Returns:
            (*float*): Batch loss.
        """
        inputs, targets = self.fetch_data(data)
        self.optimizer.zero_grad()  # Set gradients to zero before starting backpropagation
        y_pred = self.model(inputs)  # Predict output
        loss = self.calculate_criterion(y_pred, targets, train=True)  # Calculate loss

        # Perform backpropagation
        loss.backward()
        self.optimizer.step()

        if self.record_train:
            self._calculate_metrics(targets, y_pred)

        # One Cycle Policy for learning rate
        if self.lr_schedulers['one_cycle_policy'] is not None:
            self.lr_schedulers['one_cycle_policy'].step()

        # Cyclic LR policy
        if self.lr_schedulers['cyclic_lr'] is not None:
            self.lr_schedulers['cyclic_lr'].step()

        return loss.item()

    def train_epoch(self, verbose=True):
        """Run an epoch of model training.

        Args:
            verbose (:obj:`bool`, optional): Print logs. (default: True)
        """

        self.model.train()
        if verbose:
            pbar = ProgressBar(target=len(self.train_loader), width=8)

        for batch_idx, data in enumerate(self.train_loader, 0):
            # Train a batch
            loss = self.train_batch(data)

            # Update Progress Bar
            if verbose:
                pbar_values = self._get_pbar_values(loss)
                pbar.update(batch_idx, values=pbar_values)

        # Update training history
        self.update_training_history(loss)
        if verbose:
            pbar_values = self._get_pbar_values(loss)
            pbar.add(1, values=pbar_values)

        self._reset_metrics()

    def train_iterations(self, verbose=True):
        """Train model for the 'self.epochs' number of batches."""

        self.model.train()

        if verbose:
            pbar = ProgressBar(target=self.epochs, width=8)

        iterator = InfiniteDataLoader(self.train_loader)
        for iteration in range(self.epochs):
            # Train a batch
            loss = self.train_batch(iterator.get_batch())

            # Update Progress Bar
            if verbose:
                pbar_values = self._get_pbar_values(loss)
                pbar.update(iteration, values=pbar_values)

            # Update training history
            self.update_training_history(loss)

        if verbose:
            pbar.add(1, values=pbar_values)

    def evaluate(self, loader, verbose=True, log_message='Evaluation'):
        """Evaluate the model on a custom data loader.

        Args:
            loader (torch.utils.data.DataLoader): Data loader.
            verbose (:obj:`bool`, optional): Print loss and metrics. (default: True)
            log_message (str): Prefix for the logs which are printed at the end.

        Returns:
            loss and metric values
        """

        start_time = time.time()
        self.model.eval()
        eval_loss = 0
        with torch.no_grad():
            for data in loader:
                inputs, targets = self.fetch_data(data)
                output = self.model(inputs)  # Get trained model output
                eval_loss += self.calculate_criterion(
                    output, targets, train=False
                ).item()  # Sum up batch loss
                self._calculate_metrics(targets, output)  # Calculate evaluation metrics

        eval_loss /= len(loader.dataset)
        eval_metrics = deepcopy(self.metrics)
        end_time = time.time()

        # Time spent during validation
        duration = int(end_time - start_time)
        minutes = duration // 60
        seconds = duration % 60

        if verbose:
            log = f'{log_message} (took {minutes} minutes, {seconds} seconds): Average loss: {eval_loss:.4f}'
            for idx in range(len(self.metrics)):
                for metric in self.metrics[idx]:
                    log += f', {metric}: {self.metrics[idx][metric]["value"]}'
            log += '\n'
            print(log)

        self._reset_metrics()

        return eval_loss, eval_metrics

    def validate(self, verbose=True):
        """Validate an epoch of model training.

        Args:
            verbose (:obj:`bool`, optional): Print validation loss and metrics.
                (default: True)
        """
        eval_loss, eval_metrics = self.evaluate(
            self.val_loader, verbose=verbose, log_message='Validation set'
        )

        # Update validation logs
        self.val_losses.append(eval_loss)
        for idx in range(len(eval_metrics)):
            for metric in eval_metrics[idx]:
                self.val_metrics[idx][metric].append(
                    eval_metrics[idx][metric]['value']
                )

    def save_checkpoint(self, epoch=None):
        """Save model checkpoint.

        Args:
            epoch (:obj:`int`, optional): Current epoch number.
        """
        if self.checkpoint is not None:
            metric = None
            if self.checkpoint.monitor == 'train_loss':
                metric = self.train_losses[-1]
            elif self.checkpoint.monitor == 'val_loss':
                metric = self.val_losses[-1]
            elif self.metrics:
                if self.checkpoint.monitor.startswith('train_'):
                    if self.record_train:
                        metric = self.train_metrics[0][
                            self.checkpoint.monitor.split('train_')[-1]
                        ][-1]
                else:
                    metric = self.val_metrics[0][
                        self.checkpoint.monitor.split('val_')[-1]
                    ][-1]
            else:
                print('Invalid metric function, can\'t save checkpoint.')
                return

            self.checkpoint(self.model, metric, epoch)

    def write_summary(self, epoch, train):
        """Write training summary in tensorboard.

        Args:
            epoch (int): Current epoch number.
            train (bool): If True, summary will be
                written for model training else it
                will be writtern for model validation.
        """
        if self.summary_writer is not None:
            if train:
                mode = 'train'

                # Write Images
                self.summary_writer.write_images(
                    self.model, self.activate_logits, f'prediction_epoch_{epoch}'
                )
                loss = self.train_losses[-1]
            else:
                mode = 'val'
                loss = self.val_losses[-1]

            # Write Loss
            self.summary_writer.write_scalar(
                f'Loss/{mode}', loss, epoch
            )

            if not train or self.record_train:
                for idx in range(len(self.metrics)):
                    for metric, info in self.metrics[idx].items():
                        self.summary_writer.write_scalar(
                            f'{idx}/{metric.title()}/{mode}',
                            info['value'], epoch
                        )

    def fit(self, start_epoch=1, epochs=None, reset=True, verbose=True):
        """Perform model training.

        Args:
            start_epoch (:obj:`int`, optional): Start epoch for training.
                (default: 1)
            epochs (:obj:`int`, optional): Numbers of epochs/iterations to
                train the model for. If no value is given, the original
                value given during initialization of learner will be used.
            reset (:obj:`bool`, optional): Flag to indicate that training
                is starting from scratch. (default: True)
            verbose (:obj:`bool`, optional): Print logs. (default: True)
        """

        if reset:
            self.reset_history()

        if epochs is not None:
            self.epochs = epochs

        for epoch in range(start_epoch, start_epoch + self.epochs):
            if verbose:
                print(f'Epoch {epoch}:')

            # Train an epoch
            self.train_epoch(verbose=verbose)
            self.write_summary(epoch, True)

            # Validate the model
            if self.val_loader is not None:
                self.validate(verbose=verbose)
                self.write_summary(epoch, False)

            # Save model checkpoint
            self.save_checkpoint(epoch)

            # Call Step LR
            if not self.lr_schedulers['step_lr'] is None:
                self.lr_schedulers['step_lr'].step()

            # Call Reduce LR on Plateau
            if not self.lr_schedulers['lr_plateau'] is None:
                self.lr_schedulers['lr_plateau'].step(self.val_losses[-1])
