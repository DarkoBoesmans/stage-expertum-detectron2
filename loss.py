from detectron2.engine import DefaultTrainer, HookBase
from detectron2.data import build_detection_train_loader
import detectron2.utils.comm as comm
import torch
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)


class LossEvalHook(HookBase):
    """A hook to evaluate loss on the validation dataset during training."""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST  # Use TEST dataset for validation
        self._period = 20  # Run validation every 20 iterations
        self._logger = logging.getLogger(__name__)
    
    def _do_eval(self):
        data_loader = build_detection_train_loader(self.cfg)
        total_losses = []
        
        for idx, inputs in enumerate(data_loader):
            if idx >= 10:  # Limit to 10 batches for efficiency
                break
            with torch.no_grad():
                loss_dict = self.trainer.model(inputs)
                losses = sum(loss_dict.values())
                total_losses.append(losses)
        
        if len(total_losses) > 0:
            mean_loss = torch.stack(total_losses).mean().item()
            self.trainer.storage.put_scalar("validation_loss", mean_loss)
            # Direct logger call instead of log_every_n_seconds
            self._logger.info(f"Validation loss at iteration {self.trainer.iter}: {mean_loss:.4f}")
        else:
            self._logger.warning("No validation data was successfully processed")
    
    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter % self._period == 0:
            self._do_eval()


class ValidationLoss(DefaultTrainer):
    """
    A trainer class with validation loss computation capabilities.
    Extends DefaultTrainer by adding validation loss evaluation during training.
    """
    
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): The detectron2 config node.
        """
        super().__init__(cfg)
        print("Initialized ValidationLoss trainer with validation loss computation")
    
    def build_hooks(self):
        """
        Build hooks for training, adding our validation loss hook.
        Returns:
            list[HookBase]: List of training hooks.
        """
        hooks = super().build_hooks()
        # Add the validation loss hook before the end-of-iteration hook
        hooks.insert(-1, LossEvalHook(self.cfg))
        return hooks
