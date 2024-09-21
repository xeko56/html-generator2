from pytorch_lightning.callbacks import Callback
import hashlib
import torch

class PushToHubCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
        pl_module.model.push_to_hub("xeko56/html-generator",
                                    commit_message=f"Training in progress, epoch {trainer.current_epoch}")

    def on_train_end(self, trainer, pl_module):
        print(f"Pushing model to the hub after training")
        pl_module.processor.push_to_hub("xeko56/html-generator",
                                    commit_message=f"Training done")
        pl_module.model.push_to_hub("xeko56/html-generator",
                                    commit_message=f"Training done")
        
        # Compute and log the state dictionary hash
        state_dict_hash = self._get_model_hash(pl_module.model)
        print(f"State dict hash at epoch {trainer.current_epoch}: {state_dict_hash}")
        
        # Log the state dict hash (you could also log this to Wandb or any other logger)
        trainer.logger.experiment.log({"state_dict_hash": state_dict_hash})        
        
    def _get_model_hash(self, model):
        """Compute the hash of the model's state dict."""
        state_dict = model.state_dict()
        model_bytes = b''.join(param.cpu().numpy().tobytes() for param in state_dict.values())
        return hashlib.md5(model_bytes).hexdigest()        