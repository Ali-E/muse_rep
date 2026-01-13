from .utils import load_model_and_tokenizer, load_model
from .dataset import ForgetRetainDataset

import torch
import torch.nn.functional as F
from torch.cuda import device_count
import transformers
from transformers import Trainer, AutoModelForCausalLM


def unlearn(
    model_dir: str,
    data_file: str,
    out_dir: str,
    retain_data_file: str | None = None,
    loss_type: str = 'ga',
    per_device_batch_size: int = 8,
    epochs: int = 5,
    learning_rate=1e-5,
    max_len: int = 4096,
    tokenizer_dir: str | None = None,
    resume_from_checkpoint: bool = False,
    # forget_subset_indices: list[int] | None = None,
    portion: float = 1.0,
    exclude_file: str | None = None,
    include_file: str | None = None,
    rand_seed: int = 1,
    upsampling: float = 1.0,
    index_file: str | None = None,
    beta: float = 0.1,
    gamma: float = 0.0,
    npo_coeff: float = 1.0,
    coeff: float = 1.0,
    ps_file: str | None = None,
    use_wikitext: bool = False,
    wikitext_max_samples: int | None = None,
    wikitext_coeff: float = 1.0,
    retain_coeff: float = 1.0,
    retain_portion: float | None = None,
    save_only_final: bool = False,
):
    if 'gd' in loss_type:
        assert retain_data_file is not None or use_wikitext, "Retain data must be specified for grad_diff (either retain_data_file or use_wikitext)."

    if 'simnpo' in loss_type:
        print("Using SimNPO settings: ")
        print(" beta:", beta)
        print(" gamma:", gamma)
        # gamma = 0.3
        # beta = 1.0
    

    model, tokenizer = load_model_and_tokenizer(
        model_dir,
        tokenizer_dir=tokenizer_dir
    )

    ref_model = (
        load_model(model_dir)
        if 'npo' in loss_type or 'kl' in loss_type
        else None
    )


    # --- Multi-GPU support ---
    # n_gpus = torch.cuda.device_count()
    # if n_gpus > 1:
    #     print(f"Using {n_gpus} GPUs via DataParallel.")
    #     model = torch.nn.DataParallel(model, device_ids=list(range(n_gpus)))
    #     if ref_model is not None:
    #         ref_model = torch.nn.DataParallel(ref_model, device_ids=list(range(n_gpus)))
    # model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    # if ref_model is not None:
    #     ref_model = ref_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    # -------------------------

    dataset = ForgetRetainDataset(
        data_file,
        tokenizer=tokenizer,
        retain_file_path=retain_data_file,
        max_len=max_len,
        # forget_subset_indices=forget_subset_indices
        portion=portion,
        exclude_file=exclude_file,
        include_file=include_file,
        rand_seed=rand_seed,
        upsampling=upsampling,
        ps_file=ps_file,
        use_wikitext=use_wikitext,
        wikitext_max_samples=wikitext_max_samples,
        wikitext_coeff=wikitext_coeff,
        retain_coeff=retain_coeff,
        retain_portion=retain_portion
    )

    if device_count() == 0:
        raise ValueError("Device not detected!")

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=2,  # Effective batch size = per_device_batch_size * num_gpus * grad_accum_steps
        learning_rate=learning_rate,
        save_strategy='no' if save_only_final else 'epoch',  # Save only at end or every epoch
        num_train_epochs=epochs,
        optim='adamw_torch',
        lr_scheduler_type='constant',
        bf16=True,
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Recommended setting
        report_to='none',        # Disable wandb
        skip_memory_metrics=True   # For DataParallel compatibility
    )

    trainer = IterativeUnlearner(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=dataset.get_collate_fn(),
        loss_type=loss_type,
        beta=beta,
        gamma=gamma,
        npo_coeff=npo_coeff,
        coeff=coeff,
    )

    # ------------------------------------
    # If DataParallel, access config via .module
    # if hasattr(model, "module"):
    #     model.module.config.use_cache = False
    # else:
    #     model.config.use_cache = False  # silence the warnings.
    # ------------------------------------

    model.config.use_cache = False  # silence the warnings.
    
    # Gradient checkpointing is already configured in training_args

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(out_dir)



class IterativeUnlearner(Trainer):
    """Source: https://github.com/locuslab/tofu/blob/main/dataloader.py
    """

    def __init__(self, *args,
                 loss_type: str = 'ga',
                 ref_model: AutoModelForCausalLM | None = None,
                 beta: float = 0.1,
                 gamma: float = 0.0,
                 coeff: float = 1.0,
                 npo_coeff: float = 1.0,

                 **kwargs):
        self.loss_type = loss_type
        self.ref_model = ref_model
        self.beta = beta    # Only relevant when `'po' in self.loss_type`
        self.gamma = gamma
        self.coeff = coeff
        self.npo_coeff = npo_coeff
        
        # Store coefficients for retain data sources (passed from dataset)
        if 'train_dataset' in kwargs and hasattr(kwargs['train_dataset'], 'wikitext_coeff'):
            self.wikitext_coeff = kwargs['train_dataset'].wikitext_coeff
            self.retain_coeff = kwargs['train_dataset'].retain_coeff
        else:
            self.wikitext_coeff = 1.0
            self.retain_coeff = 1.0
        
        # Loss accumulation variables
        self.accumulated_loss = 0.0
        self.step_count = 0
        
        # List to store average loss at the end of each epoch
        self.epoch_avg_losses = []

        if ref_model is not None:
            assert 'po' in self.loss_type or 'kl' in self.loss_type
            ref_model = ref_model.eval()

        super().__init__(*args, **kwargs)


    def compute_loss(self, model, x, return_outputs=False):
        """Source: https://github.com/licong-lin/negative-preference-optimization/blob/main/synthetic/mymodel.py
        
        Now handles both WikiText and retain_file as separate regularization sources.
        x = (x_f, x_wikitext, x_retain)
        """
        
        ### 1. Unpack inputs ###
        x_f, x_wikitext, x_retain = x
        
        ### 2. Run model on forget data ###
        outputs_f = model(
            x_f['input_ids'],
            labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
            attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
        )
        loss_f = outputs_f.loss

        ### 3. Run model on retain data (combined from WikiText and/or retain_file) ###
        # Compute weighted retain loss from both sources
        loss_r = None
        if 'gdr' in self.loss_type or 'klr' in self.loss_type or 'simnpo' in self.loss_type:
            # Compute WikiText loss if available
            wikitext_loss = 0.0
            if x_wikitext is not None:
                outputs_wikitext = model(
                    x_wikitext['input_ids'],
                    labels=x_wikitext['labels'] if 'labels' in x_wikitext else x_wikitext['input_ids'].clone(),
                    attention_mask=x_wikitext['attention_mask'] if 'attention_mask' in x_wikitext else torch.ones_like(x_wikitext['input_ids'], dtype=torch.bool)
                )
                wikitext_loss = outputs_wikitext.loss * self.wikitext_coeff
            
            # Compute retain_file loss if available
            retain_file_loss = 0.0
            if x_retain is not None:
                outputs_retain = model(
                    x_retain['input_ids'],
                    labels=x_retain['labels'] if 'labels' in x_retain else x_retain['input_ids'].clone(),
                    attention_mask=x_retain['attention_mask'] if 'attention_mask' in x_retain else torch.ones_like(x_retain['input_ids'], dtype=torch.bool)
                )
                retain_file_loss = outputs_retain.loss * self.retain_coeff
            
            # Combined weighted retain loss
            loss_r = wikitext_loss + retain_file_loss

        ### 4. Compute reference model outputs if needed ###
        if 'klf' in self.loss_type or 'npo' in self.loss_type:
            with torch.no_grad():
                outputs_f_ref = self.ref_model(
                    x_f['input_ids'],
                    labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                    attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
                )

        outputs_r_ref = None
        if 'klr' in self.loss_type:
            # For KL regularization, need reference logits
            # Use the last computed retain data (prefer retain_file over wikitext)
            x_r_for_kl = x_retain if x_retain is not None else x_wikitext
            
            if x_r_for_kl is not None:
                with torch.no_grad():
                    outputs_r_ref = self.ref_model(
                        x_r_for_kl['input_ids'],
                        labels=x_r_for_kl['labels'] if 'labels' in x_r_for_kl else x_r_for_kl['input_ids'].clone(),
                        attention_mask=x_r_for_kl['attention_mask'] if 'attention_mask' in x_r_for_kl else torch.ones_like(x_r_for_kl['input_ids'], dtype=torch.bool)
                    )

        ### 2. Compute Loss ###
        loss = 0

        if 'ga' in self.loss_type:
            loss += -loss_f

        elif 'npo' in self.loss_type and 'simnpo' not in self.loss_type:
            neg_log_ratio = outputs_f_ref.logits - outputs_f.logits
            loss += -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta

        elif 'simnpo' in self.loss_type:
            neg_log_ratio = - outputs_f.logits - self.gamma
            loss += -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta

        else:
            raise NotImplementedError("Cannot infer the given loss type.")

        if 'gdr' in self.loss_type:
            if loss_r is not None:
                if 'simnpo' not in self.loss_type:
                    loss += loss_r
                else:
                    loss = self.npo_coeff * loss + self.coeff * loss_r

        if 'klf' in self.loss_type:
            raise NotImplementedError("KL forget not implemented yet!")

        if 'klr' in self.loss_type:
            if outputs_r_ref is not None:
                # Get the corresponding current model outputs for KL computation
                x_r_for_kl = x_retain if x_retain is not None else x_wikitext
                
                # We already computed these in the retain loss section
                if x_r_for_kl is not None:
                    if x_retain is not None:
                        current_logits = outputs_retain.logits
                    else:
                        current_logits = outputs_wikitext.logits
                    
                    kl_r = F.kl_div(
                        current_logits,
                        outputs_r_ref.logits,
                        reduction='batchmean',
                        log_target=True
                    )
                    
                    if 'simnpo' not in self.loss_type:
                        loss += kl_r
                    else:
                        loss += self.coeff * kl_r

        return (loss, outputs_f) if return_outputs else loss


    def prediction_step(self, model, x, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = x
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

    def log(self, logs):
        """Override log method to accumulate loss"""
        super().log(logs)
        
        # Accumulate training loss if present
        if 'train_loss' in logs:
            self.accumulated_loss += logs['train_loss']
            self.step_count += 1

    def on_epoch_end(self):
        """Store average loss at the end of each epoch"""
        if self.step_count > 0:
            avg_loss = self.accumulated_loss / self.step_count
            current_epoch = int(self.state.epoch)
            print(f"Epoch {current_epoch} - Average Loss: {avg_loss:.6f} (Total: {self.accumulated_loss:.6f} over {self.step_count} steps)")
            
            # Store the average loss for this epoch
            self.epoch_avg_losses.append(avg_loss)
            
            # Reset for next epoch
            self.accumulated_loss = 0.0
            self.step_count = 0
        
        super().on_epoch_end()

    def on_train_end(self):
        """Save epoch average losses to a file after training completes"""
        super().on_train_end()
        
        # Save the epoch average losses to a file
        import os
        import json
        
        output_dir = self.args.output_dir
        loss_file_path = os.path.join(output_dir, 'epoch_avg_losses.json')
        
        loss_data = {
            'epoch_avg_losses': self.epoch_avg_losses,
            'total_epochs': len(self.epoch_avg_losses),
            'loss_type': self.loss_type
        }
        
        with open(loss_file_path, 'w') as f:
            json.dump(loss_data, f, indent=2)
        
        print(f"Epoch average losses saved to: {loss_file_path}")
        print(f"Total epochs trained: {len(self.epoch_avg_losses)}")
        if self.epoch_avg_losses:
            print(f"Final epoch average loss: {self.epoch_avg_losses[-1]:.6f}")
