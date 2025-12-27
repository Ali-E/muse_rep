from .dataset import DefaultDataset
from .utils import load_model_and_tokenizer

import transformers


def finetune(
    model_dir: str,
    data_file: str,
    out_dir: str,
    epochs: int = 5,
    per_device_batch_size: int = 2,
    learning_rate: float = 1e-5,
    max_len: int = 4096,
    tokenizer_dir: str | None = None,
    portion: float = 1.0,
    exclude_file: str | None = None,
    include_file: str | None = None,
    rand_seed: int = 1,
    upsampling: float = 1.0,
    ps_file: str | None = None
):
    model, tokenizer = load_model_and_tokenizer(
        model_dir,
        tokenizer_dir=tokenizer_dir
    )

    dataset = DefaultDataset(
        data_file,
        tokenizer=tokenizer,
        max_len=max_len,
        portion=portion,
        exclude_file=exclude_file,
        include_file=include_file,
        rand_seed=rand_seed,
        upsampling=upsampling,
        ps_file=ps_file
    )

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        optim='adamw_torch',
        lr_scheduler_type='cosine',
        bf16=True,
        report_to='none'        # Disable wandb
    )

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=dataset.get_collate_fn()
    )

    model.config.use_cache = False  # silence the warnings.
    trainer.train()
    trainer.save_model(out_dir)
