from .utils import load_model_and_tokenizer, load_model
from .dataset import ForgetRetainDataset

import torch
import torch.nn.functional as F
from torch.cuda import device_count
import transformers
from transformers import Trainer, AutoModelForCausalLM
import math
from torch.utils.data import DataLoader
from transformers import AdamW
import tqdm as tqdm


def forward_with_cache(model, inputs, module, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None 
    
    hook_handle = module.register_forward_hook(hook)
    
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
    hook_handle.remove()
    # import pdb;pdb.set_trac()
    return cache[0]


def get_params(model, layer_ids, param_ids):
    params = []
    for layer_id in layer_ids:
        for i, p in enumerate(model.model.layers[layer_id].parameters()):
            if i in param_ids:
                params.append(p)
    return params

def unlearn(
    model_dir: str,
    data_file: str,
    out_dir: str,
    index_file : str,
    retain_data_file: str | None = None,
    batch_size: int = 2,
    epochs: int = 5,
    learning_rate=1e-5,
    max_len: int = 2048,
    tokenizer_dir: str | None = None,
    resume_from_checkpoint: bool = False,
    layer_id=7,
    layer_ids=[5,6,7],
    param_ids=[6],
    steering_coeffs = 6.5,
    alpha=1200,
    module_str="{model_name}.model.layers[{layer_id}]",
):

    updated_model, tokenizer = load_model_and_tokenizer(
        model_dir,
        tokenizer_dir=tokenizer_dir
    )

    frozen_model = (
        load_model(model_dir)
    )

    dataset = ForgetRetainDataset(
        data_file,
        tokenizer=tokenizer,
        retain_file_path=retain_data_file,
        max_len=max_len,
        index_file=index_file,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.get_collate_fn())

    updated_model = updated_model.train()
    params = get_params(updated_model, layer_ids, param_ids)
    optimizer = AdamW(params, lr=learning_rate)

    ## module_str what is this
    frozen_module = eval(module_str.format(model_name="frozen_model", layer_id=layer_id))
    updated_module = eval(module_str.format(model_name="updated_model", layer_id=layer_id))


    random_vector = torch.rand(1,1, updated_model.config.hidden_size, dtype=updated_model.dtype, device=updated_model.device)
    control_vec = random_vector / torch.norm(random_vector) * steering_coeffs

    num_batches = math.ceil(len(dataset)/batch_size)

    
    for epoch in range(epochs):
        print(f"======= Epoch {epoch} =======")
        with tqdm.tqdm(total=num_batches) as pbar:
            for iterno, (forget_data, retain_data) in enumerate(dataloader):
    
    
                updated_forget_activations = forward_with_cache(
                    updated_model, forget_data, module=updated_module, no_grad=False
                ).to(updated_model.device)

                unlearn_loss = torch.nn.functional.mse_loss(updated_forget_activations, control_vec)

                # Retain loss
                updated_retain_activations = forward_with_cache(
                    updated_model, retain_data, module=updated_module, no_grad=False
                ).to(updated_model.device)
                frozen_retain_activations = forward_with_cache(
                    frozen_model, retain_data, module=frozen_module, no_grad=True
                ).to(updated_model.device)

                retain_loss = torch.nn.functional.mse_loss(updated_retain_activations, frozen_retain_activations)

                # Update model
                # import pdb;pdb.set_trace()
                loss = unlearn_loss + alpha*retain_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"loss: {loss.item():.4g} | unlearn_loss: {unlearn_loss.item():.4g} | retain_loss: {retain_loss.item():.4g} | param_change: {params[0].grad.abs().mean().item():.4g}")
                

                pbar.update(1)
                # if iterno == num_batches-1:
                #     break 

    updated_model.save_pretrained(f'{out_dir}/Epoch_{epoch+1}/')
    tokenizer.save_pretrained(f'{out_dir}/Epoch_{epoch+1}/')
    print(f"Saved model to {out_dir}/Epoch_{epoch+1}")

  
