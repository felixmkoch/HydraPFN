#------------------------------------------------------------------------------------------------
#                                        IMPORTS
#------------------------------------------------------------------------------------------------
import os
print(f"currently in {os.getcwd()}")

import json
import torch
import wandb
import numpy as np

from hydrapfn.scripts.model_trainer import train_model
from hydrapfn.scripts.model_saver import save_model

from hydrapfn.scripts.eval_helper import EvalHelper

#------------------------------------------------------------------------------------------------
#                                    DEFAULT SETTINGS
#------------------------------------------------------------------------------------------------
base_path = '.'
json_file_path = "tabpfn_original_config.json"
max_features = 100

with open(json_file_path, "r") as f: 
    config = json.load(f)

uniform_int_sampler_f = (lambda a, b : lambda : round(np.random.uniform(a, b)))
choice_values = [
    torch.nn.modules.activation.Tanh, 
    torch.nn.modules.linear.Identity,
    torch.nn.modules.activation.ReLU
    ]

config["differentiable_hyperparameters"]["prior_mlp_activations"]["choice_values"] = choice_values
config["num_classes"] = uniform_int_sampler_f(2, config['max_num_classes']) # Wrong Function
config["num_features_used"] = uniform_int_sampler_f(1, max_features)

device = "cuda:0"

#------------------------------------------------------------------------------------------------
#                                          CUSTOM
#------------------------------------------------------------------------------------------------

# Reduce size for quick smoke run (safe, behaviour unchanged)
config['batch_size'] = 64
config['emsize'] = 128
#config['d_intermediate'] = 2 * config['emsize']
config['d_intermediate'] = 0
config["epochs"] = 1500
config["bptt"] = 1024
config["max_eval_pos"] = 1000

config["num_steps"] = 32

config["nlayers"] = 8
config["enable_autocast"] = True

# Mode of the cross attention. "none" -> No cross-attn; "single" -> normal cross-attn; "dual_sum" -> sum with input enc; "dual_concat" -> concat with input enc. 
config["cross_attention_mode"] = "dual_concat"
# Regularization to punish deviations from permutated hidden states.
config["perm_reg_lam"] = 0.05

config["use_tabicl_prior"] = True

config["use_col_embedding"] = True

config["model_type"] = "hydrapfn"     # {"hydra_full", "bimamba2", "hydra", "hydrapfn"}

config["loss_label_smoothing"] = 0.1    # Label smoothing for the cross entropy loss. If set to 0.0, no label smoothing is done.

#------------------------------------------------------------------------------------------------
#                                           WANDB
#------------------------------------------------------------------------------------------------

wandb_project = "hydrapfn"
wandb_job_type = f"train"
wandb_run_name = f"hydraicl_local"

wandb_config= config

wandb_run = wandb.init(project=wandb_project,job_type=wandb_job_type,config=wandb_config, name=wandb_run_name, group="DDP")

eval_class = EvalHelper()

#------------------------------------------------------------------------------------------------
#                                           MODEL
#------------------------------------------------------------------------------------------------

model, optimizer = train_model(
    config=config,
    evaluation_class=eval_class,
    best_model_path="hydrapfn/trained_models/hydraicl_local.cpkt",
    model_saver=save_model,
    continue_training={}
)


print("worked")