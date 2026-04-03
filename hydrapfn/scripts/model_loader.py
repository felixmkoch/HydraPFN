import os
import torch
import torch.nn as nn

from hydrapfn.hydra_icl import HydraModel
from hydrapfn.bimamba_icl import BiMamba2Model
from hydrapfn.hydra_full import HydraFullModel

from hydrapfn.scripts.model_trainer import train_model


def load_model(checkpoint_path, device="cuda"):
    """
    Load a single HydraPFN checkpoint from a given full path.

    :param checkpoint_path: Full path to .cpkt file
    :param device: Device to load model onto ('cpu' or 'cuda')
    :param verbose: Verbosity flag passed to get_model
    :return: (model, config_sample)
    """

    print("!! Warning: GPyTorch must be installed !!")
    print(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model_state = checkpoint["model_state_dict"]
    optimizer_state = checkpoint["optimizer"]
    config_sample = checkpoint["config"]

    # Fix activation config if needed
    if (
        "differentiable_hyperparameters" in config_sample
        and "prior_mlp_activations"
        in config_sample["differentiable_hyperparameters"]
    ):
        config_sample["differentiable_hyperparameters"][
            "prior_mlp_activations"
        ]["choice_values_used"] = config_sample[
            "differentiable_hyperparameters"
        ]["prior_mlp_activations"]["choice_values"]

        config_sample["differentiable_hyperparameters"][
            "prior_mlp_activations"
        ]["choice_values"] = [
            torch.nn.Tanh
            for _ in config_sample["differentiable_hyperparameters"][
                "prior_mlp_activations"
            ]["choice_values"]
        ]

    # Override training-specific settings for safe inference
    config_sample["categorical_features_sampler"] = lambda: lambda x: ([], [], [])

    config_sample["epochs"] = 0

    # Build model according to saved configuration
    model_type = str(config_sample.get("model_type", "hydra")).lower()

    if model_type == "bimamba2":
        model = BiMamba2Model(
            encoder=None,
            y_encoder=None,
            n_out=config_sample.get("num_classes", 1),
            ninp=config_sample.get("emsize", 64),
            nhid=config_sample.get("nhid", 128),
            num_layers=config_sample.get("nlayers", 1),
            cross_attention_mode=config_sample.get("cross_attention_mode", "single"),
            num_heads=config_sample.get("num_heads", 4),
            use_col_embedding=config_sample.get("use_col_embedding", False),
            device=device,
        )
    elif model_type == "hydra_full":
        model = HydraFullModel(
            encoder=None,
            y_encoder=None,
            n_out=config_sample.get("num_classes", 1),
            ninp=config_sample.get("emsize", 64),
            nhid=config_sample.get("nhid", 128),
            num_layers=config_sample.get("nlayers", 1),
            use_col_embedding=config_sample.get("use_col_embedding", False),
            device=device,
        )
    else:  # default to hydra
        model = HydraModel(
            encoder=None,
            y_encoder=None,
            n_out=config_sample.get("num_classes", 1),
            ninp=config_sample.get("emsize", 64),
            nhid=config_sample.get("nhid", 128),
            num_layers=config_sample.get("nlayers", 1),
            cross_attention_mode=config_sample.get("cross_attention_mode", "single"),
            num_heads=config_sample.get("num_heads", 4),
            use_col_embedding=config_sample.get("use_col_embedding", False),
            device=device,
        )

    # Remove potential DataParallel prefix
    model_state = {k.replace("module.", ""): v for k, v in model_state.items()}

    # Load weights
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    optimizer.load_state_dict(optimizer_state)

    return model, optimizer, config_sample


def load_hydrapfn_model(checkpoint_path, device="cuda"):
    """Legacy loader alias for backward compatibility."""
    return load_model(checkpoint_path, device=device)
