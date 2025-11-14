import logging
import os
import re
from typing import Any, Type

import numpy as np
import torch
import torch.nn as nn
from f9columnar.ml.dataloader_helpers import ColumnSelection, column_selection_from_dict
from f9columnar.ml.hdf5_dataloader import get_column_selection
from f9columnar.ml.scalers import CategoricalFeatureScaler
from f9columnar.utils.helpers import load_json, load_pickle
from omegaconf import DictConfig, ListConfig, open_dict

from seesaw.models.deep_sets import FlatDeepSets, JaggedDeepsets
from seesaw.models.mlp import MLP
from seesaw.models.nn_modules import BaseLightningModule
from seesaw.models.res_net import ResNet
from seesaw.models.transformers.event_transformer import EventTransformer
from seesaw.models.transformers.jagged_transformer import JaggedTransformer


def get_categories(
    dataset_conf: DictConfig, categ_columns: list[str], postfix: str
) -> tuple[np.ndarray | None, list[dict[float, int]] | None]:
    if len(categ_columns) == 0:
        return None, None

    feature_scaling = dataset_conf.get("feature_scaling", None)

    if feature_scaling is None:
        raise ValueError("Feature scaling must be enabled! Run the preprocessing script first.")

    scaler_path = feature_scaling.save_path

    if scaler_path is None:
        raise ValueError("Scaler path must be provided!")

    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(os.environ["ANALYSIS_ML_RESULTS_DIR"], "scalers")
        logging.warning(f"Using fallback scaler path: {scaler_path}")

    categ_scaler: CategoricalFeatureScaler | None

    categ_scaler = CategoricalFeatureScaler("categorical", save_path=scaler_path)
    categ_scaler = categ_scaler.load(categ_columns, postfix=postfix, extra_hash=str(dataset_conf.files))

    if categ_scaler is None:
        raise RuntimeError("Failed to load categorical feature scaler!")

    unique_categories = categ_scaler.get_unique_categories()
    # list of ints with the number of categories for each categorical column
    num_categories = [len(unique_categories[col]) for col in unique_categories.keys()]

    return np.array(num_categories), categ_scaler.categories


def _build_events_network(
    selection: ColumnSelection,
    dataset_conf: DictConfig,
    architecture_config: DictConfig,
) -> nn.Module:
    events_selection = selection["events"]

    model_name = architecture_config.model

    categ_columns = events_selection.categ_columns
    numer_columns_idx = events_selection.offset_numer_columns_idx
    categ_columns_idx = events_selection.offset_categ_columns_idx

    categories, _ = get_categories(dataset_conf, categ_columns, postfix="events_0")

    classes = dataset_conf.get("classes", None)
    if classes is not None:
        output_dim = len(classes)
    else:
        output_dim = 1

    if output_dim == 2:
        output_dim = 1

    logging.info(f"Model output dimension {output_dim}.")

    embedding_config = architecture_config.get("embeddings", None)
    post_embeddings_config = architecture_config.get("post_embeddings", None)

    if embedding_config is not None:
        embedding_config = dict(embedding_config)
    else:
        embedding_config = {}

    if embedding_config.get("use_ple", False):
        n_bins = dataset_conf.get("ple_bins", None)

        if n_bins is None:
            raise ValueError("ple_bins must be specified in dataset_config for using PLE!")

        ple_config = {}

        ple_config["n_bins"] = n_bins
        ple_config["learn_bins"] = embedding_config.get("learn_ple_bins", False)
        ple_config["uniform_bins"] = embedding_config.get("uniform_ple_bins", False)
        ple_config["ple_file_hash_str"] = str(dataset_conf.files) + str(sorted(dataset_conf.features)) + str(n_bins)

        embedding_config.pop("use_ple")
    else:
        ple_config = None

    embedding_config["ple_config"] = ple_config

    if post_embeddings_config is not None:
        embedding_config["post_embeddings_dct"] = dict(post_embeddings_config)

    model: nn.Module
    if model_name == "MLP":
        if architecture_config.get("layers_dim", None) is None:
            layers_dim = None
        else:
            layers_dim = [architecture_config["input_dim"]] + architecture_config["layers_dim"] + [output_dim]

        n_layers = architecture_config.get("n_layers", None)
        n_hidden = architecture_config.get("n_hidden", None)

        model = MLP(
            numer_idx=numer_columns_idx,
            categ_idx=categ_columns_idx,
            output_dim=output_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            layers_dim=layers_dim,
            act=architecture_config["act"],
            act_out=architecture_config.get("act_out", None),
            use_batchnorm=architecture_config.get("batchnorm", False),
            dropout=architecture_config.get("dropout", 0.0),
            categories=categories,
            embedding_config_dct=embedding_config,
            disable_embeddings=architecture_config.get("disable_embeddings", False),
            **architecture_config.get("act_kwargs", {}),
        )
    elif model_name == "ResNet":
        if architecture_config.get("res_layers", None) is None:
            res_layers = None
        else:
            res_layers = architecture_config["res_layers"]

        num_blocks = architecture_config.get("num_blocks", None)
        n_layers = architecture_config.get("n_layers", None)
        n_hidden = architecture_config.get("n_hidden", None)
        block_size = architecture_config.get("block_size", 2)

        if num_blocks is None and n_layers is not None:
            num_blocks = n_layers

        model = ResNet(
            numer_idx=numer_columns_idx,
            categ_idx=categ_columns_idx,
            output_dim=output_dim,
            num_blocks=num_blocks,
            n_hidden=n_hidden,
            block_size=block_size,
            res_layers=res_layers,
            act=architecture_config["act"],
            act_out=architecture_config.get("act_out", None),
            use_batchnorm=architecture_config.get("batchnorm", True),
            dropout=architecture_config.get("dropout", 0.1),
            categories=categories,
            embedding_config_dct=embedding_config,
            disable_embeddings=architecture_config.get("disable_embeddings", False),
            **architecture_config.get("act_kwargs", {}),
        )
    elif model_name == "DeepSets":
        encoder_layers = architecture_config.get("encoder_layers", 2)
        decoder_layers = architecture_config.get("decoder_layers", 2)

        if type(encoder_layers) is ListConfig:
            encoder_layers = list(encoder_layers)

        if type(decoder_layers) is ListConfig:
            decoder_layers = list(decoder_layers)

        model = FlatDeepSets(
            numer_idx=numer_columns_idx,
            categ_idx=categ_columns_idx,
            output_dim=output_dim,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            n_hidden=architecture_config["n_hidden"],
            embedding_dim=architecture_config.get("embedding_dim", None),
            act=architecture_config["act"],
            act_out=architecture_config.get("act_out", None),
            use_batchnorm=architecture_config.get("batchnorm", True),
            dropout=architecture_config.get("dropout", 0.0),
            categories=categories,
            embedding_config_dct=embedding_config,
            **architecture_config.get("act_kwargs", {}),
        )
    elif model_name == "EventTransformer":
        model = EventTransformer(
            numer_idx=numer_columns_idx,
            categ_idx=categ_columns_idx,
            embedding_dim=architecture_config.get("embedding_dim", 32),
            transformer_depth=architecture_config.get("transformer_depth", 6),
            ff_hidden_mult=architecture_config.get("ff_hidden_mult", 4),
            heads=architecture_config.get("heads", 8),
            dim_head=architecture_config.get("dim_head", 16),
            dim_out=output_dim,
            attn_dropout=architecture_config.get("attn_dropout", 0.1),
            ff_dropout=architecture_config.get("ff_dropout", 0.1),
            act_out=architecture_config.get("act_out", None),
            categories=categories,
            flash=architecture_config.get("flash_attention", False),
            embedding_config_dct=embedding_config,
            remove_first_attn_residual=architecture_config.get("remove_first_attn_residual", False),
            remove_first_attn_layernorm=architecture_config.get("remove_first_attn_layernorm", True),
            use_cls_token=architecture_config.get("use_cls_token", True),
        )
    else:
        raise NotImplementedError("The model set in the params is not yet implemented!")

    return model


def _build_jagged_network(
    selection: ColumnSelection,
    dataset_conf: DictConfig,
    architecture_config: DictConfig,
) -> nn.Module:
    model_name = architecture_config.model

    categ_columns_dct, numer_columns_idx_dct, categ_columns_idx_dct = {}, {}, {}

    for dataset_name in selection.keys():
        categ_columns_dct[dataset_name] = selection[dataset_name].categ_columns
        numer_columns_idx_dct[dataset_name] = selection[dataset_name].offset_numer_columns_idx
        categ_columns_idx_dct[dataset_name] = selection[dataset_name].offset_categ_columns_idx

    numer_padding_tokens: dict[str, float | None] = {}
    for dataset_name in selection.keys():
        if dataset_name == "events":
            continue
        numer_padding_tokens[dataset_name] = selection[dataset_name].pad_value

    categ_padding_tokens: dict[str, int | None] = {}
    for dataset_name in selection.keys():
        if dataset_name == "events":
            continue

        first_obj_name = f"{dataset_name}_0"
        _, categories_lst = get_categories(dataset_conf, categ_columns_dct[dataset_name], first_obj_name)

        original_token = numer_padding_tokens[dataset_name]

        if categories_lst is None or original_token is None:
            categ_padding_tokens[dataset_name] = None
        else:
            categ_padding_tokens[dataset_name] = categories_lst[0][original_token]

    categories_dct: dict[str, np.ndarray] = {}

    for dataset_name in selection.keys():
        if dataset_name == "events":
            object_names = ["events_0"]
        else:
            dataset_shape = selection[dataset_name].shape
            object_names = [f"{dataset_name}_{i}" for i in range(dataset_shape[1])]

        obj_categories = []
        for obj_name in object_names:
            categories_arr, _ = get_categories(dataset_conf, categ_columns_dct[dataset_name], obj_name)

            if categories_arr is None:
                categories_arr = np.array([])

            obj_categories.append(categories_arr)

        if dataset_name == "events":
            categories_dct[dataset_name] = obj_categories[0]
        else:
            categories_dct[dataset_name] = np.stack(obj_categories, axis=0)  # (objects, features)

    object_dims = {}  # not used for now
    for dataset_name in selection.keys():
        if dataset_name == "events":
            continue

        object_dims[dataset_name] = selection[dataset_name].shape[1]

    classes = dataset_conf.get("classes", None)
    if classes is not None:
        output_dim = len(classes)
    else:
        output_dim = 1

    if output_dim == 2:
        output_dim = 1

    logging.info(f"Model output dimension {output_dim}.")

    embedding_config = architecture_config.get("embeddings", None)
    post_embeddings_config = architecture_config.get("post_embeddings", None)

    if embedding_config is not None:
        embedding_config = dict(embedding_config)
    else:
        embedding_config = {}

    if embedding_config.get("use_ple", False):
        n_bins = dataset_conf.get("ple_bins", None)

        if n_bins is None:
            raise ValueError("ple_bins must be specified in dataset_config for using PLE!")

        ple_config = {}

        ple_config["n_bins"] = n_bins
        ple_config["learn_bins"] = embedding_config.get("learn_ple_bins", False)
        ple_config["uniform_bins"] = embedding_config.get("uniform_ple_bins", False)
        ple_config["ple_file_hash_str"] = str(dataset_conf.files) + str(sorted(dataset_conf.features)) + str(n_bins)

        embedding_config.pop("use_ple")
    else:
        ple_config = None

    embedding_config["ple_config"] = ple_config

    if post_embeddings_config is not None:
        embedding_config["post_embeddings_dct"] = dict(post_embeddings_config)

    use_flash = architecture_config.get("flash_attention", False)
    if use_flash:
        logging.info("[yellow]Using flash attention!")

    if "flat_model_config" in architecture_config:
        flat_model_arhitecture = architecture_config.flat_model_config.architecture_config
        events_model = _build_events_network(
            selection=selection,
            dataset_conf=dataset_conf,
            architecture_config=flat_model_arhitecture,
        )
    else:
        events_model = None

    model: nn.Module

    if model_name == "DeepSets":
        encoder_layers = architecture_config.get("encoder_layers", 2)
        decoder_layers = architecture_config.get("decoder_layers", 2)

        if type(encoder_layers) is ListConfig:
            encoder_layers = list(encoder_layers)

        if type(decoder_layers) is ListConfig:
            decoder_layers = list(decoder_layers)

        model = JaggedDeepsets(
            numer_idx=numer_columns_idx_dct,
            categ_idx=categ_columns_idx_dct,
            categories=categories_dct,
            output_dim=output_dim,
            object_dimensions=object_dims,
            numer_padding_tokens=numer_padding_tokens,
            categ_padding_tokens=categ_padding_tokens,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            n_hidden=architecture_config["n_hidden"],
            embedding_dim=architecture_config.get("embedding_dim", None),
            act=architecture_config["act"],
            act_out=architecture_config.get("act_out", None),
            use_batchnorm=architecture_config.get("batchnorm", True),
            dropout=architecture_config.get("dropout", 0.0),
            mean_pooling=architecture_config.get("mean_pooling", True),
            embedding_config_dct=embedding_config,
            flat_model=events_model,
            flat_fuse=architecture_config.get("flat_fuse", {}),
            **architecture_config.get("act_kwargs", {}),
        )
    elif model_name == "JaggedTransformer":
        model = JaggedTransformer(
            embedding_dim=architecture_config.get("embedding_dim", 64),
            numer_idx=numer_columns_idx_dct,
            categ_idx=categ_columns_idx_dct,
            categories=categories_dct,
            numer_padding_tokens=numer_padding_tokens,
            categ_padding_tokens=categ_padding_tokens,
            object_dimensions=object_dims,
            heads=architecture_config.get("heads", 8),
            encoder_depth=architecture_config.get("encoder_depth", 2),
            decoder_depth=architecture_config.get("decoder_depth", 2),
            dim_head=architecture_config.get("dim_head", 16),
            ff_hidden_mult=architecture_config.get("ff_hidden_mult", 4),
            dim_out=output_dim,
            act_out=architecture_config.get("act_out", None),
            attn_dropout=architecture_config.get("attn_dropout", 0.0),
            ff_dropout=architecture_config.get("ff_dropout", 0.0),
            seed_strategy=architecture_config.get("seed_strategy", "pooling"),
            set_predictor_dct=dict(architecture_config.get("set_predictor", {})),
            cross_decoder_depth=architecture_config.get("cross_decoder_depth", 2),
            embedding_config_dct=embedding_config,
            set_transform_events=architecture_config.get("set_transform_events", False),
            use_setnorm=architecture_config.get("use_setnorm", True),
            flat_model=events_model,
            flat_fuse=architecture_config.get("flat_fuse", {}),
            use_flash=use_flash,
            debug_masks=architecture_config.get("debug_masks", False),
        )
    else:
        raise NotImplementedError("The model set in the params is not yet implemented!")

    return model


def build_network(
    dataset_conf: DictConfig, model_conf: DictConfig, run_name: str | None = None
) -> tuple[nn.Module, str, ColumnSelection]:
    architecture_config = model_conf.architecture_config
    model_name = architecture_config.model

    path_to_selection = os.path.join(model_conf.training_config.model_save_path, f"{run_name}_selection.json")

    if os.path.exists(path_to_selection):
        logging.info(f"Loading saved column selection from {path_to_selection}.")
        selection_dct = load_json(path_to_selection)
        selection = column_selection_from_dict(selection_dct)
    else:
        selection = get_column_selection(dataset_conf.files, dataset_conf.features)
        logging.info(f"Using selection {str(selection)}.")

        if run_name is not None:
            os.makedirs(model_conf.training_config.model_save_path, exist_ok=True)
            selection.to_json(path_to_selection)
            logging.info(f"Saved column selection to {run_name}_selection.json.")
        else:
            logging.warning("Run name not provided! The selection will not be saved.")

    for c in selection["events"].used_columns:
        if "mlmass" in c:
            logging.info("[yellow]Using parametrized neural network.")
            break

    categ_columns_dct, numer_columns_idx_dct, categ_columns_idx_dct = {}, {}, {}

    for dataset_name in selection.keys():
        categ_columns_dct[dataset_name] = selection[dataset_name].categ_columns
        numer_columns_idx_dct[dataset_name] = selection[dataset_name].offset_numer_columns_idx
        categ_columns_idx_dct[dataset_name] = selection[dataset_name].offset_categ_columns_idx

    if len(selection) == 1 and "events" in selection:
        events_only = True
    else:
        events_only = False

    if events_only:
        model = _build_events_network(selection, dataset_conf, architecture_config)
    else:
        model = _build_jagged_network(selection, dataset_conf, architecture_config)

    if architecture_config.get("compile", False):
        compile_kwargs = architecture_config.get("compile_kwargs", {})
        logging.info(f"Compiling the model with torch.compile and args: {compile_kwargs}")
        model.compile(**compile_kwargs)

    return model, model_name, selection


def load_reports(checkpoint_path: str, add_selection: bool = False) -> dict[str, Any]:
    checkpoint_path_split = checkpoint_path.split("/")

    match = re.match(r"^(.*?)_(epoch|last)", checkpoint_path_split[-1])

    if match:
        match_group = match.group(1)
    else:
        raise ValueError("Could not extract reports path from checkpoint path!")

    base_path = "/".join(checkpoint_path_split[:-1])

    reports_path = os.path.join(base_path, f"{match_group}_reports.p")
    selection_path = os.path.join(base_path, f"{match_group}_selection.json")

    reports = load_pickle(reports_path)

    if add_selection:
        selection = load_json(selection_path)
        reports["selection"] = selection

    return reports


def load_model_from_config(
    config: DictConfig,
    model_class: Type[BaseLightningModule],
    model_config: DictConfig | None = None,
    checkpoint_path: str | None = None,
    disable_compile: bool = False,
    **kwargs: Any,
) -> tuple[BaseLightningModule, str]:
    """Load a model from the configuration given load_checkpoint and model_save_path.

    Parameters
    ----------
    config : DictConfig
        Hydra configuration object containing model and dataset configurations.
    model_config : DictConfig | None, optional
        Model configuration. If None, it will be taken from config.model_config.
    model_class : Type[BaseLightningModule]
        The class of the model to be loaded.
    checkpoint_path : str | None, optional
        Direct path to the checkpoint file. If provided, this will override the load_checkpoint in the config.
    disable_compile : bool, optional
        If True, disables model compilation even if it was enabled in the architecture_config, by default False.
    **kwargs : Any
        Additional keyword arguments to pass to torch.load.

    Note
    ----
    Make sure to have valid scaler objects saved in the scaler path specified in the configuration.

    Returns
    -------
    tuple[BaseLightningModule, str]
        The loaded lightning model and the path to the checkpoint.

    """
    if model_config is None:
        model_conf = config.model_config
    else:
        model_conf = model_config

    if checkpoint_path is None:
        if model_conf.get("load_checkpoint", None) is not None:
            load_checkpoint = os.path.join(model_conf.training_config.model_save_path, model_conf.load_checkpoint)
        else:
            raise ValueError("No checkpoint provided for model!")
    else:
        load_checkpoint = checkpoint_path

    accelerator = config.experiment_config.accelerator
    logging.info(f"Loading model from checkpoint: {load_checkpoint} on {accelerator}")

    if accelerator == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(accelerator)

    loaded = torch.load(load_checkpoint, map_location=device, weights_only=False, **kwargs)
    state_dict, hyper_params = loaded["state_dict"], loaded["hyper_parameters"]

    if "num_model" in hyper_params:
        hyper_params.pop("num_model")

    if "den_model" in hyper_params:
        hyper_params.pop("den_model")

    loaded_features = set(hyper_params["dataset_conf"]["features"])
    config_features = set(config.dataset_config.features)

    if loaded_features != config_features:
        raise ValueError(
            "Dataset features in the loaded model do not match the current dataset configuration! "
            f"Loaded: {loaded_features}, Current: {config_features}"
        )

    loaded_numer_scaler_type = hyper_params["dataset_conf"]["feature_scaling"].get("numer_scaler_type", None)
    config_numer_scaler_type = config.dataset_config.feature_scaling.get("numer_scaler_type", None)

    if loaded_numer_scaler_type != config_numer_scaler_type:
        raise ValueError(
            "Numerical feature scaling type in the loaded model does not match the current dataset configuration! "
            f"Loaded: {loaded_numer_scaler_type}, Current: {config_numer_scaler_type}"
        )

    loaded_categ_scaler_type = hyper_params["dataset_conf"]["feature_scaling"].get("categ_scaler_type", None)
    config_categ_scaler_type = config.dataset_config.feature_scaling.get("categ_scaler_type", None)

    if loaded_categ_scaler_type != config_categ_scaler_type:
        raise ValueError(
            "Categorical feature scaling type in the loaded model does not match the current dataset configuration! "
            f"Loaded: {loaded_categ_scaler_type}, Current: {config_categ_scaler_type}"
        )

    compile_model = hyper_params["model_conf"]["architecture_config"].get("compile", False)

    with open_dict(hyper_params["model_conf"]):
        hyper_params["model_conf"]["architecture_config"]["compile"] = False

    with open_dict(model_conf):
        model_conf["architecture_config"] = hyper_params["model_conf"]["architecture_config"]

    hyper_params["dataset_conf"] = config.dataset_config
    hyper_params["model_conf"] = model_conf

    model = model_class(**hyper_params).to(device)
    model.load_state_dict(state_dict)

    if compile_model and not disable_compile:
        compile_kwargs = model_conf.architecture_config.get("compile_kwargs", {})
        logging.info(f"Compiling the model with torch.compile and args: {compile_kwargs}")
        model.compile(**compile_kwargs)

    model.eval()

    return model, load_checkpoint
