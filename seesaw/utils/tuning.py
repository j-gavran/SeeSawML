from typing import Any

import optuna
from omegaconf import DictConfig, open_dict


def get_optuna_suggest(
    trial: optuna.trial.Trial,
    name: str,
    suggest_type: str = "none",
    low: float | int | None = None,
    high: float | int | None = None,
    step: int = 1,
    log: bool = False,
    choices: list[int] | None = None,
) -> Any:
    if suggest_type == "categorical":
        if choices is None:
            raise ValueError("choices must be provided for categorical suggest_type!")

        return trial.suggest_categorical(name, choices)

    if suggest_type == "int":
        if type(low) is not int or type(high) is not int:
            raise ValueError("low and high must be int!")

        if log:
            return trial.suggest_int(name, low, high, log=True)
        else:
            return trial.suggest_int(name, low, high, step=step, log=False)

    if suggest_type == "float":
        if type(low) is not float or type(high) is not float:
            raise ValueError("low and high must be float!")

        if log:
            return trial.suggest_float(name, low, high, log=True)
        else:
            return trial.suggest_float(name, low, high, step=step, log=False)

    elif suggest_type == "uniform":
        if type(low) is not float or type(high) is not float:
            raise ValueError("low and high must be float!")

        return trial.suggest_uniform(name, low, high)

    elif suggest_type == "loguniform":
        if type(low) is not float or type(high) is not float:
            raise ValueError("low and high must be float!")

        return trial.suggest_float(name, low, high, log=True)

    else:
        raise ValueError("Invalid suggest_type!")


def _set_config_keys_value(dct: DictConfig, key_string: str, value: Any) -> DictConfig:
    """Given `foo`, 'key1.key2.key3', 'something', set foo['key1']['key2']['key3'] = 'something'.

    Parameters
    ----------
    dct : dict
        The dictionary to modify.
    key_string : str
        The key string to set.
    value : Any
        The value to set.

    See: https://stackoverflow.com/questions/57560308/python-set-dictionary-nested-key-with-dot-delineated-string

    """
    # start off pointing at the original dictionary that was passed in
    here = dct

    # turn the string of key names into a list of strings
    keys = key_string.split(".")

    # for every key *before* the last one, we concentrate on navigating through the dictionary
    for key in keys[:-1]:
        # try to find here[key]. If it doesn't exist, create it with an empty dictionary
        # update our `here` pointer to refer to the thing we just found (or created)
        here = here.setdefault(key, {})

    # finally, set the final key to the given value
    here[keys[-1]] = value

    return dct


def suggest_to_config(config: DictConfig, trial: optuna.trial.Trial) -> DictConfig:
    params = config.tuning_config.hyperparameters

    with open_dict(config):
        for param_key, param_dct in params.items():
            param_name = param_key.split(".")[-1]
            suggest = get_optuna_suggest(trial, param_name, **param_dct)
            config = _set_config_keys_value(config, param_key, suggest)

    return config
