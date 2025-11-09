import logging
import os

import hydra
import numpy as np
from f9columnar.ml.hdf5_writer import ArraysHdf5Writer
from f9columnar.utils.loggers import get_progress, setup_logger
from omegaconf import DictConfig

from seesaw.fakes.toys.distributions import WeightedToyDistributions


def make_dataset(
    N_loose_data: int,
    N_tight_data: int,
    N_loose_mc: int,
    N_tight_mc: int,
    pt_min: float,
    pt_max: float,
    dist_params: DictConfig,
) -> tuple[dict[str, dict[str, np.ndarray]], np.dtype]:
    """Make toy dataset with loose and tight labels for data and mc samples."""

    N_data = N_loose_data + N_tight_data
    N_mc = N_loose_mc + N_tight_mc

    logging.info(f"Making toy dataset with {N_data + N_mc} samples.")

    data_params, mc_params = dist_params.data, dist_params.mc

    # sample data and mc
    data_dist = WeightedToyDistributions(data_params.dist_name, data_params.weight_name)
    data_dist_params, data_weight_params = data_params.dist_params, data_params.weight_params

    if data_dist_params is None:
        data_dist_params = {}

    if data_weight_params is None:
        data_weight_params = {}

    data_dist.calculate_samples(x_min=pt_min, x_max=pt_max, n_samples=N_data, **data_dist_params)
    data_dist.calculate_weights(**data_weight_params)

    mc_dist = WeightedToyDistributions(mc_params.dist_name, mc_params.weight_name)
    mc_dist_params, mc_weight_params = mc_params.dist_params, mc_params.weight_params

    if mc_dist_params is None:
        mc_dist_params = {}

    if mc_weight_params is None:
        mc_weight_params = {}

    mc_dist.calculate_samples(x_min=pt_min, x_max=pt_max, n_samples=N_mc, **mc_dist_params)
    mc_dist.calculate_weights(**mc_weight_params)

    # split data and mc into loose and tight
    data_loose, data_loose_w = data_dist.sampled_dist[:N_loose_data], data_dist.weights[:N_loose_data]
    data_tight, data_tight_w = data_dist.sampled_dist[N_loose_data:], data_dist.weights[N_loose_data:]

    mc_loose, mc_loose_w = mc_dist.sampled_dist[:N_loose_mc], mc_dist.weights[:N_loose_mc]
    mc_tight, mc_tight_w = mc_dist.sampled_dist[N_loose_mc:], mc_dist.weights[N_loose_mc:]

    dataset_dtypes = np.dtype(
        {
            "names": ["toy_pt", "weights", "data_type", "toy_type"],
            "formats": ["float32", "float32", "int64", "int64"],
        }
    )
    # TNAnalysis convention: 2 for loose, 0 for tight
    dataset_dct = {
        "loose_data": {
            "toy_pt": data_loose,
            "weights": data_loose_w,
            "data_type": np.ones(N_loose_data, dtype=np.int64),
            "toy_type": np.full(N_loose_data, 2, dtype=np.int64),
        },
        "tight_data": {
            "toy_pt": data_tight,
            "weights": data_tight_w,
            "data_type": np.ones(N_tight_data, dtype=np.int64),
            "toy_type": np.full(N_tight_data, 0, dtype=np.int64),
        },
        "loose_mc": {
            "toy_pt": mc_loose,
            "weights": mc_loose_w,
            "data_type": np.zeros(N_loose_mc, dtype=np.int64),
            "toy_type": np.full(N_loose_mc, 2, dtype=np.int64),
        },
        "tight_mc": {
            "toy_pt": mc_tight,
            "weights": mc_tight_w,
            "data_type": np.zeros(N_tight_mc, dtype=np.int64),
            "toy_type": np.full(N_tight_mc, 0, dtype=np.int64),
        },
    }

    logging.info(f"Loose data: {len(data_loose)}, Tight data: {len(data_tight)}")
    logging.info(f"Loose MC: {len(mc_loose)}, Tight MC: {len(mc_tight)}")

    return dataset_dct, dataset_dtypes


def write_dataset_to_hdf5(
    dataset_dct: dict[str, dict[str, np.ndarray]],
    dataset_dtypes: np.dtype,
    counts_dct: dict[str, int],
    save_path: str,
    n_piles: int = 128,
    chunk_shape: int = 1024,
    output_file: str = "toy_dataset.hdf5",
    merge_piles: bool = False,
) -> None:
    """Write the dataset to an HDF5 file."""
    logging.info(f"Writing toy dataset to {save_path}.")

    dataset_names = [f"events/p{p}" for p in range(n_piles)]

    writers = []
    if merge_piles:
        writers.append(ArraysHdf5Writer(os.path.join(save_path, output_file)))
    else:
        for i in range(n_piles):
            writers.append(ArraysHdf5Writer(os.path.join(save_path, f"p{i}.hdf5")))

    if merge_piles:
        writers[0].create_datasets(
            dataset_names=dataset_names,
            mode="w",
            shape=(0,),
            maxshape=(None,),
            dtype=dataset_dtypes,
        )
    else:
        for writer in writers:
            writer.create_datasets(
                dataset_names=["events"],
                mode="w",
                shape=(0,),
                maxshape=(None,),
                dtype=dataset_dtypes,
            )

    dataset_keys = list(dataset_dct.keys())
    iter_done = {name: False for name in dataset_keys}
    current_counts_dct = {name: 0 for name in dataset_keys}

    iter_pile_counts: dict[int, int] = {p: 0 for p in range(n_piles)}

    progress = get_progress()
    progress.start()
    bar = progress.add_task("Saving toy events", total=sum(counts_dct.values()))

    while True:
        for dataset_key in dataset_keys:
            if iter_done[dataset_key]:
                continue

            total_counts = counts_dct[dataset_key]
            written_counts = current_counts_dct[dataset_key]

            if written_counts == total_counts:
                iter_done[dataset_key] = True
                continue

            rng_idx = np.random.randint(0, n_piles)

            current_pile_counts = iter_pile_counts[rng_idx]

            remaining = total_counts - written_counts
            next_current_pile_counts = current_pile_counts + min(chunk_shape, remaining)

            start_write = written_counts
            end_write = written_counts + min(chunk_shape, remaining)
            delta_write_counts = end_write - start_write

            data_dct = dataset_dct[dataset_key]
            w_data_dct = {key: data_dct[key][start_write:end_write] for key in data_dct.keys()}

            if merge_piles:
                writers[0].add_data(
                    data_dct=w_data_dct,
                    dataset_name=dataset_names[rng_idx],
                    idx=(current_pile_counts, next_current_pile_counts),
                    resize=(next_current_pile_counts,),
                )
            else:
                writers[rng_idx].add_data(
                    data_dct=w_data_dct,
                    dataset_name="events",
                    idx=(current_pile_counts, next_current_pile_counts),
                    resize=(next_current_pile_counts,),
                )

            iter_pile_counts[rng_idx] = next_current_pile_counts
            current_counts_dct[dataset_key] += delta_write_counts

            progress.update(bar, advance=delta_write_counts)

        if all(iter_done.values()):
            progress.stop()
            break

    logging.info("Finished writing toy dataset to HDF5 format.")

    total_closed = 0
    for w in writers:
        is_closed = w.close_write_handle()
        if is_closed:
            total_closed += 1

    logging.info(f"Closed {total_closed}/{len(writers)} HDF5 writers.")


@hydra.main(
    config_path=os.path.join(os.environ["ANALYSIS_ML_CONFIG_DIR"], "fakes"),
    config_name="convert_config",
    version_base=None,
)
def main(config: DictConfig) -> None:
    setup_logger(config.min_logging_level)

    toy_dataset_config = config.hdf5_config
    dist_params = toy_dataset_config.distributions

    save_path = toy_dataset_config.save_path

    if save_path is None:
        save_path = os.environ["ANALYSIS_ML_DATA_DIR"]

    os.makedirs(save_path, exist_ok=True)

    N_loose_data, N_tight_data, N_loose_mc, N_tight_mc = (
        int(toy_dataset_config.dataset_composition.loose_data),
        int(toy_dataset_config.dataset_composition.tight_data),
        int(toy_dataset_config.dataset_composition.loose_mc),
        int(toy_dataset_config.dataset_composition.tight_mc),
    )
    pt_min, pt_max = dist_params.pt_min, dist_params.pt_max

    mc_data_ratio = (N_loose_mc + N_tight_mc) / (N_loose_data + N_tight_data)
    tight_loose_ratio = (N_tight_data + N_tight_mc) / (N_loose_data + N_loose_mc)

    mc_prob = (N_loose_mc + N_tight_mc) / (N_loose_data + N_tight_data + N_loose_mc + N_tight_mc)
    tight_prob = (N_tight_data + N_tight_mc) / (N_loose_data + N_tight_data + N_loose_mc + N_tight_mc)

    logging.info(f"MC/Data ratio: {mc_data_ratio:.3f}, tight/loose ratio: {tight_loose_ratio:.3f}.")
    logging.info(f"MC probability: {mc_prob:.3f}, tight probability: {tight_prob:.3f}.")

    counts_dct = {
        "loose_data": N_loose_data,
        "tight_data": N_tight_data,
        "loose_mc": N_loose_mc,
        "tight_mc": N_tight_mc,
    }
    convert_config = config.convert_config

    dataset_dct, dataset_dtypes = make_dataset(
        N_loose_data,
        N_tight_data,
        N_loose_mc,
        N_tight_mc,
        pt_min,
        pt_max,
        dist_params,
    )
    write_dataset_to_hdf5(
        dataset_dct,
        dataset_dtypes,
        counts_dct,
        save_path,
        n_piles=convert_config.n_piles,
        chunk_shape=convert_config.chunk_shape,
        output_file=convert_config.output_file,
        merge_piles=convert_config.merge_piles,
    )


if __name__ == "__main__":
    main()
