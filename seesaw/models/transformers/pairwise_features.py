from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from f9columnar.ml.dataloader_helpers import ColumnSelection
from omegaconf import DictConfig, ListConfig
from torch import nn

from seesaw.models.activations import get_activation
from seesaw.models.masked_batchnorm import MaskedBatchNorm1d

DEFAULT_PARTICLE_MASSES = {
    "el": 0.000511,
    "mu": 0.105658,
    "tau": 1.77686,
    "jet": None,
}

OBJECTS_SHORT_NAMES = {
    "electrons": "el",
    "muons": "mu",
    "taus": "tau",
    "jets": "jet",
}


@dataclass(frozen=True)
class ParticleFeatureSpec:
    object_name: str
    pt_index: int
    eta_index: int
    phi_index: int
    energy_index: int | None = None
    rest_mass: float | None = None

    def requires_energy_derivation(self) -> bool:
        return self.energy_index is None


@dataclass(frozen=True)
class ParticleAttentionConfig:
    specs: list[ParticleFeatureSpec]
    participating_objects: list[str]
    embed_dims: tuple[int, ...]
    quantities: list[str]

    @property
    def feature_dim(self) -> int:
        return 4

    def objects(self) -> Iterable[str]:
        return (spec.object_name for spec in self.specs)

    def as_dict(self) -> dict[str, ParticleFeatureSpec]:
        return {spec.object_name: spec for spec in self.specs}

    def is_participating(self, object_name: str) -> bool:
        """Check if an object participates in pairwise attention."""
        return object_name in self.participating_objects


def _build_particle_feature_spec(
    dataset_name: str, dataset_selection: ColumnSelection, rest_mass: float | None = None
) -> ParticleFeatureSpec:
    if len(dataset_selection.numer_columns) == 0:
        raise ValueError(f"No numerical columns available for '{dataset_name}' to enable particle attention.")

    columns = list(dataset_selection.offset_used_columns)

    short_name = OBJECTS_SHORT_NAMES[dataset_name]

    vars_set = {"pt", "eta", "phi", "e"}
    available_vars: dict[str, int] = {}  # variable name -> column index

    for i, c in enumerate(columns):
        var_name = c.split(f"{short_name}_")[-1]
        if var_name in vars_set:
            available_vars[var_name] = i

    non_optional = {"pt", "eta", "phi"}
    for var in non_optional:
        if var not in available_vars:
            raise ValueError(f"Missing required column for '{dataset_name}': {short_name}_{var}.")

    if rest_mass is None:
        rest_mass = DEFAULT_PARTICLE_MASSES[short_name]
    else:
        rest_mass = float(rest_mass)

    if "e" not in available_vars and rest_mass is None:
        raise ValueError(f"No energy column found for '{dataset_name}' and no rest mass specified.")

    return ParticleFeatureSpec(
        object_name=dataset_name,
        pt_index=available_vars["pt"],
        eta_index=available_vars["eta"],
        phi_index=available_vars["phi"],
        energy_index=available_vars.get("e", None),
        rest_mass=rest_mass,
    )


def build_particle_attention_config(
    selection: ColumnSelection,
    particle_attention_cfg: DictConfig | None,
    num_heads: int,
) -> ParticleAttentionConfig | None:
    """Build particle attention config from architecture config.

    particle_attention:
        objects: all / list of object names / list of short names / one object name / one short name
        embedding_layers: 2 / list of ints
        embedding_dim: 64 / null
    """
    if particle_attention_cfg is None:
        return None

    objects = particle_attention_cfg.get("objects", None)
    if objects is None:
        return None

    available_objects = [name for name in selection.keys() if name != "events"]
    available_objects_aliases = [OBJECTS_SHORT_NAMES[name] for name in available_objects]

    if objects == "all":
        participating_objects = available_objects
    elif isinstance(objects, str):
        if objects in available_objects:
            participating_objects = [objects]
        elif objects in available_objects_aliases:
            idx = available_objects_aliases.index(objects)
            participating_objects = [available_objects[idx]]
        else:
            raise ValueError("Invalid objects value in particle_attention config.")
    elif isinstance(objects, ListConfig):
        participating_objects = []
        for obj in objects:
            if obj in available_objects:
                participating_objects.append(obj)
            elif obj in available_objects_aliases:
                idx = available_objects_aliases.index(obj)
                participating_objects.append(available_objects[idx])
            else:
                raise ValueError("Invalid objects value in particle_attention config.")
    else:
        raise ValueError("Invalid objects value in particle_attention config.")

    embedding_dim = particle_attention_cfg.get("embedding_dim", 48)
    embedding_layers = particle_attention_cfg.get("embedding_layers", 2)

    if isinstance(embedding_layers, ListConfig) and embedding_dim is None:
        embed_dims_tuple = tuple([int(dim) for dim in embedding_layers] + [num_heads])
    else:
        embed_dims_tuple = (*[embedding_dim] * embedding_layers, num_heads)

    specs: list[ParticleFeatureSpec] = []
    for dataset_name in available_objects:
        spec = _build_particle_feature_spec(
            dataset_name, selection[dataset_name], particle_attention_cfg.get("rest_mass", None)
        )
        specs.append(spec)

    quantities = particle_attention_cfg.get("quantities", ["delta_r", "kt", "z", "m2"])

    return ParticleAttentionConfig(
        specs=specs,
        participating_objects=participating_objects,
        embed_dims=embed_dims_tuple,
        quantities=quantities,
    )


def derive_energy_from_mass(
    pt: torch.Tensor,
    eta: torch.Tensor,
    rest_mass: float,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute energy from transverse momentum, pseudorapidity and rest mass."""
    if rest_mass < 0.0:
        raise ValueError("rest_mass must be non-negative when deriving energy.")

    mass = pt.new_tensor(rest_mass)
    cosh_eta = torch.cosh(eta)
    energy_sq = (pt * cosh_eta) ** 2 + mass**2
    return torch.sqrt(torch.clamp(energy_sq, min=epsilon))


class PairwiseFeaturesCalculator(nn.Module):
    def __init__(self, epsilon: float = 1e-6, use_log: bool = True, quantities: list[str] | None = None) -> None:
        """Pairwise features calculator.

        Parameters
        ----------
        epsilon : float, optional
            Small value for numerical stability, by default 1e-6.
        use_log : bool, optional
            Whether to apply log(1 + x) transformation to features, by default True.
        quantities : list[str] | None, optional
            List of quantities to compute. If None, computes all available quantities. Available quantities: "delta_r",
            "kt", "z", "m2". If specified, must be a subset of these, by default None. If None, all quantities are computed.

        """
        super().__init__()
        self.epsilon = epsilon
        self.use_log = use_log

        available_quantities = {
            "delta_r": False,
            "kt": False,
            "z": False,
            "m2": False,
        }

        if quantities is None:
            for key in available_quantities.keys():
                available_quantities[key] = True
        else:
            for key in quantities:
                if key not in available_quantities:
                    raise ValueError(f"Invalid quantity for pairwise features: {key}")
                available_quantities[key] = True

        for q, v in available_quantities.items():
            setattr(self, f"use_{q}", v)

    def _log_clamp(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=self.epsilon))

    def _log1p_clamp(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.clamp(x, min=self.epsilon))

    def _calculate_pz(self, pt: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        return pt * torch.sinh(eta)

    def _calculate_rapidity(self, pz: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        e_plus_pz = torch.clamp(energy + pz, min=self.epsilon)
        e_minus_pz = torch.clamp(energy - pz, min=self.epsilon)

        rapidity = 0.5 * self._log_clamp(e_plus_pz / e_minus_pz)
        return rapidity

    @staticmethod
    def _wrap_delta_phi(delta_phi: torch.Tensor) -> torch.Tensor:
        """Wrap phi into [-pi, pi]."""
        return (delta_phi + torch.pi) % (2 * torch.pi) - torch.pi

    def _calculate_delta_r_ij(
        self,
        phi: torch.Tensor,
        rapidity: torch.Tensor,
        i: torch.Tensor,
        j: torch.Tensor,
    ) -> torch.Tensor:
        phi_i, phi_j = phi[:, i], phi[:, j]
        rap_i, rap_j = rapidity[:, i], rapidity[:, j]

        delta_rap = rap_i - rap_j
        delta_phi = self._wrap_delta_phi(phi_i - phi_j)

        delta_r = torch.sqrt(delta_rap.square() + delta_phi.square())

        if self.use_log:
            delta_r = self._log1p_clamp(delta_r)

        return delta_r

    def _calculate_kt_ij(
        self,
        pt: torch.Tensor,
        delta_r_ij: torch.Tensor,
        i: torch.Tensor,
        j: torch.Tensor,
    ) -> torch.Tensor:
        pt_i, pt_j = pt[:, i], pt[:, j]

        min_pt = torch.minimum(pt_i, pt_j)

        kt = min_pt * delta_r_ij

        if self.use_log:
            kt = self._log1p_clamp(kt)

        return kt

    def _calculate_z_ij(
        self,
        pt: torch.Tensor,
        i: torch.Tensor,
        j: torch.Tensor,
    ) -> torch.Tensor:
        pt_i, pt_j = pt[:, i], pt[:, j]

        min_pt = torch.minimum(pt_i, pt_j)

        z = min_pt / (pt_i + pt_j + self.epsilon)

        if self.use_log:
            z = self._log1p_clamp(z)

        return z

    def _calculate_invariant_mass_sq_ij(
        self,
        pt: torch.Tensor,
        phi: torch.Tensor,
        pz: torch.Tensor,
        energy: torch.Tensor,
        i: torch.Tensor,
        j: torch.Tensor,
    ) -> torch.Tensor:
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)

        px_i, px_j = px[:, i], px[:, j]
        py_i, py_j = py[:, i], py[:, j]
        pz_i, pz_j = pz[:, i], pz[:, j]
        energy_i, energy_j = energy[:, i], energy[:, j]

        sum_px = px_i + px_j
        sum_py = py_i + py_j
        sum_pz = pz_i + pz_j
        sum_energy = energy_i + energy_j

        invariant_mass_sq = sum_energy.square() - sum_px.square() - sum_py.square() - sum_pz.square()

        if self.use_log:
            invariant_mass_sq = self._log1p_clamp(invariant_mass_sq)

        return invariant_mass_sq

    @torch.no_grad()
    def forward(
        self,
        pt: torch.Tensor,
        eta: torch.Tensor,
        phi: torch.Tensor,
        energy: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ParT-style pairwise features with shape (B, N, N, C).

        Uses rapidity (computed from E and pz) instead of pseudorapidity for angular features. Only off-diagonal pairs
        (lower triangle) are computed and mirrored to upper triangle. Diagonal is zero.

        """
        b, n = pt.shape

        # Get lower triangular indices (offset=-1 excludes diagonal)
        i, j = torch.tril_indices(n, n, offset=-1, device=pt.device)

        mask_i, mask_j = mask[:, i], mask[:, j]

        calculated: list[torch.Tensor] = []

        if self.use_delta_r or self.use_kt:
            pz = self._calculate_pz(pt, eta)
            rapidity = self._calculate_rapidity(pz, energy)
            delta_r_ij = self._calculate_delta_r_ij(phi, rapidity, i, j)

            if self.use_delta_r:
                calculated.append(delta_r_ij)

        if self.use_kt:
            kt_ij = self._calculate_kt_ij(pt, delta_r_ij, i, j)
            calculated.append(kt_ij)

        if self.use_z:
            z_ij = self._calculate_z_ij(pt, i, j)
            calculated.append(z_ij)

        if self.use_m2:
            if not (self.use_delta_r or self.use_kt):
                pz = self._calculate_pz(pt, eta)

            invariant_mass_sq_ij = self._calculate_invariant_mass_sq_ij(pt, phi, pz, energy, i, j)
            calculated.append(invariant_mass_sq_ij)

        pair_features = torch.stack(calculated, dim=-1)  # (B, num_pairs, C)

        pair_mask = mask_i | mask_j  # True if padded and should be masked
        pair_features = pair_features.masked_fill(pair_mask.unsqueeze(-1), 0.0)

        # Build symmetric output matrix (B, N, N, C) with zero diagonal
        features = pt.new_zeros(b, n, n, len(calculated))
        features[:, i, j, :] = pair_features
        features[:, j, i, :] = pair_features  # Mirror to upper triangle

        return features, pair_mask


class PairwiseFeaturesEmbeddingModule(nn.Module):
    def __init__(
        self, input_dim: int, embed_dims: tuple[int, ...], activation: str = "GELU", input_bn: bool = True
    ) -> None:
        super().__init__()

        batch_norms: list[nn.Module] = [MaskedBatchNorm1d(input_dim) if input_bn else nn.Identity()]
        convs: list[nn.Module] = []
        acts: list[nn.Module] = []

        in_dim = input_dim
        for out_dim in embed_dims:
            convs.append(nn.Conv1d(in_dim, out_dim, kernel_size=1))
            batch_norms.append(MaskedBatchNorm1d(out_dim))
            acts.append(get_activation(activation))

            in_dim = out_dim

        self.batch_norms = nn.ModuleList(batch_norms)
        self.convs = nn.ModuleList(convs)
        self.acts = nn.ModuleList(acts)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            mask = ~mask
            mask = mask.to(dtype=x.dtype)

        x = self.batch_norms[0](x, mask)

        for i in range(len(self.convs)):
            x = self.convs[i](x)
            x = self.batch_norms[i + 1](x, mask)
            x = self.acts[i](x)

        return x


class PairwiseFeaturesEmbedder(nn.Module):
    def __init__(
        self, input_dim: int, embed_dims: tuple[int, ...], activation: str = "GELU", post_ln: bool = False
    ) -> None:
        """CNN-based embedding for pairwise features.

        Computes embeddings only for off-diagonal pairs (lower triangle) and mirrors to upper triangle for efficiency.

        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        embed_dims : tuple[int]
            Sequence of output dimensions for each embedding layer.
        activation : str, optional
            Activation function to use between layers, by default "GELU".
        post_ln : bool, optional
            Whether to apply LayerNorm after the final embedding layer, by default False.

        Note
        ----
        Forward embeds pairwise features and returns tensor shaped (B, output_dim, N, N). Only processes lower triangle
        pairs and mirrors to upper triangle. Diagonal remains zero.

        """
        super().__init__()
        if len(embed_dims) == 0:
            raise ValueError("embed_dims must contain at least one dimension.")

        self.embed = PairwiseFeaturesEmbeddingModule(input_dim, embed_dims, activation)

        self.output_dim = embed_dims[-1]

        self.post_ln: nn.LayerNorm | None

        if post_ln:
            self.post_ln = nn.LayerNorm(self.output_dim)
        else:
            self.post_ln = None

    def forward(self, pairwise_features: torch.Tensor, pairwise_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, _, _ = pairwise_features.shape

        # Get lower triangular indices (offset=-1 excludes diagonal)
        i, j = torch.tril_indices(n, n, offset=-1, device=pairwise_features.device)

        # Extract lower triangle features: (B, num_pairs, C)
        lower_tri_features = pairwise_features[:, i, j, :]

        # Reshape for Conv1d: (B, C, num_pairs)
        x = lower_tri_features.permute(0, 2, 1)

        # Apply embedding network
        x = self.embed(x, pairwise_mask)

        # x shape: (B, output_dim, num_pairs) -> need (B, num_pairs, output_dim) for assignment
        embedded = x.permute(0, 2, 1)  # (B, num_pairs, output_dim)

        # Optional post LayerNorm
        if self.post_ln is not None:
            embedded = self.post_ln(embedded)

        # Build symmetric output matrix (B, output_dim, N, N) with zero diagonal
        output = embedded.new_zeros(b, self.output_dim, n, n)

        # Assign to lower and upper triangle (mirror)
        output[:, :, i, j] = embedded.permute(0, 2, 1)
        output[:, :, j, i] = embedded.permute(0, 2, 1)

        return output
