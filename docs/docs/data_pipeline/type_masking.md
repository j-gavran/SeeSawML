# Type-Based Particle Masking

Type-based masking allows filtering particles (jets, electrons, muons) based on their reconstruction quality during training. This is useful when your dataset contains particles of varying quality levels (e.g., Loose, Medium, Tight) and you want the model to only learn from high-quality particles.

## Overview

There are two strategies for handling particle types:

| Strategy | When Applied | Effect | Use Case |
|----------|--------------|--------|----------|
| `remove_invalid: true` | HDF5 conversion | Physically removes invalid particles from dataset | Smaller files, simpler training |
| `remove_invalid: false` | Training time | Keeps all particles but masks invalid ones in attention | Flexibility to change criteria later |

Both strategies use the `valid_type_values` configuration to define which particle types are considered valid.

## Configuration

### HDF5 Conversion Config

In `hdf5_config/*.yaml`, configure type fields for each particle collection:

```yaml
jagged:
  jets:
    output:
      - jet_e
      - jet_pt
      - jet_eta
      - jet_phi
      - jet_ftag_quantile
    extra_input:
      - jet_type  # Type field for masking
    max_length: 15
    pad_value: 999.0
    valid_type_values: [2]  # Only Tight jets (type=2) are valid
    remove_invalid: false   # Keep in HDF5, mask during training

  electrons:
    output:
      - el_charge
      - el_pt
      - el_eta
      - el_phi
    extra_input:
      - el_type
    max_length: 2
    pad_value: 999.0
    valid_type_values: [2]
    remove_invalid: false

  muons:
    output:
      - mu_charge
      - mu_pt
      - mu_eta
      - mu_phi
    extra_input:
      - mu_type
    max_length: 2
    pad_value: 999.0
    valid_type_values: [2]
    remove_invalid: false
```

### Model Config

In `model_config/*.yaml`, specify which type values are valid for each particle collection:

```yaml
architecture_config:
  model: ParticleTransformer  # or SetTransformer, DeepSets

  valid_type_values:
    jets: [2]       # Tight jets
    electrons: [2]  # Tight electrons
    muons: [2]      # Tight muons

  # ... other model settings
```

!!! Note "Type Value Conventions"
    Common type values in ATLAS:

    - `0`: No ID
    - `1`: Loose
    - `2`: Tight

    Check your ntuple documentation for exact definitions.

## How It Works

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HDF5 Conversion                               │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────────┐   │
│  │ ROOT NTuple │───>│ TypeFieldProcessor│───>│ HDF5 Dataset      │   │
│  │ (all types) │    │                  │    │                   │   │
│  └─────────────┘    │ remove_invalid:  │    │ jets: (N, 15, 6)  │   │
│                     │   true  -> filter│    │ electrons: (N,2,5)│   │
│                     │   false -> keep  │    │ muons: (N, 2, 5)  │   │
│                     └──────────────────┘    └───────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Training                                     │
│  ┌───────────────┐    ┌──────────────────┐    ┌─────────────────┐   │
│  │ DataLoader    │───>│ Batch            │───>│ Model           │   │
│  │               │    │                  │    │                 │   │
│  │ Loads:        │    │ X: features      │    │ JaggedPreproc.  │   │
│  │ - features    │    │ type_tensor: aux │    │ derives mask    │   │
│  │ - type fields │    │                  │    │ from type_tensor│   │
│  └───────────────┘    └──────────────────┘    └─────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Batch Structure

The dataloader returns batches with the following structure for each particle collection:

```python
batch[0][object_name] = (
    X,           # index 0: Feature tensor (batch, max_objects, n_features)
    y,           # index 1: Labels (None for particles)
    w,           # index 2: Weights (None for particles)
    type_tensor, # index 3: Type values (batch, max_objects) - y_aux field
)
```

The `type_tensor` (stored in `y_aux`) contains the particle type values for each particle position.

### Mask Derivation

During the forward pass, `JaggedPreprocessor` derives attention masks from type tensors:

```python
# In JaggedPreprocessor.forward()
if type_tensors is not None and type_tensors[i] is not None:
    # Compare each particle's type against valid types
    # valid_types = [2.0] for Tight particles
    type_matches = (type_tensor.unsqueeze(-1) == valid_types).any(dim=-1)
    mask = ~type_matches  # True = masked (invalid), False = valid
```

The resulting boolean mask is used in transformer attention:

- `True` = particle is **masked** (invalid type, excluded from attention)
- `False` = particle is **valid** (included in attention computation)

!!! Warning "Attention Mask Convention"
    PyTorch attention uses `True = masked out`. This is the opposite of some other frameworks where `True = attend to`.

### Logging Output

When training starts, you'll see masking statistics for the first batch:

```
TYPE-BASED MASKING ENABLED
  electrons: valid_values=[2]
  jets: valid_values=[2]
  muons: valid_values=[2]

[electrons] Type masking: 4557/8192 (55.6%) masked, valid_types=[2.0]
[jets] Type masking: 39198/61440 (63.8%) masked, valid_types=[2.0]
[muons] Type masking: 3635/8192 (44.4%) masked, valid_types=[2.0]
```

This shows what percentage of particles are being masked out due to failing the type requirement.

## Choosing a Strategy

### Use `remove_invalid: true` when:

- You're certain about your quality requirements
- You want smaller HDF5 files
- You won't need to experiment with different type cuts

### Use `remove_invalid: false` when:

- You want to experiment with different type selections
- You need to compare Loose vs Tight performance
- You want to keep the option to use all particles later

!!! Tip "Recommended Approach"
    Start with `remove_invalid: false` during development. Once you've finalized your type selection, you can regenerate HDF5 files with `remove_invalid: true` for production to save disk space.

## Implementation Details

### Key Files

| File | Purpose |
|------|---------|
| `seesaw/signal/dataset/hdf5_converter.py` | `TypeFieldProcessor` handles `remove_invalid` during HDF5 conversion |
| `seesaw/models/jagged_preprocessor.py` | `JaggedPreprocessor` derives masks from type tensors during training |
| `seesaw/models/constants.py` | `PARTICLE_PREFIX_MAP` maps object names to field prefixes |
| `seesaw/signal/models/sig_bkg_classifiers.py` | Extracts `type_tensors` from batch and passes to model |

### Auto-Added Type Fields

When `valid_type_values` is configured, the type fields (`jet_type`, `el_type`, `mu_type`) are automatically added to the features list if not already present. This ensures the dataloader loads them from HDF5.

```python
# In seesaw/models/constants.py
PARTICLE_PREFIX_MAP = {
    "jets": "jet",
    "electrons": "el",
    "muons": "mu",
}

def add_type_fields_to_features(features, valid_type_values):
    """Auto-add type fields to features list if valid_type_values is configured."""
    for obj_name in valid_type_values:
        prefix = PARTICLE_PREFIX_MAP.get(obj_name)
        if prefix:
            type_field = f"{prefix}_type"
            if type_field not in features:
                features.append(type_field)
    return features
```
