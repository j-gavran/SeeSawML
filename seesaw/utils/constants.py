# Maps object collection names to their field name prefixes
PARTICLE_PREFIX_MAP = {
    "jets": "jet",
    "electrons": "el",
    "muons": "mu",
}


def add_type_fields_to_features(
    features: list[str],
    valid_type_values: dict[str, list[int]] | None,
) -> list[str]:
    """Auto-add type fields to features list if valid_type_values is configured.

    Parameters
    ----------
    features : list[str]
        List of feature names to potentially extend.
    valid_type_values : dict[str, list[int]] | None
        Mapping from object name (e.g., "jets") to valid type values.
        If None or empty, returns features unchanged.

    Returns
    -------
    list[str]
        Features list with type fields added if needed.
    """
    import logging

    if not valid_type_values:
        return features

    added_fields = []
    for obj_name in valid_type_values:
        prefix = PARTICLE_PREFIX_MAP.get(obj_name)
        if prefix:
            type_field = f"{prefix}_type"
            if type_field not in features:
                features.append(type_field)
                added_fields.append(type_field)

    if added_fields:
        logging.info(f"[green]Auto-added type fields for masking: {added_fields}[/green]")

    return features
