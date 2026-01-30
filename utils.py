def extract_smiles_and_labels(dc_dataset):
    """
    Converts DeepChem dataset to raw components.
    """
    smiles = list(dc_dataset.ids)
    y = dc_dataset.y
    w = dc_dataset.w
    return smiles, y, w
