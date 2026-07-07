"""ppb_model: reusable library for human plasma protein binding (PPB) prediction on PPBR_AZ.

Modules are added stage by stage:
    utils           - seeds, logging, config loading, project paths
    data            - loading and filtering the raw PPBR_AZ dataset
    standardisation - RDKit canonicalisation, salt stripping, charge handling
    targets         - PPB target transformations (percent/fraction/logit/lnKa/...)
    features        - descriptors, Morgan/MACCS fingerprints, hybrid representations
    splitting       - Bemis-Murcko scaffold split and random split
    baselines       - conventional baseline models
    models          - proposed improved model
    tuning          - hyperparameter optimisation
    evaluation      - metrics and per-band evaluation
    uncertainty     - applicability domain and prediction intervals
    interpretation  - permutation / SHAP-style importance
"""

__version__ = "0.1.0"
