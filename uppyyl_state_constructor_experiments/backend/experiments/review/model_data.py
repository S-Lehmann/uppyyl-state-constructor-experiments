"""This module contains the models data for the state construction experiments."""
from uppyyl_state_constructor_experiments.definitions import RES_DIR

all_model_data = [
    # Supported Uppaal demo models
    {"path": RES_DIR.joinpath("uppaal_demo_models/2doors.xml")},
    {"path": RES_DIR.joinpath("uppaal_demo_models/bridge.xml")},
    {"path": RES_DIR.joinpath("uppaal_demo_models/fischer.xml")},
    {"path": RES_DIR.joinpath("uppaal_demo_models/fischer-symmetry.xml")},
    {"path": RES_DIR.joinpath("uppaal_demo_models/train-gate.xml")},
    {"path": RES_DIR.joinpath("uppaal_demo_models/train-gate-orig.xml")},

    # Additional case study models
    {"path": RES_DIR.joinpath("uppaal_demo_models/case-study/csmacd2.xml")},
    {"path": RES_DIR.joinpath("uppaal_demo_models/case-study/tdma.xml")},
]
