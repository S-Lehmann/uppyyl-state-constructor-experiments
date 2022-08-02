"""This module contains the introduction model example."""

import os

from uppyyl_simulator.backend.ast.parsers.uppaal_xml_model_parser import uppaal_system_to_xml
from uppyyl_simulator.backend.data_structures.dbm.dbm import DBM
from uppyyl_simulator.backend.models.ta.nta_modifier import SystemModifier
from uppyyl_simulator.backend.simulator.simulator import Simulator
from uppyyl_state_constructor.backend.dbm_constructor.oc.dbm_constructor_oc import DBMReconstructorOC, \
    ApproximationStrategy, ConstraintStrategy
from uppyyl_state_constructor.backend.dbm_constructor.rinast.dbm_constructor_rinast import DBMReconstructorRinast
from uppyyl_state_constructor.backend.dbm_constructor.trivial.dbm_constructor_trivial import DBMReconstructorTrivial
from uppyyl_state_constructor.backend.model_adaptor.model_adaptor import StateConstructionModelAdaptor
from uppyyl_state_constructor_experiments.definitions import RES_DIR

model_base_folder = RES_DIR.joinpath(f'example_models')
introduction_model_path = os.path.join(model_base_folder, f'introductory_example.xml')
adapted_introduction_model_path = os.path.join(model_base_folder, f'introductory_example[adapted].xml')

introduction_model_trivial_csc_path = os.path.join(model_base_folder, f'introductory_example[trivial].xml')
introduction_model_rinast_csc_path = os.path.join(model_base_folder, f'introductory_example[rinast].xml')
introduction_model_oc_csc_path = os.path.join(model_base_folder, f'introductory_example[oc].xml')


def prepare_adapted_model():
    """Prepares the adapted model by converting all instances to separate templates."""
    simulator = Simulator()
    simulator.load_system(introduction_model_path)
    system = simulator.system

    instance_data = simulator.system_state.instance_data
    SystemModifier.convert_instances_to_templates(
        system=system, instance_data=instance_data, keep_original_templates=False)

    output_model_xml = uppaal_system_to_xml(system=system)
    with open(adapted_introduction_model_path, "w") as file:
        file.write(output_model_xml)


def get_reference_data():
    """Simulates reference data with the given model.

    Returns:
        The reference data.
    """
    simulator = Simulator()
    simulator.load_system(adapted_introduction_model_path)

    # Simulate 10 times over "Execute", then "Off", and another 10 times "Execute"
    next_transition = None
    for i in range(0, 10):
        # Execute transition "On" -> "Execute"
        for transition in simulator.transitions:
            if transition.target_state.location_state["Ex"].name == "Execute":
                next_transition = transition
                break
        simulator.execute_transition(next_transition)

        # Execute transition "Execute" -> "On"
        for transition in simulator.transitions:
            if transition.target_state.location_state["Ex"].name == "On":
                next_transition = transition
                break
        simulator.execute_transition(next_transition)

    # Execute transition "On" -> "Off"
    for transition in simulator.transitions:
        if transition.target_state.location_state["Ex"].name == "Off":
            next_transition = transition
            break
    simulator.execute_transition(next_transition)

    # Execute transition "Off" -> "On"
    for transition in simulator.transitions:
        if transition.target_state.location_state["Ex"].name == "On":
            next_transition = transition
            break
    simulator.execute_transition(next_transition)

    for i in range(0, 10):
        # Execute transition "On" -> "Execute"
        for transition in simulator.transitions:
            if transition.target_state.location_state["Ex"].name == "Execute":
                next_transition = transition
                break
        simulator.execute_transition(next_transition)

        # Execute transition "Execute" -> "On"
        for transition in simulator.transitions:
            if transition.target_state.location_state["Ex"].name == "On":
                next_transition = transition
                break
        simulator.execute_transition(next_transition)

    # Get resulting sequence and DBM
    seq = simulator.get_sequence()
    dbm = simulator.system_state.dbm_state.copy()
    print(f'\n== Original DBM operation sequence (len: {len(seq)}) ==')
    print(seq)

    instance_data = simulator.system_state.instance_data
    loc_state = {inst_name: loc.id for inst_name, loc in simulator.system_state.location_state.items()}
    var_state = simulator.system_state.get_compact_variable_state()

    return {"seq": seq, "dbm": dbm, "instance_data": instance_data, "loc_state": loc_state, "var_state": var_state}


def experiment_introduction_example():
    """Executes the experiments for the introduction model example."""
    prepare_adapted_model()
    reference_data = get_reference_data()

    # Execute the three approaches "Trivial", "Rinast", and "OC"
    experiment_introduction_example_trivial(reference_data=reference_data)
    experiment_introduction_example_rinast(reference_data=reference_data)
    experiment_introduction_example_oc(reference_data=reference_data)


def experiment_introduction_example_trivial(reference_data):
    """Executes the introduction model experiment based on the Trivial approach.

    Args:
        reference_data: The reference data based on which the construction is performed.
    """
    simulator = Simulator()
    simulator.load_system(adapted_introduction_model_path)
    system = simulator.system

    # Generate initial DBM
    clocks_without_ref = simulator.system_state.dbm_state.clocks[1:]
    dbm_init = DBM(clocks=clocks_without_ref, zero_init=True)

    # Generate "Trivial" DBM construction sequence from simulated sequence
    dbm_constructor_trivial = DBMReconstructorTrivial()
    data_for_rec = {
        "dbm_init": dbm_init,
        "dbm_target": reference_data["dbm"],
        "seq_full": reference_data["seq"],
        "seq_incr": reference_data["seq"]
    }
    res_csc = dbm_constructor_trivial.time_measured_reconstruction(data_for_rec=data_for_rec)
    seq_csc = res_csc["seq_rec"]
    print(f'\n== DBM construction sequence (Trivial) (len: {len(seq_csc)}) ==')
    print(seq_csc)

    # Adapt the model for state construction
    StateConstructionModelAdaptor.add_state_construction_components(
        system=simulator.system,
        seq=seq_csc,
        instance_data=reference_data["instance_data"],
        loc_state=reference_data["loc_state"],
        var_state=reference_data["var_state"]
    )

    # Export the generated model
    output_model_xml = uppaal_system_to_xml(system=system)
    with open(introduction_model_trivial_csc_path, "w") as file:
        file.write(output_model_xml)


def experiment_introduction_example_rinast(reference_data):
    """Executes the introduction model experiment based on the Rinast approach.

    Args:
        reference_data: The reference data based on which the construction is performed.
    """
    simulator = Simulator()
    simulator.load_system(adapted_introduction_model_path)
    system = simulator.system

    # Generate initial DBM
    clocks_without_ref = simulator.system_state.dbm_state.clocks[1:]
    dbm_init = DBM(clocks=clocks_without_ref, zero_init=True)

    # Generate "Rinast" DBM construction sequence from simulated sequence
    dbm_constructor_rinast = DBMReconstructorRinast(clock_names=simulator.system_state.dbm_state.clocks, var_names=[])
    data_for_rec = {
        "dbm_init": dbm_init,
        "dbm_target": reference_data["dbm"],
        "seq_full": reference_data["seq"],
        "seq_incr": reference_data["seq"]
    }
    res_csc = dbm_constructor_rinast.time_measured_reconstruction(data_for_rec=data_for_rec)
    seq_csc = res_csc["seq_rec"]
    print(f'\n== DBM construction sequence (Rinast) (len: {len(seq_csc)}) ==')
    print(seq_csc)

    # Adapt the model for state construction
    StateConstructionModelAdaptor.add_state_construction_components(
        system=simulator.system,
        seq=seq_csc,
        instance_data=reference_data["instance_data"],
        loc_state=reference_data["loc_state"],
        var_state=reference_data["var_state"]
    )

    # Export the generated model
    output_model_xml = uppaal_system_to_xml(system=system)
    with open(introduction_model_rinast_csc_path, "w") as file:
        file.write(output_model_xml)


def experiment_introduction_example_oc(reference_data):
    """Executes the introduction model experiment based on the OC approach.

    Args:
        reference_data: The reference data based on which the construction is performed.
    """
    simulator = Simulator()
    simulator.load_system(adapted_introduction_model_path)
    system = simulator.system

    # Generate initial DBM
    clocks_without_ref = simulator.system_state.dbm_state.clocks[1:]
    dbm_init = DBM(clocks=clocks_without_ref, zero_init=True)

    # Generate "OC" DBM construction sequence from simulated sequence
    dbm_constructor_oc = DBMReconstructorOC(dbm_init=dbm_init)
    dbm_constructor_oc.set_strategies(approximation=ApproximationStrategy.SEQ, constraint=ConstraintStrategy.FCS)
    data_for_rec = {
        "dbm_init": dbm_init,
        "dbm_target": reference_data["dbm"],
        "seq_full": reference_data["seq"],
        "seq_incr": reference_data["seq"]
    }
    res_csc = dbm_constructor_oc.time_measured_reconstruction(data_for_rec=data_for_rec)
    seq_csc = res_csc["seq_rec"]
    print(f'\n== DBM construction sequence (OC) (len: {len(seq_csc)}) ==')
    print(seq_csc)

    # Adapt the model for state construction
    StateConstructionModelAdaptor.add_state_construction_components(
        system=simulator.system,
        seq=seq_csc,
        instance_data=reference_data["instance_data"],
        loc_state=reference_data["loc_state"],
        var_state=reference_data["var_state"]
    )

    # Export the generated model
    output_model_xml = uppaal_system_to_xml(system=system)
    with open(introduction_model_oc_csc_path, "w") as file:
        file.write(output_model_xml)
