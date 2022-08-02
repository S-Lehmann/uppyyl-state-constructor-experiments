import numpy as np


def calculate_min_max_avg_int(vals):
    """Calculates the minimum, maximum, and average value of a given list.

    Args:
        vals: The value list.

    Returns:
        The minimum, maximum, and average value.
    """
    val_min = int(np.around(np.amin(vals)))
    val_max = int(np.around(np.amax(vals)))
    val_avg = int(np.around(np.average(vals)))
    return val_min, val_max, val_avg


def calculate_min_max_avg_float(vals):
    """Calculates the minimum, maximum, and average value of a given list.

    Args:
        vals: The value list.

    Returns:
        The minimum, maximum, and average value.
    """
    val_min = float(np.amin(vals))
    val_max = float(np.amax(vals))
    val_avg = float(np.average(vals))
    return val_min, val_max, val_avg


def calculate_derived_step_measures_over_random_sequence_runs(all_runs_measures):
    """Calculates derived step measures (such as min, max, and avg values) over random sequence run measures.

    Args:
        all_runs_measures: All random sequence run measures.

    Returns:
        The derived step measures.
    """
    # Sequence lengths
    seq_ref_lengths = list(map(lambda m: m["ref"]["seq_full_length"], all_runs_measures))
    seq_trivial_lengths = list(map(lambda m: m["trivial"]["seq_full_length"], all_runs_measures))
    seq_rinast_lengths = list(map(lambda m: m["rinast"]["seq_full_length"], all_runs_measures))

    seq_oc_o_dbm_c_fcs_lengths = list(
        map(lambda m: m["oc"]["o_dbm_c_fcs"]["full"]["seq_full_length"], all_runs_measures))
    seq_oc_o_dbm_c_mcs_lengths = list(
        map(lambda m: m["oc"]["o_dbm_c_mcs"]["full"]["seq_full_length"], all_runs_measures))
    seq_oc_o_dbm_c_rcs_lengths = list(
        map(lambda m: m["oc"]["o_dbm_c_rcs"]["full"]["seq_full_length"], all_runs_measures))
    seq_oc_o_seq_c_fcs_lengths = list(
        map(lambda m: m["oc"]["o_seq_c_fcs"]["full"]["seq_full_length"], all_runs_measures))
    seq_oc_o_seq_c_mcs_lengths = list(
        map(lambda m: m["oc"]["o_seq_c_mcs"]["full"]["seq_full_length"], all_runs_measures))
    seq_oc_o_seq_c_rcs_lengths = list(
        map(lambda m: m["oc"]["o_seq_c_rcs"]["full"]["seq_full_length"], all_runs_measures))

    seq_approx_dbm_lengths = list(
        map(lambda m: m["oc"]["o_dbm_c_fcs"]["o_phase"]["seq_full_length"], all_runs_measures))
    seq_approx_seq_lengths = list(
        map(lambda m: m["oc"]["o_seq_c_fcs"]["o_phase"]["seq_full_length"], all_runs_measures))

    seq_constr_fcs_lengths = list(
        map(lambda m: m["oc"]["o_dbm_c_fcs"]["c_phase"]["seq_full_length"], all_runs_measures))
    seq_constr_mcs_lengths = list(
        map(lambda m: m["oc"]["o_dbm_c_mcs"]["c_phase"]["seq_full_length"], all_runs_measures))
    seq_constr_rcs_lengths = list(
        map(lambda m: m["oc"]["o_dbm_c_rcs"]["c_phase"]["seq_full_length"], all_runs_measures))

    # Generation times
    seq_trivial_gen_times = list(map(lambda m: m["trivial"]["full_gen_time"], all_runs_measures))
    seq_rinast_gen_times = list(map(lambda m: m["rinast"]["full_gen_time"], all_runs_measures))

    seq_oc_o_dbm_c_fcs_gen_times = list(
        map(lambda m: m["oc"]["o_dbm_c_fcs"]["full"]["full_gen_time"], all_runs_measures))
    seq_oc_o_dbm_c_mcs_gen_times = list(
        map(lambda m: m["oc"]["o_dbm_c_mcs"]["full"]["full_gen_time"], all_runs_measures))
    seq_oc_o_dbm_c_rcs_gen_times = list(
        map(lambda m: m["oc"]["o_dbm_c_rcs"]["full"]["full_gen_time"], all_runs_measures))
    seq_oc_o_seq_c_fcs_gen_times = list(
        map(lambda m: m["oc"]["o_seq_c_fcs"]["full"]["full_gen_time"], all_runs_measures))
    seq_oc_o_seq_c_mcs_gen_times = list(
        map(lambda m: m["oc"]["o_seq_c_mcs"]["full"]["full_gen_time"], all_runs_measures))
    seq_oc_o_seq_c_rcs_gen_times = list(
        map(lambda m: m["oc"]["o_seq_c_rcs"]["full"]["full_gen_time"], all_runs_measures))

    seq_approx_dbm_gen_times = list(
        map(lambda m: m["oc"]["o_dbm_c_fcs"]["o_phase"]["full_gen_time"], all_runs_measures))
    seq_approx_seq_gen_times = list(
        map(lambda m: m["oc"]["o_seq_c_fcs"]["o_phase"]["full_gen_time"], all_runs_measures))

    seq_constr_fcs_gen_times = list(
        map(lambda m: m["oc"]["o_dbm_c_fcs"]["c_phase"]["full_gen_time"], all_runs_measures))
    seq_constr_mcs_gen_times = list(
        map(lambda m: m["oc"]["o_dbm_c_mcs"]["c_phase"]["full_gen_time"], all_runs_measures))
    seq_constr_rcs_gen_times = list(
        map(lambda m: m["oc"]["o_dbm_c_rcs"]["c_phase"]["full_gen_time"], all_runs_measures))

    # Generation times
    seq_trivial_app_times = list(map(lambda m: m["trivial"]["full_app_time"], all_runs_measures))
    seq_rinast_app_times = list(map(lambda m: m["rinast"]["full_app_time"], all_runs_measures))

    seq_oc_o_dbm_c_fcs_app_times = list(
        map(lambda m: m["oc"]["o_dbm_c_fcs"]["full"]["full_app_time"], all_runs_measures))
    seq_oc_o_dbm_c_mcs_app_times = list(
        map(lambda m: m["oc"]["o_dbm_c_mcs"]["full"]["full_app_time"], all_runs_measures))
    seq_oc_o_dbm_c_rcs_app_times = list(
        map(lambda m: m["oc"]["o_dbm_c_rcs"]["full"]["full_app_time"], all_runs_measures))
    seq_oc_o_seq_c_fcs_app_times = list(
        map(lambda m: m["oc"]["o_seq_c_fcs"]["full"]["full_app_time"], all_runs_measures))
    seq_oc_o_seq_c_mcs_app_times = list(
        map(lambda m: m["oc"]["o_seq_c_mcs"]["full"]["full_app_time"], all_runs_measures))
    seq_oc_o_seq_c_rcs_app_times = list(
        map(lambda m: m["oc"]["o_seq_c_rcs"]["full"]["full_app_time"], all_runs_measures))

    seq_approx_dbm_app_times = list(
        map(lambda m: m["oc"]["o_dbm_c_fcs"]["o_phase"]["full_app_time"], all_runs_measures))
    seq_approx_seq_app_times = list(
        map(lambda m: m["oc"]["o_seq_c_fcs"]["o_phase"]["full_app_time"], all_runs_measures))

    seq_constr_fcs_app_times = list(
        map(lambda m: m["oc"]["o_dbm_c_fcs"]["c_phase"]["full_app_time"], all_runs_measures))
    seq_constr_mcs_app_times = list(
        map(lambda m: m["oc"]["o_dbm_c_mcs"]["c_phase"]["full_app_time"], all_runs_measures))
    seq_constr_rcs_app_times = list(
        map(lambda m: m["oc"]["o_dbm_c_rcs"]["c_phase"]["full_app_time"], all_runs_measures))

    derived_step_measure = {
        # Sequence lengths
        "seq_ref_length_min_max_avg": calculate_min_max_avg_int(seq_ref_lengths),
        "seq_trivial_length_min_max_avg": calculate_min_max_avg_int(seq_trivial_lengths),
        "seq_rinast_length_min_max_avg": calculate_min_max_avg_int(seq_rinast_lengths),
        "seq_oc_o_dbm_c_fcs_lengths_min_max_avg": calculate_min_max_avg_int(seq_oc_o_dbm_c_fcs_lengths),
        "seq_oc_o_dbm_c_mcs_lengths_min_max_avg": calculate_min_max_avg_int(seq_oc_o_dbm_c_mcs_lengths),
        "seq_oc_o_dbm_c_rcs_lengths_min_max_avg": calculate_min_max_avg_int(seq_oc_o_dbm_c_rcs_lengths),
        "seq_oc_o_seq_c_fcs_lengths_min_max_avg": calculate_min_max_avg_int(seq_oc_o_seq_c_fcs_lengths),
        "seq_oc_o_seq_c_mcs_lengths_min_max_avg": calculate_min_max_avg_int(seq_oc_o_seq_c_mcs_lengths),
        "seq_oc_o_seq_c_rcs_lengths_min_max_avg": calculate_min_max_avg_int(seq_oc_o_seq_c_rcs_lengths),

        "seq_approx_dbm_length_min_max_avg": calculate_min_max_avg_int(seq_approx_dbm_lengths),
        "seq_approx_seq_length_min_max_avg": calculate_min_max_avg_int(seq_approx_seq_lengths),

        "seq_constr_fcs_length_min_max_avg": calculate_min_max_avg_int(seq_constr_fcs_lengths),
        "seq_constr_mcs_length_min_max_avg": calculate_min_max_avg_int(seq_constr_mcs_lengths),
        "seq_constr_rcs_length_min_max_avg": calculate_min_max_avg_int(seq_constr_rcs_lengths),

        # Generation times
        "seq_trivial_gen_time_min_max_avg": calculate_min_max_avg_float(seq_trivial_gen_times),
        "seq_rinast_gen_time_min_max_avg": calculate_min_max_avg_float(seq_rinast_gen_times),

        "seq_oc_o_dbm_c_fcs_gen_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_dbm_c_fcs_gen_times),
        "seq_oc_o_dbm_c_mcs_gen_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_dbm_c_mcs_gen_times),
        "seq_oc_o_dbm_c_rcs_gen_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_dbm_c_rcs_gen_times),
        "seq_oc_o_seq_c_fcs_gen_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_seq_c_fcs_gen_times),
        "seq_oc_o_seq_c_mcs_gen_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_seq_c_mcs_gen_times),
        "seq_oc_o_seq_c_rcs_gen_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_seq_c_rcs_gen_times),

        "seq_approx_dbm_gen_time_min_max_avg": calculate_min_max_avg_float(seq_approx_dbm_gen_times),
        "seq_approx_seq_gen_time_min_max_avg": calculate_min_max_avg_float(seq_approx_seq_gen_times),

        "seq_constr_fcs_gen_time_min_max_avg": calculate_min_max_avg_float(seq_constr_fcs_gen_times),
        "seq_constr_mcs_gen_time_min_max_avg": calculate_min_max_avg_float(seq_constr_mcs_gen_times),
        "seq_constr_rcs_gen_time_min_max_avg": calculate_min_max_avg_float(seq_constr_rcs_gen_times),

        # Application times
        "seq_trivial_app_time_min_max_avg": calculate_min_max_avg_float(seq_trivial_app_times),
        "seq_rinast_app_time_min_max_avg": calculate_min_max_avg_float(seq_rinast_app_times),

        "seq_oc_o_dbm_c_fcs_app_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_dbm_c_fcs_app_times),
        "seq_oc_o_dbm_c_mcs_app_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_dbm_c_mcs_app_times),
        "seq_oc_o_dbm_c_rcs_app_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_dbm_c_rcs_app_times),
        "seq_oc_o_seq_c_fcs_app_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_seq_c_fcs_app_times),
        "seq_oc_o_seq_c_mcs_app_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_seq_c_mcs_app_times),
        "seq_oc_o_seq_c_rcs_app_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_seq_c_rcs_app_times),

        "seq_approx_dbm_app_time_min_max_avg": calculate_min_max_avg_float(seq_approx_dbm_app_times),
        "seq_approx_seq_app_time_min_max_avg": calculate_min_max_avg_float(seq_approx_seq_app_times),

        "seq_constr_fcs_app_time_min_max_avg": calculate_min_max_avg_float(seq_constr_fcs_app_times),
        "seq_constr_mcs_app_time_min_max_avg": calculate_min_max_avg_float(seq_constr_mcs_app_times),
        "seq_constr_rcs_app_time_min_max_avg": calculate_min_max_avg_float(seq_constr_rcs_app_times),
    }

    return derived_step_measure


def calculate_derived_step_measures_over_model_simulation_runs(all_runs_measures):
    """Calculates derived step measures (such as min, max, and avg values) over random model simulation runs.

    Args:
        all_runs_measures: All random model simulation run measures.

    Returns:
        The derived step measures.
    """
    all_steps_measures_over_runs = []
    for i, step in enumerate(all_runs_measures["tracked_steps"]):
        seq_ref_lengths = list(map(lambda m: m["measures"][i]["ref"]["seq_full_length"], all_runs_measures["measures"]))
        seq_trivial_lengths = list(
            map(lambda m: m["measures"][i]["trivial"]["seq_full_length"], all_runs_measures["measures"]))
        seq_rinast_lengths = list(
            map(lambda m: m["measures"][i]["rinast"]["seq_full_length"], all_runs_measures["measures"]))

        seq_oc_o_dbm_c_fcs_lengths = list(
            map(lambda m: m["measures"][i]["oc"]["o_dbm_c_fcs"]["full"]["seq_full_length"],
                all_runs_measures["measures"]))
        seq_oc_o_dbm_c_mcs_lengths = list(
            map(lambda m: m["measures"][i]["oc"]["o_dbm_c_mcs"]["full"]["seq_full_length"],
                all_runs_measures["measures"]))
        seq_oc_o_dbm_c_rcs_lengths = list(
            map(lambda m: m["measures"][i]["oc"]["o_dbm_c_rcs"]["full"]["seq_full_length"],
                all_runs_measures["measures"]))
        seq_oc_o_seq_c_fcs_lengths = list(
            map(lambda m: m["measures"][i]["oc"]["o_seq_c_fcs"]["full"]["seq_full_length"],
                all_runs_measures["measures"]))
        seq_oc_o_seq_c_mcs_lengths = list(
            map(lambda m: m["measures"][i]["oc"]["o_seq_c_mcs"]["full"]["seq_full_length"],
                all_runs_measures["measures"]))
        seq_oc_o_seq_c_rcs_lengths = list(
            map(lambda m: m["measures"][i]["oc"]["o_seq_c_rcs"]["full"]["seq_full_length"],
                all_runs_measures["measures"]))

        seq_approx_dbm_lengths = list(map(lambda m: m["measures"][i]["oc"]["o_dbm_c_fcs"]["o_phase"]["seq_full_length"],
                                          all_runs_measures["measures"]))
        seq_approx_seq_lengths = list(map(lambda m: m["measures"][i]["oc"]["o_seq_c_fcs"]["o_phase"]["seq_full_length"],
                                          all_runs_measures["measures"]))

        seq_constr_fcs_lengths = list(map(lambda m: m["measures"][i]["oc"]["o_dbm_c_fcs"]["c_phase"]["seq_full_length"],
                                          all_runs_measures["measures"]))
        seq_constr_mcs_lengths = list(map(lambda m: m["measures"][i]["oc"]["o_dbm_c_mcs"]["c_phase"]["seq_full_length"],
                                          all_runs_measures["measures"]))
        seq_constr_rcs_lengths = list(map(lambda m: m["measures"][i]["oc"]["o_dbm_c_rcs"]["c_phase"]["seq_full_length"],
                                          all_runs_measures["measures"]))

        # Generation times
        seq_trivial_gen_times = list(
            map(lambda m: m["measures"][i]["trivial"]["full_gen_time"], all_runs_measures["measures"]))
        seq_rinast_gen_times = list(
            map(lambda m: m["measures"][i]["rinast"]["full_gen_time"], all_runs_measures["measures"]))

        seq_oc_o_dbm_c_fcs_gen_times = list(
            map(lambda m: m["measures"][i]["oc"]["o_dbm_c_fcs"]["full"]["full_gen_time"],
                all_runs_measures["measures"]))
        seq_oc_o_dbm_c_mcs_gen_times = list(
            map(lambda m: m["measures"][i]["oc"]["o_dbm_c_mcs"]["full"]["full_gen_time"],
                all_runs_measures["measures"]))
        seq_oc_o_dbm_c_rcs_gen_times = list(
            map(lambda m: m["measures"][i]["oc"]["o_dbm_c_rcs"]["full"]["full_gen_time"],
                all_runs_measures["measures"]))
        seq_oc_o_seq_c_fcs_gen_times = list(
            map(lambda m: m["measures"][i]["oc"]["o_seq_c_fcs"]["full"]["full_gen_time"],
                all_runs_measures["measures"]))
        seq_oc_o_seq_c_mcs_gen_times = list(
            map(lambda m: m["measures"][i]["oc"]["o_seq_c_mcs"]["full"]["full_gen_time"],
                all_runs_measures["measures"]))
        seq_oc_o_seq_c_rcs_gen_times = list(
            map(lambda m: m["measures"][i]["oc"]["o_seq_c_rcs"]["full"]["full_gen_time"],
                all_runs_measures["measures"]))

        seq_approx_dbm_gen_times = list(map(lambda m: m["measures"][i]["oc"]["o_dbm_c_fcs"]["o_phase"]["full_gen_time"],
                                            all_runs_measures["measures"]))
        seq_approx_seq_gen_times = list(map(lambda m: m["measures"][i]["oc"]["o_seq_c_fcs"]["o_phase"]["full_gen_time"],
                                            all_runs_measures["measures"]))

        seq_constr_fcs_gen_times = list(map(lambda m: m["measures"][i]["oc"]["o_dbm_c_fcs"]["c_phase"]["full_gen_time"],
                                            all_runs_measures["measures"]))
        seq_constr_mcs_gen_times = list(map(lambda m: m["measures"][i]["oc"]["o_dbm_c_mcs"]["c_phase"]["full_gen_time"],
                                            all_runs_measures["measures"]))
        seq_constr_rcs_gen_times = list(map(lambda m: m["measures"][i]["oc"]["o_dbm_c_rcs"]["c_phase"]["full_gen_time"],
                                            all_runs_measures["measures"]))

        # Generation times
        seq_trivial_app_times = list(
            map(lambda m: m["measures"][i]["trivial"]["full_app_time"], all_runs_measures["measures"]))
        seq_rinast_app_times = list(
            map(lambda m: m["measures"][i]["rinast"]["full_app_time"], all_runs_measures["measures"]))

        seq_oc_o_dbm_c_fcs_app_times = list(
            map(lambda m: m["measures"][i]["oc"]["o_dbm_c_fcs"]["full"]["full_app_time"],
                all_runs_measures["measures"]))
        seq_oc_o_dbm_c_mcs_app_times = list(
            map(lambda m: m["measures"][i]["oc"]["o_dbm_c_mcs"]["full"]["full_app_time"],
                all_runs_measures["measures"]))
        seq_oc_o_dbm_c_rcs_app_times = list(
            map(lambda m: m["measures"][i]["oc"]["o_dbm_c_rcs"]["full"]["full_app_time"],
                all_runs_measures["measures"]))
        seq_oc_o_seq_c_fcs_app_times = list(
            map(lambda m: m["measures"][i]["oc"]["o_seq_c_fcs"]["full"]["full_app_time"],
                all_runs_measures["measures"]))
        seq_oc_o_seq_c_mcs_app_times = list(
            map(lambda m: m["measures"][i]["oc"]["o_seq_c_mcs"]["full"]["full_app_time"],
                all_runs_measures["measures"]))
        seq_oc_o_seq_c_rcs_app_times = list(
            map(lambda m: m["measures"][i]["oc"]["o_seq_c_rcs"]["full"]["full_app_time"],
                all_runs_measures["measures"]))

        seq_approx_dbm_app_times = list(map(lambda m: m["measures"][i]["oc"]["o_dbm_c_fcs"]["o_phase"]["full_app_time"],
                                            all_runs_measures["measures"]))
        seq_approx_seq_app_times = list(map(lambda m: m["measures"][i]["oc"]["o_seq_c_fcs"]["o_phase"]["full_app_time"],
                                            all_runs_measures["measures"]))

        seq_constr_fcs_app_times = list(map(lambda m: m["measures"][i]["oc"]["o_dbm_c_fcs"]["c_phase"]["full_app_time"],
                                            all_runs_measures["measures"]))
        seq_constr_mcs_app_times = list(map(lambda m: m["measures"][i]["oc"]["o_dbm_c_mcs"]["c_phase"]["full_app_time"],
                                            all_runs_measures["measures"]))
        seq_constr_rcs_app_times = list(map(lambda m: m["measures"][i]["oc"]["o_dbm_c_rcs"]["c_phase"]["full_app_time"],
                                            all_runs_measures["measures"]))

        derived_step_measure = {
            # Sequence lengths
            "seq_ref_length_min_max_avg": calculate_min_max_avg_int(seq_ref_lengths),
            "seq_trivial_length_min_max_avg": calculate_min_max_avg_int(seq_trivial_lengths),
            "seq_rinast_length_min_max_avg": calculate_min_max_avg_int(seq_rinast_lengths),
            "seq_oc_o_dbm_c_fcs_lengths_min_max_avg": calculate_min_max_avg_int(seq_oc_o_dbm_c_fcs_lengths),
            "seq_oc_o_dbm_c_mcs_lengths_min_max_avg": calculate_min_max_avg_int(seq_oc_o_dbm_c_mcs_lengths),
            "seq_oc_o_dbm_c_rcs_lengths_min_max_avg": calculate_min_max_avg_int(seq_oc_o_dbm_c_rcs_lengths),
            "seq_oc_o_seq_c_fcs_lengths_min_max_avg": calculate_min_max_avg_int(seq_oc_o_seq_c_fcs_lengths),
            "seq_oc_o_seq_c_mcs_lengths_min_max_avg": calculate_min_max_avg_int(seq_oc_o_seq_c_mcs_lengths),
            "seq_oc_o_seq_c_rcs_lengths_min_max_avg": calculate_min_max_avg_int(seq_oc_o_seq_c_rcs_lengths),

            "seq_approx_dbm_length_min_max_avg": calculate_min_max_avg_int(seq_approx_dbm_lengths),
            "seq_approx_seq_length_min_max_avg": calculate_min_max_avg_int(seq_approx_seq_lengths),

            "seq_constr_fcs_length_min_max_avg": calculate_min_max_avg_int(seq_constr_fcs_lengths),
            "seq_constr_mcs_length_min_max_avg": calculate_min_max_avg_int(seq_constr_mcs_lengths),
            "seq_constr_rcs_length_min_max_avg": calculate_min_max_avg_int(seq_constr_rcs_lengths),

            # Generation times
            "seq_trivial_gen_time_min_max_avg": calculate_min_max_avg_float(seq_trivial_gen_times),
            "seq_rinast_gen_time_min_max_avg": calculate_min_max_avg_float(seq_rinast_gen_times),

            "seq_oc_o_dbm_c_fcs_gen_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_dbm_c_fcs_gen_times),
            "seq_oc_o_dbm_c_mcs_gen_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_dbm_c_mcs_gen_times),
            "seq_oc_o_dbm_c_rcs_gen_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_dbm_c_rcs_gen_times),
            "seq_oc_o_seq_c_fcs_gen_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_seq_c_fcs_gen_times),
            "seq_oc_o_seq_c_mcs_gen_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_seq_c_mcs_gen_times),
            "seq_oc_o_seq_c_rcs_gen_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_seq_c_rcs_gen_times),

            "seq_approx_dbm_gen_time_min_max_avg": calculate_min_max_avg_float(seq_approx_dbm_gen_times),
            "seq_approx_seq_gen_time_min_max_avg": calculate_min_max_avg_float(seq_approx_seq_gen_times),

            "seq_constr_fcs_gen_time_min_max_avg": calculate_min_max_avg_float(seq_constr_fcs_gen_times),
            "seq_constr_mcs_gen_time_min_max_avg": calculate_min_max_avg_float(seq_constr_mcs_gen_times),
            "seq_constr_rcs_gen_time_min_max_avg": calculate_min_max_avg_float(seq_constr_rcs_gen_times),

            # Application times
            "seq_trivial_app_time_min_max_avg": calculate_min_max_avg_float(seq_trivial_app_times),
            "seq_rinast_app_time_min_max_avg": calculate_min_max_avg_float(seq_rinast_app_times),

            "seq_oc_o_dbm_c_fcs_app_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_dbm_c_fcs_app_times),
            "seq_oc_o_dbm_c_mcs_app_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_dbm_c_mcs_app_times),
            "seq_oc_o_dbm_c_rcs_app_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_dbm_c_rcs_app_times),
            "seq_oc_o_seq_c_fcs_app_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_seq_c_fcs_app_times),
            "seq_oc_o_seq_c_mcs_app_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_seq_c_mcs_app_times),
            "seq_oc_o_seq_c_rcs_app_time_min_max_avg": calculate_min_max_avg_float(seq_oc_o_seq_c_rcs_app_times),

            "seq_approx_dbm_app_time_min_max_avg": calculate_min_max_avg_float(seq_approx_dbm_app_times),
            "seq_approx_seq_app_time_min_max_avg": calculate_min_max_avg_float(seq_approx_seq_app_times),

            "seq_constr_fcs_app_time_min_max_avg": calculate_min_max_avg_float(seq_constr_fcs_app_times),
            "seq_constr_mcs_app_time_min_max_avg": calculate_min_max_avg_float(seq_constr_mcs_app_times),
            "seq_constr_rcs_app_time_min_max_avg": calculate_min_max_avg_float(seq_constr_rcs_app_times),
        }

        all_steps_measures_over_runs.append(derived_step_measure)

    return all_steps_measures_over_runs


def get_model_details(uppyyl_simulator, model_data):
    """Get the details (e.g., component counts) of a model.

    Args:
        uppyyl_simulator: The Uppaal simulator into which the model has been loaded.
        model_data: The input model data containing the path.

    Returns:
        The model details dict.
    """
    uppyyl_simulator.load_system(model_data["path"])
    model_details = uppyyl_simulator.get_system_details()
    return model_details
