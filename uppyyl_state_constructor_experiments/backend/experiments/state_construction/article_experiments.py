"""This module provides all DBM state construction experiments."""

import json
import os
import pathlib
import pprint
import unittest

from uppyyl_state_constructor_experiments.backend.experiments.review.model_data import all_model_data
from uppyyl_simulator.backend.data_structures.dbm.dbm import (
    DBM
)
from uppyyl_simulator.backend.simulator.simulator import (
    Simulator
)
from uppyyl_state_constructor.backend.dbm_constructor.oc.dbm_constructor_oc import (
    DBMReconstructorOC, ApproximationStrategy, ConstraintStrategy
)
from uppyyl_state_constructor.backend.dbm_constructor.random_sequence_generator import (
    RandomSequenceGenerator
)
from uppyyl_state_constructor.backend.dbm_constructor.rinast.dbm_constructor_rinast import (
    DBMReconstructorRinast
)
from uppyyl_state_constructor.backend.dbm_constructor.trivial.dbm_constructor_trivial import (
    DBMReconstructorTrivial
)
from uppyyl_state_constructor_experiments.backend.helper import \
    calculate_derived_step_measures_over_model_simulation_runs, get_model_details, \
    calculate_derived_step_measures_over_random_sequence_runs
from uppyyl_state_constructor_experiments.definitions import RES_DIR

pp = pprint.PrettyPrinter(indent=4, compact=True)
printExpectedResults = False
printActualResults = False

print_details = True


##########
# Helper #
##########
def print_header(text):
    """Prints a header line for an experiment.

    Args:
        text: The header text.

    Returns:
        The header line.
    """
    main_line = f'=== {text} ==='
    highlight_line = '=' * len(main_line)
    header = f'{highlight_line}\n{main_line}\n{highlight_line}'
    print(header)

    return header


class Error(Exception):
    """Base class for exceptions."""
    pass


class SimulationError(Error):
    """Exception raised for simulation errors due to deadlocks.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


# def print_measure_data(measure_data):
#     string = ""
#     string += "[OC:O(DBM)+C(FCS)]"
#     string += "" Len: {}
#     string += ", GenSeq: {:.4f}".format(rec_measures["full_gen_time"])
#     string += ", AppTime: {:.4f}".format(rec_measures["full_app_time"])
#     print(string)

##################################
# State Construction Experiments #
##################################
class Experiments:
    """The state construction experiments class."""

    def __init__(self):
        """Initializes Experiments."""
        self.uppyyl_simulator = Simulator()
        self.output_base_folder = RES_DIR.joinpath(f'logs/clock_state_construction/data_logs/')

    #################
    # Model Details #
    #################
    def gather_model_details(self):
        """Performs state construction for the model suite at selected steps."""
        print_header("Test SSR on Models at selected steps")
        output_folder = os.path.join(self.output_base_folder, f'experiment_model_data')

        selected_model_data = all_model_data

        all_model_details = {}
        for i, model_data in enumerate(selected_model_data):
            model_file_path = model_data["path"]
            model_file_name = os.path.basename(model_file_path)
            model_name = os.path.splitext(model_file_name)[0]
            print(f'Gathering details for model "{model_name}" ...')

            self.uppyyl_simulator.load_system(model_data["path"])
            model_details = self.uppyyl_simulator.get_system_details()
            all_model_details[model_name] = model_details

        pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
        output_file_path = '{}/data.json'.format(output_folder)
        with open(output_file_path, 'w') as file:
            json.dump(all_model_details, file, sort_keys=False, separators=(',', ':'))  # indent=0)
        print(f'Data saved to file {output_file_path}.')

    #####################
    # Model Simulations #
    #####################
    def state_construction_on_models_at_selected_steps(self):
        """Performs state construction for the model suite at selected steps."""
        print_header("Test SSR on Models at selected steps")
        run_count = 1000
        step_count = 100
        tracked_steps = [1, 10, 20, 50, 100]
        tracked_steps = list(filter(lambda x: x <= step_count, tracked_steps))
        output_folder = os.path.join(self.output_base_folder, f'test_ssr_on_models_at_selected_steps')

        selected_model_data = all_model_data
        all_model_measures = {}

        for i, model_data in enumerate(selected_model_data):
            model_file_path = model_data["path"]
            model_file_name = os.path.basename(model_file_path)
            model_name = os.path.splitext(model_file_name)[0]

            model_measures = self._perform_model_simulation_runs(model_data=model_data, run_count=run_count,
                                                                 step_count=step_count, tracked_steps=tracked_steps)
            all_model_measures[model_name] = model_measures

            pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
            output_file_path = '{}/{:03d}_{}.json'.format(output_folder, i + 1, model_name)
            with open(output_file_path, 'w') as file:
                json.dump(model_measures, file, sort_keys=False, separators=(',', ':'))  # indent=0)
            print(f'Data saved to file {output_file_path}.')
        print(f'All data saved to folder {output_folder}.')

    def state_construction_on_models_at_every_step(self):
        """Performs state construction for the model suite at every steps."""
        print_header("Test SSR on Models at every steps")
        run_count = 1000
        step_count = 100
        tracked_steps = list(range(1, 101))  # [1,10,20,50,100]
        tracked_steps = list(filter(lambda x: x <= step_count, tracked_steps))
        output_folder = os.path.join(self.output_base_folder, f'test_ssr_on_models_at_every_step')

        selected_model_data = all_model_data  # all_model_data[0:2]
        all_model_measures = {}

        for i, model_data in enumerate(selected_model_data):
            model_file_path = model_data["path"]
            model_file_name = os.path.basename(model_file_path)
            model_name = os.path.splitext(model_file_name)[0]

            model_measures = self._perform_model_simulation_runs(model_data=model_data, run_count=run_count,
                                                                 step_count=step_count, tracked_steps=tracked_steps)
            all_model_measures[model_name] = model_measures

            pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
            output_file_path = '{}/{:03d}_{}.json'.format(output_folder, i + 1, model_name)
            with open(output_file_path, 'w') as file:
                json.dump(model_measures, file, sort_keys=False, separators=(',', ':'))  # indent=0)
            print(f'Data saved to file {output_file_path}.')
        print(f'All data saved to folder {output_folder}.')

    ####################
    # Random Sequences #
    ####################
    def state_construction_on_random_sequences_with_varying_clocks(self):
        """Performs state construction for randomly generated operation sequences with varying clock counts."""
        print_header("Test SSR on Random Sequences")
        max_clock_count = 10
        sequence_length = 100
        run_count = 1000
        all_measures = []
        output_folder = os.path.join(self.output_base_folder, f'test_ssr_on_random_sequences_with_varying_clocks')
        for clock_count in range(1, max_clock_count + 1):
            all_runs_measures = self._perform_random_sequence_runs(clock_count=clock_count,
                                                                   sequence_length=sequence_length, run_count=run_count)
            measures_for_clock_count = {
                "clock_count": clock_count,
                "sequence_length": sequence_length,
                "all_runs_measures": all_runs_measures,
            }
            all_measures.append(measures_for_clock_count)

        pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
        output_path = '{}/data.json'.format(output_folder)
        with open(output_path, 'w') as file:
            json.dump(all_measures, file, sort_keys=False, separators=(',', ':'))  # indent=0)
        print(f'All data saved to file {output_path}.')

    def state_construction_on_random_sequences_with_varying_lengths(self):
        """Performs state construction for randomly generated operation sequences of varying lengths."""
        print_header("Test SSR on Random Sequences")
        max_sequence_length = 500
        sequence_length_step_size = 50
        clock_count = 5
        run_count = 1000
        all_measures = []
        output_folder = os.path.join(self.output_base_folder, f'test_ssr_on_random_sequences_with_varying_lengths')
        for sequence_length in range(sequence_length_step_size, max_sequence_length + 1, sequence_length_step_size):
            all_runs_measures = self._perform_random_sequence_runs(clock_count=clock_count,
                                                                   sequence_length=sequence_length, run_count=run_count)
            measures_for_clock_count = {
                "clock_count": clock_count,
                "sequence_length": sequence_length,
                "all_runs_measures": all_runs_measures,
            }
            all_measures.append(measures_for_clock_count)

        pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
        output_path = '{}/data.json'.format(output_folder)
        with open(output_path, 'w') as file:
            json.dump(all_measures, file, sort_keys=False, separators=(',', ':'))  # indent=0)
        print(f'All data saved to file {output_path}.')

    ################################################################################
    # Experiment Helper Functions #
    ################################################################################

    # Perform Random Sequence Runs #
    def _perform_random_sequence_runs(self, clock_count, sequence_length, run_count):
        all_runs_measures = []
        clocks = list(map(lambda c: f't{c}', range(1, clock_count + 1)))
        dbm_init = DBM(clocks=clocks, zero_init=True)
        random_seq_gen = RandomSequenceGenerator(sequence_length=sequence_length, dbm_init=dbm_init,
                                                 non_zero_resets=True, include_init_sequence=True)

        reconstructors = {
            "dbm_reconstructor_trivial": DBMReconstructorTrivial(on_the_fly=False),
            "dbm_reconstructor_rinast": DBMReconstructorRinast(
                clock_names=dbm_init.clocks.copy(), var_names=[], on_the_fly=True),
            "dbm_reconstructor_oc": DBMReconstructorOC(dbm_init=dbm_init.copy(), on_the_fly=False),
        }

        for i in range(0, run_count):
            print(f'Run {i + 1}/{run_count} (clocks={clock_count}, length={sequence_length})...')
            for reconstructor in reconstructors.values():
                reconstructor.clear()
            seq_gen = random_seq_gen.generate()
            dbm_target = random_seq_gen.dbm.copy()
            assert len(seq_gen) == sequence_length

            # curr_data = {
            #     "seq": seq_gen,
            #     "dbm": dbm_target,
            # }
            # incr_data = {
            #     "seq": seq_gen,
            #     "dbm": dbm_target,
            # }

            data_for_rec = {
                "dbm_init": dbm_init,
                "dbm_target": dbm_target,
                "seq_full": seq_gen,
                "seq_incr": seq_gen
            }

            rec_measures = self._perform_reconstructions(reconstructors=reconstructors, data_for_rec=data_for_rec)
            all_runs_measures.append(rec_measures)

        extended_measures = {
            "derived_measures": calculate_derived_step_measures_over_random_sequence_runs(all_runs_measures),
            "measures": all_runs_measures,
        }
        return extended_measures

    # Perform Model Simulation runs #
    def _perform_model_simulation_runs(self, model_data, run_count, step_count, tracked_steps):
        model_path = model_data["path"]
        model_name = os.path.basename(model_path)

        all_runs_measures = {
            "tracked_steps": tracked_steps,
            "measures": [],
            "all_run_measures": None
        }

        self.uppyyl_simulator.load_system(model_data["path"])

        run = 0
        while run < run_count:
            # for run in range(0,run_count):
            print(f'Executing run {run + 1}/{run_count} [Model: {model_name}]...')
            try:
                single_run_measures = self._perform_model_simulation_run(step_count=step_count,
                                                                         tracked_steps=tracked_steps)
            except SimulationError as e:
                print("Run aborted as deadlock was reached. Starting a new run instead.")
                print(e)
                continue
            all_runs_measures["measures"].append(single_run_measures)
            run += 1

        model_measures = {
            "model_details": get_model_details(self.uppyyl_simulator, model_data),
            "derived_measures": calculate_derived_step_measures_over_model_simulation_runs(all_runs_measures),
            "all_runs_measures": all_runs_measures,
        }
        return model_measures

    # Perform Model Simulation run #
    def _perform_model_simulation_run(self, step_count, tracked_steps):
        single_run_measures = {
            "steps": None,
            "measures": [],
        }
        step = 0

        self.uppyyl_simulator.init_simulator()

        clocks_without_ref = self.uppyyl_simulator.system_state.dbm_state.clocks[1:]
        dbm_init = DBM(clocks=clocks_without_ref, zero_init=True)

        reconstructors = {
            "dbm_reconstructor_trivial": DBMReconstructorTrivial(on_the_fly=False),
            "dbm_reconstructor_rinast": DBMReconstructorRinast(
                clock_names=dbm_init.clocks.copy(), var_names=[], on_the_fly=True),
            "dbm_reconstructor_oc": DBMReconstructorOC(dbm_init=dbm_init.copy(), on_the_fly=False),
        }

        seq_length = 0
        for i in range(0, step_count):
            if print_details:
                print(f'Step {i + 1}/{step_count} ...')  # , end='\r')
            transition = self.uppyyl_simulator.simulate_step()
            if not transition:
                raise SimulationError("No transition possible.")
            step = i + 1

            dbm_target = self.uppyyl_simulator.system_state.dbm_state.copy()

            if step in tracked_steps:
                # if True:
                seq_tracked = self.uppyyl_simulator.get_sequence()
                seq_diff = seq_tracked[seq_length:]
                seq_length = len(seq_tracked)
                dbm_rec_from_seq_tracked = seq_tracked.apply(dbm_init.copy())
                assert dbm_rec_from_seq_tracked == dbm_target

                # curr_data = {
                #     "seq": seq_tracked,
                #     "dbm": dbm_rec_from_seq_tracked,
                # }
                # incr_data = {
                #     "seq": seq_diff,
                #     "dbm": dbm_rec_from_seq_tracked,
                # }

                data_for_rec = {
                    "dbm_init": dbm_init,
                    "dbm_target": dbm_target,
                    "seq_full": seq_tracked,
                    "seq_incr": seq_diff
                }

                rec_measures = self._perform_reconstructions(reconstructors=reconstructors, data_for_rec=data_for_rec)
                rec_measures["step"] = step
                single_run_measures["measures"].append(rec_measures)

        single_run_measures["steps"] = step
        return single_run_measures

    # Perform Reconstructions #
    @staticmethod
    def _perform_reconstructions(reconstructors, data_for_rec):
        run_measures = {
            "trivial": None,
            "rinast": None,
            "oc": {
                "o_seq_c_fcs": {},
                "o_seq_c_mcs": {},
                "o_seq_c_rcs": {},
                "o_dbm_c_fcs": {},
                "o_dbm_c_mcs": {},
                "o_dbm_c_rcs": {},
            }
        }

        dbm_reconstructor_trivial = reconstructors["dbm_reconstructor_trivial"]
        dbm_reconstructor_rinast = reconstructors["dbm_reconstructor_rinast"]
        dbm_reconstructor_oc = reconstructors["dbm_reconstructor_oc"]

        dbm_target = data_for_rec["dbm_target"]

        # Reference sequence
        run_measures["ref"] = {"seq_full_length": len(data_for_rec["seq_full"])}

        # Trivial approach
        # dbm_reconstructor_trivial.dbm_init = dbm_init
        res = dbm_reconstructor_trivial.time_measured_reconstruction(data_for_rec=data_for_rec)
        run_measures["trivial"] = res["measures"]
        if print_details:
            print("[Trivial] Len: {}, GenSeq: {:.4f}, AppTime: {:.4f}".format(
                res["measures"]["seq_full_length"], res["measures"]["full_gen_time"], res["measures"]["full_app_time"]))
        assert res["dbm_rec"] == dbm_target

        # Rinast approach
        # dbm_reconstructor_rinast.dbm_init = dbm_init
        res = dbm_reconstructor_rinast.time_measured_reconstruction(data_for_rec=data_for_rec)
        run_measures["rinast"] = res["measures"]
        if print_details:
            print("[Rinast] Len: {}, GenSeq: {:.4f}, AppTime: {:.4f}".format(
                res["measures"]["seq_full_length"], res["measures"]["full_gen_time"], res["measures"]["full_app_time"]))
        assert res["dbm_rec"] == dbm_target

        # OC approach - O(SEQ) + C
        # dbm_reconstructor_oc.dbm_init = dbm_init
        dbm_reconstructor_oc.set_strategies(approximation=ApproximationStrategy.SEQ)
        res_approx = dbm_reconstructor_oc.time_measured_approximation(data_for_rec=data_for_rec)

        data_for_rec_constr = data_for_rec.copy()
        data_for_rec_constr["dbm_init"] = res_approx["dbm_approx"]

        dbm_reconstructor_oc.set_strategies(constraint=ConstraintStrategy.FCS)
        res_constr = dbm_reconstructor_oc.time_measured_constraint(data_for_rec=data_for_rec_constr)
        rec_measures = dbm_reconstructor_oc.calculate_full_measures(res_approx["measures"], res_constr["measures"])
        run_measures["oc"]["o_seq_c_fcs"] = {"o_phase": res_approx["measures"], "c_phase": res_constr["measures"],
                                             "full": rec_measures}
        if print_details:
            print("[OC:O(SEQ)+C(FCS)] Len: {}, GenSeq: {:.4f}, AppTime: {:.4f}".format(
                rec_measures["seq_full_length"], rec_measures["full_gen_time"], rec_measures["full_app_time"]))
        assert res_constr["dbm_constr"] == dbm_target

        dbm_reconstructor_oc.set_strategies(constraint=ConstraintStrategy.MCS)
        res_constr = dbm_reconstructor_oc.time_measured_constraint(data_for_rec=data_for_rec_constr)
        rec_measures = dbm_reconstructor_oc.calculate_full_measures(res_approx["measures"], res_constr["measures"])
        run_measures["oc"]["o_seq_c_mcs"] = {"o_phase": res_approx["measures"], "c_phase": res_constr["measures"],
                                             "full": rec_measures}
        if print_details:
            print("[OC:O(SEQ)+C(MCS)] Len: {}, GenSeq: {:.4f}, AppTime: {:.4f}".format(
                rec_measures["seq_full_length"], rec_measures["full_gen_time"], rec_measures["full_app_time"]))
        assert res_constr["dbm_constr"] == dbm_target

        dbm_reconstructor_oc.set_strategies(constraint=ConstraintStrategy.RCS)
        res_constr = dbm_reconstructor_oc.time_measured_constraint(data_for_rec=data_for_rec_constr)
        rec_measures = dbm_reconstructor_oc.calculate_full_measures(res_approx["measures"], res_constr["measures"])
        run_measures["oc"]["o_seq_c_rcs"] = {"o_phase": res_approx["measures"], "c_phase": res_constr["measures"],
                                             "full": rec_measures}
        if print_details:
            print("[OC:O(SEQ)+C(RCS)] Len: {}, GenSeq: {:.4f}, AppTime: {:.4f}".format(
                rec_measures["seq_full_length"], rec_measures["full_gen_time"], rec_measures["full_app_time"]))
        assert res_constr["dbm_constr"] == dbm_target

        # OC approach - O(DBM) + C
        # dbm_reconstructor_oc.dbm_init = dbm_init
        dbm_reconstructor_oc.set_strategies(approximation=ApproximationStrategy.DBM)
        res_approx = dbm_reconstructor_oc.time_measured_approximation(data_for_rec=data_for_rec)

        data_for_rec_constr = data_for_rec.copy()
        data_for_rec_constr["dbm_init"] = res_approx["dbm_approx"]

        dbm_reconstructor_oc.set_strategies(constraint=ConstraintStrategy.FCS)
        res_constr = dbm_reconstructor_oc.time_measured_constraint(data_for_rec=data_for_rec_constr)
        rec_measures = dbm_reconstructor_oc.calculate_full_measures(res_approx["measures"], res_constr["measures"])
        run_measures["oc"]["o_dbm_c_fcs"] = {"o_phase": res_approx["measures"], "c_phase": res_constr["measures"],
                                             "full": rec_measures}
        if print_details:
            print("[OC:O(DBM)+C(FCS)] Len: {}, GenSeq: {:.4f}, AppTime: {:.4f}".format(
                rec_measures["seq_full_length"], rec_measures["full_gen_time"], rec_measures["full_app_time"]))
        assert res_constr["dbm_constr"] == dbm_target

        dbm_reconstructor_oc.set_strategies(constraint=ConstraintStrategy.MCS)
        res_constr = dbm_reconstructor_oc.time_measured_constraint(data_for_rec=data_for_rec_constr)
        rec_measures = dbm_reconstructor_oc.calculate_full_measures(res_approx["measures"], res_constr["measures"])
        run_measures["oc"]["o_dbm_c_mcs"] = {"o_phase": res_approx["measures"], "c_phase": res_constr["measures"],
                                             "full": rec_measures}
        if print_details:
            print("[OC:O(DBM)+C(MCS)] Len: {}, GenSeq: {:.4f}, AppTime: {:.4f}".format(
                rec_measures["seq_full_length"], rec_measures["full_gen_time"], rec_measures["full_app_time"]))
        assert res_constr["dbm_constr"] == dbm_target

        dbm_reconstructor_oc.set_strategies(constraint=ConstraintStrategy.RCS)
        res_constr = dbm_reconstructor_oc.time_measured_constraint(data_for_rec=data_for_rec_constr)
        rec_measures = dbm_reconstructor_oc.calculate_full_measures(res_approx["measures"], res_constr["measures"])
        run_measures["oc"]["o_dbm_c_rcs"] = {"o_phase": res_approx["measures"], "c_phase": res_constr["measures"],
                                             "full": rec_measures}
        if print_details:
            print("[OC:O(DBM)+C(RCS)] Len: {}, GenSeq: {:.4f}, AppTime: {:.4f}".format(
                rec_measures["seq_full_length"], rec_measures["full_gen_time"], rec_measures["full_app_time"]))
        assert res_constr["dbm_constr"] == dbm_target

        return run_measures


################################################################################

if __name__ == '__main__':
    unittest.main()
