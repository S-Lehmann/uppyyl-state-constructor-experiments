"""This module provides all plot functions for the DBM state construction experiments."""

import json
import os
import pprint
import pathlib
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.font_manager import FontProperties

pp = pprint.PrettyPrinter(indent=4, compact=True)


def load_all_model_data_from_folder(data_folder):
    """Loads all existing model data found in a given folder.

    Args:
        data_folder: The data folder.

    Returns:
        All loaded model data.
    """
    data_file_names = [f for f in sorted(os.listdir(data_folder)) if os.path.isfile(os.path.join(data_folder, f))]

    all_model_data = {}
    for data_file_name_with_ext in data_file_names:
        data_file_name = os.path.splitext(data_file_name_with_ext)[0]
        model_num, model_name = tuple(data_file_name.split("_", maxsplit=1))

        data_file_path = f'{data_folder}/{data_file_name_with_ext}'
        with open(data_file_path, 'r') as file:
            all_model_data[model_name] = json.load(file)

    return all_model_data


################################################################################

#######################################
# State Construction Experiment Plots #
#######################################
class Plots:
    """The state construction experiments class."""

    def __init__(self):
        """Initializes Plots."""
        self.data_base_folder = f'../logs/clock_state_construction/data_logs/'
        self.output_base_folder = f'../logs/clock_state_construction/plots/'

    ################################################################################
    # Model Details #
    ################################################################################
    def create_latex_table_model_details(
            self, all_model_details=None, save_plot=False, show_plot=False):
        """Creates a LaTeX table from the sequence reconstruction results obtained via model simulations.

        Args:
            all_model_details: All model details.
            save_plot: Choose whether the generated plot should be saved.
            show_plot: Choose whether the generated plot should be shown.
        """
        input_folder = os.path.join(self.data_base_folder, 'experiment_model_data')
        output_folder = os.path.join(self.output_base_folder, 'csv')
        data_file_path = os.path.join(input_folder, 'data.json')
        if all_model_details is None:
            with open(data_file_path, 'r') as file:
                all_model_details = json.load(file)

        csv_data = [["modelname", "locations", "edges", "instances", "clocks", "cyclic", "deadlock"]]
        for model_name, model_details in all_model_details.items():
            model_details_row = [
                model_name,
                str(model_details["location_count"]),
                str(model_details["edge_count"]),
                str(model_details["instance_count"]),
                str(model_details["clock_count"]),
                "TODO",
                "TODO",
            ]
            csv_data.append(model_details_row)

        csv_str = "\n".join(list(map(lambda row_: ";".join(row_), csv_data)))
        # print(latex_str)

        if show_plot:
            print(csv_str)
        if save_plot:
            pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

            output_csv_file = f'{output_folder}/experiment-model-data.csv'
            with open(output_csv_file, 'w') as file:
                file.write(csv_str)
            print(f'File "{output_csv_file}" saved.')

    ################################################################################
    # Model Simulations #
    ################################################################################

    ###################################################
    # Latex Table / CSV: Seq_rec length after n steps #
    ###################################################
    def create_latex_table_seq_csc_length_after_n_steps(
            self, all_measures=None, save_plot=False, show_plot=False):
        """Creates a LaTeX table from the sequence reconstruction results obtained via model simulations.

        Args:
            all_measures: All reconstruction measure data.
            save_plot: Choose whether the generated plot should be saved.
            show_plot: Choose whether the generated plot should be shown.
        """
        # input_folder = os.path.join(self.data_base_folder, 'test_ssr_on_models_at_selected_steps')
        input_folder = os.path.join(self.data_base_folder, 'test_ssr_on_models_at_every_step')
        output_folder = os.path.join(self.output_base_folder, 'csv')
        if all_measures is None:
            all_measures = load_all_model_data_from_folder(input_folder)

        tracked_steps = list(all_measures.values())[0]["all_runs_measures"]["tracked_steps"]
        selected_tracked_steps = [1, 10, 50, 100]
        tracked_step_indices = list(map(lambda step: tracked_steps.index(step), selected_tracked_steps))
        csv_data = []
        latex_str = ""
        latex_str += f' & & \\multicolumn{{{len(selected_tracked_steps)}}}{{c|}}' \
                     f'{{\\textbf{{(min/avg/max) seq. length after n steps}}}} \\\\\\hline\n'
        latex_tracked_step_strs = list(map(lambda x: f'\\textbf{{{x}}}', selected_tracked_steps))
        latex_str += f'\\textbf{{Model}} & \\textbf{{Approach}} ' \
                     f'& {" & ".join(latex_tracked_step_strs)} \\\\\\hline\n'
        for model_name, measure_data in all_measures.items():
            latex_str += f'\\multirow{{8}}{{*}}{{\\texttt{{{model_name}}}}}'
            row = [model_name, "Trivial"]
            latex_str += f' & Trivial'
            for idx in tracked_step_indices:
                min_val = measure_data["derived_measures"][idx]["seq_trivial_length_min_max_avg"][0]
                max_val = measure_data["derived_measures"][idx]["seq_trivial_length_min_max_avg"][1]
                avg_val = measure_data["derived_measures"][idx]["seq_trivial_length_min_max_avg"][2]
                row.append(f'min:{min_val}/avg:{avg_val}/max:{max_val}')
                latex_str += f' & ({min_val},{avg_val},{max_val})'
            latex_str += f' \\\\\n'
            csv_data.append(row)

            row = [model_name, "Rinast"]
            latex_str += f' & Rinast'
            for idx in tracked_step_indices:
                min_val = measure_data["derived_measures"][idx]["seq_rinast_length_min_max_avg"][0]
                max_val = measure_data["derived_measures"][idx]["seq_rinast_length_min_max_avg"][1]
                avg_val = measure_data["derived_measures"][idx]["seq_rinast_length_min_max_avg"][2]
                row.append(f'min:{min_val}/avg:{avg_val}/max:{max_val}')
                latex_str += f' & ({min_val},{avg_val},{max_val})'
            latex_str += f' \\\\\n'
            csv_data.append(row)

            row = [model_name, "O(DBM)+C(FCS)"]
            latex_str += f' & O(DBM)+C(FCS)'
            for idx in tracked_step_indices:
                min_val = measure_data["derived_measures"][idx]["seq_oc_o_dbm_c_fcs_lengths_min_max_avg"][0]
                max_val = measure_data["derived_measures"][idx]["seq_oc_o_dbm_c_fcs_lengths_min_max_avg"][1]
                avg_val = measure_data["derived_measures"][idx]["seq_oc_o_dbm_c_fcs_lengths_min_max_avg"][2]
                row.append(f'min:{min_val}/avg:{avg_val}/max:{max_val}')
                latex_str += f' & ({min_val},{avg_val},{max_val})'
            latex_str += f' \\\\\n'
            csv_data.append(row)

            row = [model_name, "O(DBM)+C(MCS)"]
            latex_str += f' & O(DBM)+C(MCS)'
            for idx in tracked_step_indices:
                min_val = measure_data["derived_measures"][idx]["seq_oc_o_dbm_c_mcs_lengths_min_max_avg"][0]
                max_val = measure_data["derived_measures"][idx]["seq_oc_o_dbm_c_mcs_lengths_min_max_avg"][1]
                avg_val = measure_data["derived_measures"][idx]["seq_oc_o_dbm_c_mcs_lengths_min_max_avg"][2]
                row.append(f'min:{min_val}/avg:{avg_val}/max:{max_val}')
                latex_str += f' & ({min_val},{avg_val},{max_val})'
            latex_str += f' \\\\\n'
            csv_data.append(row)

            row = [model_name, "O(DBM)+C(RCS)"]
            latex_str += f' & O(DBM)+C(RCS)'
            for idx in tracked_step_indices:
                min_val = measure_data["derived_measures"][idx]["seq_oc_o_dbm_c_rcs_lengths_min_max_avg"][0]
                max_val = measure_data["derived_measures"][idx]["seq_oc_o_dbm_c_rcs_lengths_min_max_avg"][1]
                avg_val = measure_data["derived_measures"][idx]["seq_oc_o_dbm_c_rcs_lengths_min_max_avg"][2]
                row.append(f'min:{min_val}/avg:{avg_val}/max:{max_val}')
                latex_str += f' & ({min_val},{avg_val},{max_val})'
            latex_str += f' \\\\\n'
            csv_data.append(row)

            row = [model_name, "O(SEQ)+C(FCS)"]
            latex_str += f' & O(SEQ)+C(FCS)'
            for idx in tracked_step_indices:
                min_val = measure_data["derived_measures"][idx]["seq_oc_o_seq_c_fcs_lengths_min_max_avg"][0]
                max_val = measure_data["derived_measures"][idx]["seq_oc_o_seq_c_fcs_lengths_min_max_avg"][1]
                avg_val = measure_data["derived_measures"][idx]["seq_oc_o_seq_c_fcs_lengths_min_max_avg"][2]
                row.append(f'min:{min_val}/avg:{avg_val}/max:{max_val}')
                latex_str += f' & ({min_val},{avg_val},{max_val})'
            latex_str += f' \\\\\n'
            csv_data.append(row)

            row = [model_name, "O(SEQ)+C(MCS)"]
            latex_str += f' & O(SEQ)+C(MCS)'
            for idx in tracked_step_indices:
                min_val = measure_data["derived_measures"][idx]["seq_oc_o_seq_c_mcs_lengths_min_max_avg"][0]
                max_val = measure_data["derived_measures"][idx]["seq_oc_o_seq_c_mcs_lengths_min_max_avg"][1]
                avg_val = measure_data["derived_measures"][idx]["seq_oc_o_seq_c_mcs_lengths_min_max_avg"][2]
                row.append(f'min:{min_val}/avg:{avg_val}/max:{max_val}')
                latex_str += f' & ({min_val},{avg_val},{max_val})'
            latex_str += f' \\\\\n'
            csv_data.append(row)

            row = [model_name, "O(SEQ)+C(RCS)"]
            latex_str += f' & O(SEQ)+C(RCS)'
            for idx in tracked_step_indices:
                min_val = measure_data["derived_measures"][idx]["seq_oc_o_seq_c_rcs_lengths_min_max_avg"][0]
                max_val = measure_data["derived_measures"][idx]["seq_oc_o_seq_c_rcs_lengths_min_max_avg"][1]
                avg_val = measure_data["derived_measures"][idx]["seq_oc_o_seq_c_rcs_lengths_min_max_avg"][2]
                row.append(f'min:{min_val}/avg:{avg_val}/max:{max_val}')
                latex_str += f' & ({min_val},{avg_val},{max_val})'
            latex_str += f' \\\\\\hline\n'
            csv_data.append(row)

        csv_str = "\n".join(list(map(lambda row_: ";".join(row_), csv_data)))
        # print(latex_str)

        if show_plot:
            print(csv_str)
        if save_plot:
            pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

            output_csv_file = f'{output_folder}/seq-csc-length-after-n-steps.csv'
            with open(output_csv_file, 'w') as file:
                file.write(csv_str)
            print(f'File "{output_csv_file}" saved.')

            output_latex_file = f'{output_folder}/seq-csc-length-after-n-steps.tex'
            with open(output_latex_file, 'w') as file:
                file.write(latex_str)
            print(f'File "{output_latex_file}" saved.')

    #######################################
    # Graph: Seq_rec length after n steps #
    #######################################
    def create_plot_seq_csc_length_after_every_step(
            self, all_measures=None, save_plot=False, show_plot=False):
        """Creates a plot from the sequence reconstruction results obtained via model simulations.

        Args:
            all_measures: All reconstruction measure data.
            save_plot: Choose whether the generated plot should be saved.
            show_plot: Choose whether the generated plot should be shown.
        """
        input_folder = os.path.join(self.data_base_folder, 'test_ssr_on_models_at_every_step')
        output_folder = os.path.join(self.output_base_folder, 'graphs')
        if all_measures is None:
            all_measures = load_all_model_data_from_folder(input_folder)

        plt.clf()
        dpi = 300
        plt.figure(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi)

        markersize = 2
        font_p = FontProperties()
        font_p.set_size('small')

        # plt.subplots_adjust(hspace=1, top=0.8)
        ax = plt.subplot(1, 1, 1)

        tracked_steps = list(all_measures.values())[0]["all_runs_measures"]["tracked_steps"]
        selected_measures = {k: all_measures[k] for k in ("2doors", "bridge", "csmacd2") if k in all_measures}
        for model_name, measure_data in selected_measures.items():
            x_vals = tracked_steps

            # Trivial
            y_vals = list(map(lambda step_measure: step_measure["seq_trivial_length_min_max_avg"][0],
                              measure_data["derived_measures"]))
            ref_plot, = plt.plot(x_vals, y_vals, 'x', markersize=markersize,
                                 markeredgewidth=0.5,
                                 label=f'"{model_name}" - Trivial')

            # Rinast
            y_vals = list(map(lambda step_measure: step_measure["seq_rinast_length_min_max_avg"][0],
                              measure_data["derived_measures"]))
            plt.plot(x_vals, y_vals, '^', markersize=markersize, color=ref_plot.get_color(),
                     fillstyle='none', markeredgewidth=0.5, label=f'"{model_name}" - Rinast')

            # OC
            y_vals = list(map(lambda step_measure: max(
                step_measure["seq_oc_o_seq_c_fcs_lengths_min_max_avg"][1],
                step_measure["seq_oc_o_seq_c_mcs_lengths_min_max_avg"][1],
                step_measure["seq_oc_o_seq_c_rcs_lengths_min_max_avg"][1],
                step_measure["seq_oc_o_dbm_c_fcs_lengths_min_max_avg"][1],
                step_measure["seq_oc_o_dbm_c_mcs_lengths_min_max_avg"][1],
                step_measure["seq_oc_o_dbm_c_rcs_lengths_min_max_avg"][1],
            ), measure_data["derived_measures"]))
            plt.plot(x_vals, y_vals, 'o', markersize=markersize, color=ref_plot.get_color(),
                     fillstyle='none', markeredgewidth=0.5,
                     label=f'"{model_name}" - OC')

            clock_count = measure_data["model_details"]["clock_count"]
            max_seq_constr_length = clock_count * 2 + ((clock_count + 1) ** 2 - (clock_count + 1))
            plt.plot([0, x_vals[-1]], [max_seq_constr_length, max_seq_constr_length], '-', linewidth=0.5,
                     color=ref_plot.get_color())

        plt.xlabel('simulated steps')
        plt.ylabel('sequence length')
        plt.legend(bbox_to_anchor=(0, 1.0), loc='upper left', ncol=3, prop=font_p)
        plt.grid(which='both', alpha=0.5)
        # ax.set_xticks(range(0,x_vals[-1]))
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        if show_plot:
            plt.show()
        if save_plot:
            output_file_path = f'{output_folder}/seq-csc-length-after-n-steps.png'
            pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file_path, dpi=300)
            print(f'File "{output_file_path}" saved.')

    #########################################
    # Graph: Generation times after n steps #
    #########################################
    def create_plot_gen_time_after_every_step(
            self, all_measures=None, save_plot=False, show_plot=False):
        """Creates a plot from the sequence reconstruction results obtained via model simulations.

        Args:
            all_measures: All reconstruction measure data.
            save_plot: Choose whether the generated plot should be saved.
            show_plot: Choose whether the generated plot should be shown.
        """
        input_folder = os.path.join(self.data_base_folder, 'test_ssr_on_models_at_every_step')
        output_folder = os.path.join(self.output_base_folder, 'graphs')
        if all_measures is None:
            all_measures = load_all_model_data_from_folder(input_folder)

        plt.clf()
        dpi = 300
        plt.figure(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi)

        markersize = 2
        font_p = FontProperties()
        font_p.set_size('small')

        # plt.subplots_adjust(hspace=1, top=0.6, bottom=0.2)
        ax = plt.subplot(1, 1, 1)

        tracked_steps = list(all_measures.values())[0]["all_runs_measures"]["tracked_steps"]
        for model_name, measure_data in all_measures.items():
            x_vals = tracked_steps

            y_vals = list(map(lambda step_measure: max(
                step_measure["seq_approx_dbm_gen_time_min_max_avg"][1] +
                step_measure["seq_approx_dbm_app_time_min_max_avg"][1],
                step_measure["seq_approx_seq_gen_time_min_max_avg"][1] +
                step_measure["seq_approx_seq_app_time_min_max_avg"][1],
            ), measure_data["derived_measures"]))
            ref_plot, = plt.plot(x_vals, y_vals, 'o', markersize=markersize, fillstyle='none',
                                 markeredgewidth=0.5,
                                 label=f'"{model_name}" - o-phase')

            y_vals = list(map(lambda step_measure: max(
                step_measure["seq_constr_fcs_gen_time_min_max_avg"][1] +
                step_measure["seq_constr_fcs_app_time_min_max_avg"][1],
                step_measure["seq_constr_mcs_gen_time_min_max_avg"][1] +
                step_measure["seq_constr_mcs_app_time_min_max_avg"][1],
                step_measure["seq_constr_rcs_gen_time_min_max_avg"][1] +
                step_measure["seq_constr_rcs_app_time_min_max_avg"][1],
            ), measure_data["derived_measures"]))
            plt.plot(x_vals, y_vals, 'x', markersize=markersize, color=ref_plot.get_color(),
                     markeredgewidth=0.5,
                     label=f'"{model_name}" - c-phase')

        plt.xlabel('simulated steps')
        plt.ylabel('maximum gen. + app. time [s]')
        # plt.yscale("log")
        # plt.legend(bbox_to_anchor=(0,1.8), loc='upper left', ncol=2, prop=fontP)
        plt.grid(which='both', alpha=0.5)
        # ax.set_xticks(range(0,x_vals[-1]))
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        if show_plot:
            plt.show()
        if save_plot:
            output_file_path = f'{output_folder}/gen-time-after-n-steps.png'
            pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file_path, dpi=300)
            print(f'File "{output_file_path}" saved.')

    ################################################################################
    # Random Sequences #
    ################################################################################

    ####################
    # Sequence lengths #
    ####################

    #######################################################
    # Graph: Seq_approx length for varying seq_ref length #
    #######################################################
    def create_plot_seq_approx_length_for_varying_seq_ref_length(
            self, all_measures=None, save_plot=False, show_plot=False):
        """Creates a plot from the sequence reconstruction results obtained via randomly generated sequences.

        Args:
            all_measures: All reconstruction measure data.
            save_plot: Choose whether the generated plot should be saved.
            show_plot: Choose whether the generated plot should be shown.
        """
        input_folder = os.path.join(self.data_base_folder, 'test_ssr_on_random_sequences_with_varying_lengths')
        output_folder = os.path.join(self.output_base_folder, 'graphs')
        data_file_path = os.path.join(input_folder, 'data.json')
        if all_measures is None:
            with open(data_file_path, 'r') as file:
                all_measures = json.load(file)
        plt.clf()

        dpi = 300
        plt.figure(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi)

        markersize = 2
        font_p = FontProperties()
        font_p.set_size('small')

        # plt.subplots_adjust(hspace=1, top=0.8)
        ax = plt.subplot(1, 1, 1)
        x_vals = np.asarray(list(map(lambda x: x["sequence_length"], all_measures)))

        min_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_approx_dbm_length_min_max_avg"][0],
                all_measures)))
        max_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_approx_dbm_length_min_max_avg"][1],
                all_measures)))
        mean_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_approx_dbm_length_min_max_avg"][2],
                all_measures)))
        plt.errorbar(x_vals - 3, mean_vals, [mean_vals - min_vals, max_vals - mean_vals], fmt='ok', lw=1,
                     markersize=markersize, markerfacecolor="red", markeredgecolor="red", ecolor="black", elinewidth=1,
                     capsize=2, label="O(DBM)")

        min_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_approx_seq_length_min_max_avg"][0],
                all_measures)))
        max_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_approx_seq_length_min_max_avg"][1],
                all_measures)))
        mean_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_approx_seq_length_min_max_avg"][2],
                all_measures)))
        plt.errorbar(x_vals, mean_vals, [mean_vals - min_vals, max_vals - mean_vals], fmt='ok', lw=1,
                     markersize=markersize, markerfacecolor="blue", markeredgecolor="blue", ecolor="black",
                     elinewidth=1,
                     capsize=2, label="O(SEQ)")

        plt.xlabel('orig. sequence length')
        plt.ylabel('reduced approx. sequence length')
        plt.legend(bbox_to_anchor=(0, 1.1), loc='upper left', ncol=2, prop=font_p)
        plt.grid(which='both', alpha=0.5)
        ax.set_xticks(x_vals[::2])
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        clock_count = all_measures[0]["clock_count"]
        max_seq_constr_length = 2 * clock_count
        plt.plot([0, x_vals[-1]], [max_seq_constr_length, max_seq_constr_length], '--', linewidth=1, color='black')

        plt.tight_layout()
        if show_plot:
            plt.show()
        if save_plot:
            output_file_path = f'{output_folder}/seq-approx-length-for-varying-seq-ref-length.png'
            pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file_path, dpi=300)
            print(f'File "{output_file_path}" saved.')

    ###############################################
    # Graph: Seq_approx length for varying clocks #
    ###############################################
    def create_plot_seq_approx_length_for_varying_clocks(
            self, all_measures=None, save_plot=False, show_plot=False):
        """Creates a plot from the sequence reconstruction results obtained via randomly generated sequences.

        Args:
            all_measures: All reconstruction measure data.
            save_plot: Choose whether the generated plot should be saved.
            show_plot: Choose whether the generated plot should be shown.
        """
        input_folder = os.path.join(self.data_base_folder, 'test_ssr_on_random_sequences_with_varying_clocks')
        output_folder = os.path.join(self.output_base_folder, 'graphs')
        data_file_path = os.path.join(input_folder, 'data.json')
        if all_measures is None:
            with open(data_file_path, 'r') as file:
                all_measures = json.load(file)
        plt.clf()

        dpi = 300
        plt.figure(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi)

        markersize = 2
        font_p = FontProperties()
        font_p.set_size('small')

        # plt.subplots_adjust(hspace=1, top=0.8)
        ax = plt.subplot(1, 1, 1)
        x_vals = np.asarray(list(map(lambda x: x["clock_count"], all_measures)))

        min_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_approx_dbm_length_min_max_avg"][0],
                all_measures)))
        max_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_approx_dbm_length_min_max_avg"][1],
                all_measures)))
        mean_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_approx_dbm_length_min_max_avg"][2],
                all_measures)))
        plt.errorbar(x_vals - 0.05, mean_vals, [mean_vals - min_vals, max_vals - mean_vals], fmt='ok', lw=1,
                     markersize=markersize, markerfacecolor="red", markeredgecolor="red", ecolor="black", elinewidth=1,
                     capsize=2, label="O(DBM)")

        min_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_approx_seq_length_min_max_avg"][0],
                all_measures)))
        max_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_approx_seq_length_min_max_avg"][1],
                all_measures)))
        mean_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_approx_seq_length_min_max_avg"][2],
                all_measures)))
        plt.errorbar(x_vals, mean_vals, [mean_vals - min_vals, max_vals - mean_vals], fmt='ok', lw=1,
                     markersize=markersize, markerfacecolor="blue", markeredgecolor="blue", ecolor="black",
                     elinewidth=1,
                     capsize=2, label="O(SEQ)")

        plt.xlabel('clock count')
        plt.ylabel('reduced approx. sequence length')
        plt.legend(bbox_to_anchor=(0, 1.1), loc='upper left', ncol=2, prop=font_p)
        plt.grid(which='both', alpha=0.5)
        ax.set_xticks(x_vals[::2])
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        for clock_count in x_vals:
            max_seq_constr_length = 2 * clock_count
            plt.plot([clock_count - 0.5, clock_count + 0.5], [max_seq_constr_length, max_seq_constr_length], '--',
                     linewidth=1, color='black')

        plt.tight_layout()
        if show_plot:
            plt.show()
        if save_plot:
            output_file_path = f'{output_folder}/seq-approx-length-for-varying-clocks.png'
            pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file_path, dpi=300)
            print(f'File "{output_file_path}" saved.')

    #######################################################
    # Graph: Seq_constr length for varying seq_ref length #
    #######################################################
    def create_plot_seq_constr_length_for_varying_seq_ref_length(
            self, all_measures=None, save_plot=False, show_plot=False):
        """Creates a plot from the sequence reconstruction results obtained via randomly generated sequences.

        Args:
            all_measures: All reconstruction measure data.
            save_plot: Choose whether the generated plot should be saved.
            show_plot: Choose whether the generated plot should be shown.
        """
        input_folder = os.path.join(self.data_base_folder, 'test_ssr_on_random_sequences_with_varying_lengths')
        output_folder = os.path.join(self.output_base_folder, 'graphs')
        data_file_path = os.path.join(input_folder, 'data.json')
        if all_measures is None:
            with open(data_file_path, 'r') as file:
                all_measures = json.load(file)
        plt.clf()

        dpi = 300
        plt.figure(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi)

        markersize = 2
        font_p = FontProperties()
        font_p.set_size('small')

        # plt.subplots_adjust(hspace=1, top=0.8)
        ax = plt.subplot(1, 1, 1)
        x_vals = np.asarray(list(map(lambda x: x["sequence_length"], all_measures)))

        min_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_constr_fcs_length_min_max_avg"][0],
                all_measures)))
        max_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_constr_fcs_length_min_max_avg"][1],
                all_measures)))
        mean_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_constr_fcs_length_min_max_avg"][2],
                all_measures)))
        plt.errorbar(x_vals - 3, mean_vals, [mean_vals - min_vals, max_vals - mean_vals], fmt='ok', lw=1,
                     markersize=markersize, markerfacecolor="red", markeredgecolor="red", ecolor="black", elinewidth=1,
                     capsize=2, label="C(FCS)")

        min_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_constr_mcs_length_min_max_avg"][0],
                all_measures)))
        max_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_constr_mcs_length_min_max_avg"][1],
                all_measures)))
        mean_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_constr_mcs_length_min_max_avg"][2],
                all_measures)))
        plt.errorbar(x_vals, mean_vals, [mean_vals - min_vals, max_vals - mean_vals], fmt='ok', lw=1,
                     markersize=markersize, markerfacecolor="blue", markeredgecolor="blue", ecolor="black",
                     elinewidth=1,
                     capsize=2, label="C(MCS)")

        min_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_constr_rcs_length_min_max_avg"][0],
                all_measures)))
        max_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_constr_rcs_length_min_max_avg"][1],
                all_measures)))
        mean_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_constr_rcs_length_min_max_avg"][2],
                all_measures)))
        plt.errorbar(x_vals + 3, mean_vals, [mean_vals - min_vals, max_vals - mean_vals], fmt='ok', lw=1,
                     markersize=markersize, markerfacecolor="green", markeredgecolor="green", ecolor="black",
                     elinewidth=1,
                     capsize=2, label="C(RCS)")

        plt.xlabel('original sequence length')
        plt.ylabel('reduced constr. sequence length')
        plt.legend(bbox_to_anchor=(0, 1.1), loc='upper left', ncol=3, prop=font_p)
        plt.grid(which='both', alpha=0.5)
        ax.set_xticks(x_vals[::2])
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        clock_count = all_measures[0]["clock_count"]
        max_seq_constr_length = ((clock_count + 1) ** 2 - (clock_count + 1))
        plt.plot([0, x_vals[-1]], [max_seq_constr_length, max_seq_constr_length], '--', linewidth=1, color='black')

        plt.tight_layout()
        if show_plot:
            plt.show()
        if save_plot:
            output_file_path = f'{output_folder}/seq-constr-length-for-varying-seq-ref-length.png'
            pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file_path, dpi=300)
            print(f'File "{output_file_path}" saved.')

    ###############################################
    # Graph: Seq_constr length for varying clocks #
    ###############################################
    def create_plot_seq_constr_length_for_varying_clocks(
            self, all_measures=None, save_plot=False, show_plot=False):
        """Creates a plot from the sequence reconstruction results obtained via randomly generated sequences.

        Args:
            all_measures: All reconstruction measure data.
            save_plot: Choose whether the generated plot should be saved.
            show_plot: Choose whether the generated plot should be shown.
        """
        input_folder = os.path.join(self.data_base_folder, 'test_ssr_on_random_sequences_with_varying_clocks')
        output_folder = os.path.join(self.output_base_folder, 'graphs')
        data_file_path = os.path.join(input_folder, 'data.json')
        if all_measures is None:
            with open(data_file_path, 'r') as file:
                all_measures = json.load(file)
        plt.clf()

        dpi = 300
        plt.figure(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi)

        markersize = 2
        font_p = FontProperties()
        font_p.set_size('small')

        # plt.subplots_adjust(hspace=1, top=0.8)
        ax = plt.subplot(1, 1, 1)
        x_vals = np.asarray(list(map(lambda x: x["clock_count"], all_measures)))

        min_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_constr_fcs_length_min_max_avg"][0],
                all_measures)))
        max_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_constr_fcs_length_min_max_avg"][1],
                all_measures)))
        mean_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_constr_fcs_length_min_max_avg"][2],
                all_measures)))
        plt.errorbar(x_vals - 0.05, mean_vals, [mean_vals - min_vals, max_vals - mean_vals], fmt='ok', lw=1,
                     markersize=markersize, markerfacecolor="red", markeredgecolor="red", ecolor="black", elinewidth=1,
                     capsize=2, label="C(FCS)")

        min_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_constr_mcs_length_min_max_avg"][0],
                all_measures)))
        max_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_constr_mcs_length_min_max_avg"][1],
                all_measures)))
        mean_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_constr_mcs_length_min_max_avg"][2],
                all_measures)))
        plt.errorbar(x_vals, mean_vals, [mean_vals - min_vals, max_vals - mean_vals], fmt='ok', lw=1,
                     markersize=markersize, markerfacecolor="blue", markeredgecolor="blue", ecolor="black",
                     elinewidth=1,
                     capsize=2, label="C(MCS)")

        min_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_constr_rcs_length_min_max_avg"][0],
                all_measures)))
        max_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_constr_rcs_length_min_max_avg"][1],
                all_measures)))
        mean_vals = np.asarray(list(
            map(lambda x: x["all_runs_measures"]["derived_measures"]["seq_constr_rcs_length_min_max_avg"][2],
                all_measures)))
        plt.errorbar(x_vals + 0.05, mean_vals, [mean_vals - min_vals, max_vals - mean_vals], fmt='ok', lw=1,
                     markersize=markersize, markerfacecolor="green", markeredgecolor="green", ecolor="black",
                     elinewidth=1,
                     capsize=2, label="C(RCS)")

        plt.xlabel('clock count')
        plt.ylabel('reduced constr. sequence length')
        plt.legend(bbox_to_anchor=(0, 1.1), loc='upper left', ncol=3, prop=font_p)
        plt.grid(which='both', alpha=0.5)
        ax.set_xticks(x_vals)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        for clock_count in x_vals:
            max_seq_constr_length = ((clock_count + 1) ** 2 - (clock_count + 1))
            plt.plot([clock_count - 0.5, clock_count + 0.5], [max_seq_constr_length, max_seq_constr_length], '--',
                     linewidth=1, color='black')

        plt.tight_layout()
        if show_plot:
            plt.show()
        if save_plot:
            output_file_path = f'{output_folder}/seq-constr-length-for-varying-clocks.png'
            pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file_path, dpi=300)
            print(f'File "{output_file_path}" saved.')

    ######################
    # Construction times #
    ######################

    #######################################################
    # Graph: Seq_approx time for varying seq_ref length #
    #######################################################
    def create_plot_seq_approx_time_for_varying_seq_ref_length(
            self, all_measures=None, save_plot=False, show_plot=False):
        """Creates a plot from the sequence reconstruction results obtained via randomly generated sequences.

        Args:
            all_measures: All reconstruction measure data.
            save_plot: Choose whether the generated plot should be saved.
            show_plot: Choose whether the generated plot should be shown.
        """
        input_folder = os.path.join(self.data_base_folder, 'test_ssr_on_random_sequences_with_varying_lengths')
        output_folder = os.path.join(self.output_base_folder, 'graphs')
        data_file_path = os.path.join(input_folder, 'data.json')
        if all_measures is None:
            with open(data_file_path, 'r') as file:
                all_measures = json.load(file)
        plt.clf()

        dpi = 300
        plt.figure(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi)

        markersize = 2
        font_p = FontProperties()
        font_p.set_size('small')

        # plt.subplots_adjust(hspace=1, top=0.8)
        ax = plt.subplot(1, 1, 1)
        x_vals = np.asarray(list(map(lambda x: x["sequence_length"], all_measures)))

        min_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_approx_dbm_gen_time_min_max_avg"][0] +
                x["all_runs_measures"]["derived_measures"]["seq_approx_dbm_app_time_min_max_avg"][0],
                all_measures)))
        max_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_approx_dbm_gen_time_min_max_avg"][1] +
                x["all_runs_measures"]["derived_measures"]["seq_approx_dbm_app_time_min_max_avg"][1],
                all_measures)))
        mean_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_approx_dbm_gen_time_min_max_avg"][2] +
                x["all_runs_measures"]["derived_measures"]["seq_approx_dbm_app_time_min_max_avg"][2],
                all_measures)))
        plt.errorbar(x_vals - 3, mean_vals, [mean_vals - min_vals, max_vals - mean_vals], fmt='ok', lw=1,
                     markersize=markersize, markerfacecolor="red", markeredgecolor="red", ecolor="black", elinewidth=1,
                     capsize=2, label="O(DBM)")

        min_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_approx_seq_gen_time_min_max_avg"][0] +
                x["all_runs_measures"]["derived_measures"]["seq_approx_seq_app_time_min_max_avg"][0],
                all_measures)))
        max_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_approx_seq_gen_time_min_max_avg"][1] +
                x["all_runs_measures"]["derived_measures"]["seq_approx_seq_app_time_min_max_avg"][1],
                all_measures)))
        mean_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_approx_seq_gen_time_min_max_avg"][2] +
                x["all_runs_measures"]["derived_measures"]["seq_approx_seq_app_time_min_max_avg"][2],
                all_measures)))
        plt.errorbar(x_vals, mean_vals, [mean_vals - min_vals, max_vals - mean_vals], fmt='ok', lw=1,
                     markersize=markersize, markerfacecolor="blue", markeredgecolor="blue", ecolor="black",
                     elinewidth=1,
                     capsize=2, label="O(SEQ)")

        plt.xlabel('orig. sequence length')
        plt.ylabel('construction time [s]')
        plt.legend(bbox_to_anchor=(0, 1.1), loc='upper left', ncol=2, prop=font_p)
        plt.grid(which='both', alpha=0.5)
        ax.set_xticks(x_vals[::2])
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        if show_plot:
            plt.show()
        if save_plot:
            output_file_path = f'{output_folder}/seq-approx-time-for-varying-seq-ref-length.png'
            pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file_path, dpi=300)
            print(f'File "{output_file_path}" saved.')

    ###############################################
    # Graph: Seq_approx time for varying clocks #
    ###############################################
    def create_plot_seq_approx_time_for_varying_clocks(
            self, all_measures=None, save_plot=False, show_plot=False):
        """Creates a plot from the sequence reconstruction results obtained via randomly generated sequences.

        Args:
            all_measures: All reconstruction measure data.
            save_plot: Choose whether the generated plot should be saved.
            show_plot: Choose whether the generated plot should be shown.
        """
        input_folder = os.path.join(self.data_base_folder, 'test_ssr_on_random_sequences_with_varying_clocks')
        output_folder = os.path.join(self.output_base_folder, 'graphs')
        data_file_path = os.path.join(input_folder, 'data.json')
        if all_measures is None:
            with open(data_file_path, 'r') as file:
                all_measures = json.load(file)
        plt.clf()

        dpi = 300
        plt.figure(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi)

        markersize = 2
        font_p = FontProperties()
        font_p.set_size('small')

        # plt.subplots_adjust(hspace=1, top=0.8)
        ax = plt.subplot(1, 1, 1)
        x_vals = np.asarray(list(map(lambda x: x["clock_count"], all_measures)))

        min_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_approx_dbm_gen_time_min_max_avg"][0] +
                x["all_runs_measures"]["derived_measures"]["seq_approx_dbm_app_time_min_max_avg"][0],
                all_measures)))
        max_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_approx_dbm_gen_time_min_max_avg"][1] +
                x["all_runs_measures"]["derived_measures"]["seq_approx_dbm_app_time_min_max_avg"][1],
                all_measures)))
        mean_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_approx_dbm_gen_time_min_max_avg"][2] +
                x["all_runs_measures"]["derived_measures"]["seq_approx_dbm_app_time_min_max_avg"][2],
                all_measures)))
        plt.errorbar(x_vals - 0.05, mean_vals, [mean_vals - min_vals, max_vals - mean_vals], fmt='ok', lw=1,
                     markersize=markersize, markerfacecolor="red", markeredgecolor="red", ecolor="black", elinewidth=1,
                     capsize=2, label="O(DBM)")

        min_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_approx_seq_gen_time_min_max_avg"][0] +
                x["all_runs_measures"]["derived_measures"]["seq_approx_seq_app_time_min_max_avg"][0],
                all_measures)))
        max_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_approx_seq_gen_time_min_max_avg"][1] +
                x["all_runs_measures"]["derived_measures"]["seq_approx_seq_app_time_min_max_avg"][1],
                all_measures)))
        mean_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_approx_seq_gen_time_min_max_avg"][2] +
                x["all_runs_measures"]["derived_measures"]["seq_approx_seq_app_time_min_max_avg"][2],
                all_measures)))
        plt.errorbar(x_vals, mean_vals, [mean_vals - min_vals, max_vals - mean_vals], fmt='ok', lw=1,
                     markersize=markersize, markerfacecolor="blue", markeredgecolor="blue", ecolor="black",
                     elinewidth=1,
                     capsize=2, label="O(SEQ)")

        plt.xlabel('clock count')
        plt.ylabel('construction time [s]')
        plt.legend(bbox_to_anchor=(0, 1.1), loc='upper left', ncol=2, prop=font_p)
        plt.grid(which='both', alpha=0.5)
        ax.set_xticks(x_vals[::2])
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        if show_plot:
            plt.show()
        if save_plot:
            output_file_path = f'{output_folder}/seq-approx-time-for-varying-clocks.png'
            pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file_path, dpi=300)
            print(f'File "{output_file_path}" saved.')

    #######################################################
    # Graph: Seq_constr time for varying seq_ref length #
    #######################################################
    def create_plot_seq_constr_time_for_varying_seq_ref_length(
            self, all_measures=None, save_plot=False, show_plot=False):
        """Creates a plot from the sequence reconstruction results obtained via randomly generated sequences.

        Args:
            all_measures: All reconstruction measure data.
            save_plot: Choose whether the generated plot should be saved.
            show_plot: Choose whether the generated plot should be shown.
        """
        input_folder = os.path.join(self.data_base_folder, 'test_ssr_on_random_sequences_with_varying_lengths')
        output_folder = os.path.join(self.output_base_folder, 'graphs')
        data_file_path = os.path.join(input_folder, 'data.json')
        if all_measures is None:
            with open(data_file_path, 'r') as file:
                all_measures = json.load(file)
        plt.clf()

        dpi = 300
        plt.figure(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi)

        markersize = 2
        font_p = FontProperties()
        font_p.set_size('small')

        # plt.subplots_adjust(hspace=1, top=0.8)
        ax = plt.subplot(1, 1, 1)
        x_vals = np.asarray(list(map(lambda x: x["sequence_length"], all_measures)))

        min_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_constr_fcs_gen_time_min_max_avg"][0] +
                x["all_runs_measures"]["derived_measures"]["seq_constr_fcs_app_time_min_max_avg"][0],
                all_measures)))
        max_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_constr_fcs_gen_time_min_max_avg"][1] +
                x["all_runs_measures"]["derived_measures"]["seq_constr_fcs_app_time_min_max_avg"][1],
                all_measures)))
        mean_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_constr_fcs_gen_time_min_max_avg"][2] +
                x["all_runs_measures"]["derived_measures"]["seq_constr_fcs_app_time_min_max_avg"][2],
                all_measures)))
        plt.errorbar(x_vals - 3, mean_vals, [mean_vals - min_vals, max_vals - mean_vals], fmt='ok', lw=1,
                     markersize=markersize, markerfacecolor="red", markeredgecolor="red", ecolor="black", elinewidth=1,
                     capsize=2, label="C(FCS)")

        min_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_constr_mcs_gen_time_min_max_avg"][0] +
                x["all_runs_measures"]["derived_measures"]["seq_constr_mcs_app_time_min_max_avg"][0],
                all_measures)))
        max_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_constr_mcs_gen_time_min_max_avg"][1] +
                x["all_runs_measures"]["derived_measures"]["seq_constr_mcs_app_time_min_max_avg"][1],
                all_measures)))
        mean_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_constr_mcs_gen_time_min_max_avg"][2] +
                x["all_runs_measures"]["derived_measures"]["seq_constr_mcs_app_time_min_max_avg"][2],
                all_measures)))
        plt.errorbar(x_vals, mean_vals, [mean_vals - min_vals, max_vals - mean_vals], fmt='ok', lw=1,
                     markersize=markersize, markerfacecolor="blue", markeredgecolor="blue", ecolor="black",
                     elinewidth=1,
                     capsize=2, label="C(MCS)")

        min_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_constr_rcs_gen_time_min_max_avg"][0] +
                x["all_runs_measures"]["derived_measures"]["seq_constr_rcs_app_time_min_max_avg"][0],
                all_measures)))
        max_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_constr_rcs_gen_time_min_max_avg"][1] +
                x["all_runs_measures"]["derived_measures"]["seq_constr_rcs_app_time_min_max_avg"][1],
                all_measures)))
        mean_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_constr_rcs_gen_time_min_max_avg"][2] +
                x["all_runs_measures"]["derived_measures"]["seq_constr_rcs_app_time_min_max_avg"][2],
                all_measures)))
        plt.errorbar(x_vals + 3, mean_vals, [mean_vals - min_vals, max_vals - mean_vals], fmt='ok', lw=1,
                     markersize=markersize, markerfacecolor="green", markeredgecolor="green", ecolor="black",
                     elinewidth=1,
                     capsize=2, label="C(RCS)")

        plt.xlabel('original sequence length')
        plt.ylabel('construction time [s]')
        plt.legend(bbox_to_anchor=(0, 1.1), loc='upper left', ncol=3, prop=font_p)
        plt.grid(which='both', alpha=0.5)
        ax.set_xticks(x_vals[::2])
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        if show_plot:
            plt.show()
        if save_plot:
            output_file_path = f'{output_folder}/seq-constr-time-for-varying-seq-ref-length.png'
            pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file_path, dpi=300)
            print(f'File "{output_file_path}" saved.')

    #############################################
    # Graph: Seq_constr time for varying clocks #
    #############################################
    def create_plot_seq_constr_time_for_varying_clocks(
            self, all_measures=None, save_plot=False, show_plot=False):
        """Creates a plot from the sequence reconstruction results obtained via randomly generated sequences.

        Args:
            all_measures: All reconstruction measure data.
            save_plot: Choose whether the generated plot should be saved.
            show_plot: Choose whether the generated plot should be shown.
        """
        input_folder = os.path.join(self.data_base_folder, 'test_ssr_on_random_sequences_with_varying_clocks')
        output_folder = os.path.join(self.output_base_folder, 'graphs')
        data_file_path = os.path.join(input_folder, 'data.json')
        if all_measures is None:
            with open(data_file_path, 'r') as file:
                all_measures = json.load(file)
        plt.clf()

        dpi = 300
        plt.figure(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi)

        markersize = 2
        font_p = FontProperties()
        font_p.set_size('small')

        # plt.subplots_adjust(hspace=1, top=0.8)
        ax = plt.subplot(1, 1, 1)
        x_vals = np.asarray(list(map(lambda x: x["clock_count"], all_measures)))

        min_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_constr_fcs_gen_time_min_max_avg"][0] +
                x["all_runs_measures"]["derived_measures"]["seq_constr_fcs_app_time_min_max_avg"][0],
                all_measures)))
        max_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_constr_fcs_gen_time_min_max_avg"][1] +
                x["all_runs_measures"]["derived_measures"]["seq_constr_fcs_app_time_min_max_avg"][1],
                all_measures)))
        mean_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_constr_fcs_gen_time_min_max_avg"][2] +
                x["all_runs_measures"]["derived_measures"]["seq_constr_fcs_app_time_min_max_avg"][2],
                all_measures)))
        plt.errorbar(x_vals - 0.05, mean_vals, [mean_vals - min_vals, max_vals - mean_vals], fmt='ok', lw=1,
                     markersize=markersize, markerfacecolor="red", markeredgecolor="red", ecolor="black", elinewidth=1,
                     capsize=2, label="C(FCS)")

        min_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_constr_mcs_gen_time_min_max_avg"][0] +
                x["all_runs_measures"]["derived_measures"]["seq_constr_mcs_app_time_min_max_avg"][0],
                all_measures)))
        max_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_constr_mcs_gen_time_min_max_avg"][1] +
                x["all_runs_measures"]["derived_measures"]["seq_constr_mcs_app_time_min_max_avg"][1],
                all_measures)))
        mean_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_constr_mcs_gen_time_min_max_avg"][2] +
                x["all_runs_measures"]["derived_measures"]["seq_constr_mcs_app_time_min_max_avg"][2],
                all_measures)))
        plt.errorbar(x_vals, mean_vals, [mean_vals - min_vals, max_vals - mean_vals], fmt='ok', lw=1,
                     markersize=markersize, markerfacecolor="blue", markeredgecolor="blue", ecolor="black",
                     elinewidth=1,
                     capsize=2, label="C(MCS)")

        min_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_constr_rcs_gen_time_min_max_avg"][0] +
                x["all_runs_measures"]["derived_measures"]["seq_constr_rcs_app_time_min_max_avg"][0],
                all_measures)))
        max_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_constr_rcs_gen_time_min_max_avg"][1] +
                x["all_runs_measures"]["derived_measures"]["seq_constr_rcs_app_time_min_max_avg"][1],
                all_measures)))
        mean_vals = np.asarray(list(
            map(lambda x:
                x["all_runs_measures"]["derived_measures"]["seq_constr_rcs_gen_time_min_max_avg"][2] +
                x["all_runs_measures"]["derived_measures"]["seq_constr_rcs_app_time_min_max_avg"][2],
                all_measures)))
        plt.errorbar(x_vals + 0.05, mean_vals, [mean_vals - min_vals, max_vals - mean_vals], fmt='ok', lw=1,
                     markersize=markersize, markerfacecolor="green", markeredgecolor="green", ecolor="black",
                     elinewidth=1,
                     capsize=2, label="C(RCS)")

        plt.xlabel('clock count')
        plt.ylabel('construction time [s]')
        plt.legend(bbox_to_anchor=(0, 1.1), loc='upper left', ncol=3, prop=font_p)
        plt.grid(which='both', alpha=0.5)
        ax.set_xticks(x_vals)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        if show_plot:
            plt.show()
        if save_plot:
            output_file_path = f'{output_folder}/seq-constr-time-for-varying-clocks.png'
            pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file_path, dpi=300)
            print(f'File "{output_file_path}" saved.')
