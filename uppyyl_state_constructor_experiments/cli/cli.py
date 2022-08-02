"""This module implements a CLI for the Uppyyl state constructor experiments."""
import glob
import os
import readline
import shlex
from cmd import Cmd
from enum import Enum

from colorama import Fore

from uppyyl_state_constructor_experiments.backend.experiments.approach_examples.approach_examples import experiment_approach_examples
from uppyyl_state_constructor_experiments.backend.experiments.introduction_example.introduction_example import experiment_introduction_example
from uppyyl_state_constructor_experiments.backend.experiments.state_construction.article_experiments import Experiments
from uppyyl_state_constructor_experiments.backend.experiments.state_construction.plots import Plots


class View(Enum):
    """An enum for the possible graphical views."""
    EXPERIMENTS = 1
    PLOTS = 2


readline.set_completer_delims(' \t\n')


def _complete_path(path):
    if os.path.isdir(path):
        return False, glob.glob(os.path.join(path, '*'))
    else:
        return True, glob.glob(path + '*')


default_data_folder = f'../logs/default'


############################################
# Uppyyl State Constructor Experiments CLI #
############################################
class UppyylStateConstructorExperimentsCLI(Cmd):
    """A CLI for the Uppyyl State Constructor Experiments."""

    prompt = 'state_constructor_experiments> '
    intro = ""

    def __init__(self):
        """Initializes UppyylStateConstructorExperimentsCLI."""
        super(UppyylStateConstructorExperimentsCLI, self).__init__()

        self.experiments = Experiments()
        self.all_experiment_data = {
            "exp.sim_seq.every_step": {
                "function": self.experiments.state_construction_on_models_at_every_step,
                "path": "./experiments/state_construction/test_experiments.py"
                        "::test_state_construction_on_models_at_every_step",
                "description": "Performs state construction for simulated models at every step",
            },
            "exp.gen_seq.var_clocks": {
                "function": self.experiments.state_construction_on_random_sequences_with_varying_clocks,
                "path": "./experiments/state_construction/test_experiments.py"
                        "::test_state_construction_on_random_sequences_with_varying_clocks",
                "description": "Performs state construction for generated sequences with varying clock counts",
            },
            "exp.gen_seq.var_lengths": {
                "function": self.experiments.state_construction_on_random_sequences_with_varying_lengths,
                "path": "./experiments/state_construction/test_experiments.py"
                        "::test_state_construction_on_random_sequences_with_varying_lengths",
                "description": "Performs state construction for generated sequences with varying input sequence length",
            },

            "exp.examples.introduction": {
                "function": experiment_introduction_example,
                "path": "./experiments/introduction_example/test_introduction_example.py"
                        "::experiment_introduction_example",
                "description": "Runs the introduction model example experiment",
            },
            "exp.examples.approaches": {
                "function": experiment_approach_examples,
                "path": "./experiments/approach_examples/test_approach_examples.py"
                        "::experiment_approach_examples",
                "description": "Runs the O- and C-phase approach example experiments",
            },

            "general.model_details": {
                "function": self.experiments.gather_model_details,
                "path": "./experiments/state_construction/test_experiments.py"
                        "::gather_model_details",
                "description": "Gathers details for all simulated models",
            },
        }

        self.plots = Plots()
        self.all_plot_data = {
            "plot.seq_csc_length_after_n_steps": {
                "function": self.plots.create_plot_seq_csc_length_after_every_step,
                "description": "Plots CSC sequence lengths after n steps"
            }
        }

        self.active_view = View.EXPERIMENTS

        self.experiments.output_base_folder = default_data_folder
        self.plots.data_base_folder = default_data_folder

        help_message = "These are the Uppyyl State Constructor experiments. Type ? to list commands."
        self.print_view(message=help_message)

    def print_view(self, message=""):
        """Prints the current view.

        Args:
            message: An optional information message.
        """
        if self.active_view == View.EXPERIMENTS:
            self.print_experiments_view(message=message)
        elif self.active_view == View.PLOTS:
            self.print_plots_view(message=message)

    def print_experiments_view(self, message=""):
        """Prints the experiments view.

        Args:
            message: An optional information message.
        """
        exp_string = self.get_experiments_string()

        self.clear()
        print(f'{Fore.BLUE}Message:{Fore.RESET} {message}')
        print(f'\n--| {Fore.BLUE}Available Experiments{Fore.RESET} |----------------------------')
        print(exp_string)
        print(f'')

    def get_experiments_string(self):
        """Constructs the string representation of the experiments.

        Returns:
            The experiments string.
        """
        experiment_descr_strs = []
        max_name_length = max(map(len, self.all_experiment_data.keys()))
        for i, (exp_name, exp_data) in enumerate(self.all_experiment_data.items()):
            experiment_descr_strs.append(f'{(exp_name + ":").ljust(max_name_length + 1)} {exp_data["description"]}')
        string = "\n".join(experiment_descr_strs)
        return string

    def print_plots_view(self, message=""):
        """Prints the plots view.

        Args:
            message: An optional information message.
        """
        plots_string = self.get_plots_string()

        self.clear()
        print(f'{Fore.BLUE}Message:{Fore.RESET} {message}')
        print(f'\n--| {Fore.BLUE}Available Plots{Fore.RESET} |----------------------------')
        print(plots_string)
        print(f'')

    def get_plots_string(self):
        """Constructs the string representation of the plots.

        Returns:
            The plots string.
        """
        plots_descr_strs = []
        max_name_length = max(map(len, self.all_plot_data.keys()))
        for i, (plot_name, plot_data) in enumerate(self.all_plot_data.items()):
            plots_descr_strs.append(f'{(plot_name + ":").ljust(max_name_length + 1)} {plot_data["description"]}')
        string = "\n".join(plots_descr_strs)
        return string

    def do_experiments(self, _):
        """Performs the "experiments" command."""
        self.active_view = View.EXPERIMENTS
        self.print_view(message=f'Switched to experiments view.')

    @staticmethod
    def help_experiments():
        """Shows help for the "experiments" command."""
        print('Switches to experiments view.')

    def do_plots(self, _):
        """Performs the "plots" command."""
        self.active_view = View.PLOTS
        self.print_view(message=f'Switched to plots view.')

    @staticmethod
    def help_plots():
        """Shows help for the "plots" command."""
        print('Switches to plots view.')

    def do_set_folder(self, arg):
        """Performs the "experiments" command."""
        folder = arg
        if not os.path.isdir(folder):
            self.print_view(message=f'{Fore.RED}"{folder}" is not a directory.{Fore.RESET}')
            return

        self.experiments.output_base_folder = folder
        self.plots.data_base_folder = folder
        self.print_view(message=f'Experiment data folder "{folder}" set successfully.')

    @staticmethod
    def help_set_folder():
        """Shows help for the "set_folder" command."""
        print('Sets the experiment data folder.')

    @staticmethod
    def complete_set_folder(text, _line, _start_idx, _end_idx):
        """Autocompletes the path argument of the "load" command."""
        if text[0] in ['"', "'"]:
            is_file, choices = _complete_path(text[1:])
            choices = list(map(lambda p: text[0] + p, choices))
            if len(choices) == 1 and is_file:
                choices[0] += text[0]
        else:
            is_file, choices = _complete_path(text)
        return choices

    def do_run(self, arg):
        """Performs the "run" command."""
        args = shlex.split(arg)
        exp_names = args
        selected_exps = {}
        if args:
            for exp_name in exp_names:
                try:
                    selected_exps[exp_name] = self.all_experiment_data[exp_name]
                except KeyError:
                    self.print_view(message=f'{Fore.RED}Experiment "{exp_name}" does not exist.{Fore.RESET}')
                    return
        else:
            selected_exps = self.all_experiment_data

        for expr_name, exp_data in selected_exps.items():
            exp_data["function"]()
            # test_exit_code = pytest.main([exp_data["path"], "--verbose", "-s"])
            # if test_exit_code != 0:
            #     raise Exception("Experiments did not execute successfully.")

        input("Experiment(s) successfully executed. Press Enter to return to menu ...")
        self.print_view(message=f'Experiments successfully executed.')

    @staticmethod
    def help_run():
        """Shows help for the "run" command."""
        print('Run a specific experiment (default: Run all experiments).')

    def complete_run(self, text, _line, _start_idx, _end_idx):
        """Autocompletes the experiment name argument of the "run" command."""
        choices = list(filter(lambda n: n.startswith(text), self.all_experiment_data.keys()))
        return choices

    def do_plot(self, arg):
        """Performs the "plot" command."""
        args = shlex.split(arg)
        plot_names = args
        selected_plots = {}
        if args:
            for plot_name in plot_names:
                try:
                    selected_plots[plot_name] = self.all_plot_data[plot_name]
                except KeyError:
                    self.print_view(message=f'{Fore.RED}Plot task "{plot_name}" does not exist.{Fore.RESET}')
                    return
        else:
            selected_plots = self.all_plot_data

        for plot_name, plot_data in selected_plots.items():
            plot_data["function"]()

        self.print_view(message=f'Plots successfully created.')

    @staticmethod
    def help_plot():
        """Shows help for the "run" command."""
        print('Creates a specific plot (default: Create all plots).')

    def complete_plot(self, text, _line, _start_idx, _end_idx):
        """Autocompletes the plot name argument of the "plot" command."""
        choices = list(filter(lambda n: n.startswith(text), self.all_plot_data.keys()))
        return choices

    @staticmethod
    def do_exit(_=None):
        """Performs the "exit" command."""
        return True

    @staticmethod
    def help_exit():
        """Shows help for the "exit" command."""
        print('exit the application. Shorthand: x q Ctrl-D.')

    @staticmethod
    def clear():
        """Clears the console."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def emptyline(self):
        """Performs an empty command."""
        self.print_view(message=f'')

    def default(self, inp):
        """Performs a "default" command (i.e., no specific "do_..." function exists)."""

        # Exit
        if inp == 'x' or inp == 'q':
            return self.do_exit()

        self.print_view(message=f'')

    do_EOF = do_exit
    help_EOF = help_exit
