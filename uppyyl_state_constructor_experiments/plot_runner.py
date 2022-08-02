"""The main entry point of the Uppyyl state constructor experiments plot module."""
from uppyyl_state_constructor_experiments.backend.experiments.review.plots import Plots

def main():
    """The main function."""
    plots = Plots()

    ###############
    # General plots
    ###############
    # plots.create_latex_table_model_details(show_plot=False, save_plot=True)

    ##############################
    # Model-based experiment plots
    ##############################
    # plots.create_latex_table_seq_csc_length_after_n_steps(show_plot=False, save_plot=True)
    # plots.create_plot_seq_csc_length_after_every_step(show_plot=False, save_plot=True)
    # plots.create_plot_gen_time_after_every_step(show_plot=False, save_plot=True)

    ########################################
    # Random-sequence-based experiment plots
    ########################################
    # plots.create_plot_seq_approx_time_for_varying_seq_ref_length(show_plot=False, save_plot=True)
    # plots.create_plot_seq_approx_length_for_varying_seq_ref_length(show_plot=False, save_plot=True)
    #
    plots.create_plot_seq_approx_time_for_varying_clocks(show_plot=False, save_plot=True)
    plots.create_plot_seq_approx_length_for_varying_clocks(show_plot=False, save_plot=True)
    #
    # plots.create_plot_seq_constr_time_for_varying_seq_ref_length(show_plot=False, save_plot=True)
    # plots.create_plot_seq_constr_length_for_varying_seq_ref_length(show_plot=False, save_plot=True)
    #
    plots.create_plot_seq_constr_time_for_varying_clocks(show_plot=False, save_plot=True)
    plots.create_plot_seq_constr_length_for_varying_clocks(show_plot=False, save_plot=True)


if __name__ == '__main__':
    main()
