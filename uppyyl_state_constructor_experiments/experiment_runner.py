"""The main entry point of the Uppyyl state constructor experiments module."""
from uppyyl_state_constructor_experiments.backend.experiments.approach_examples.approach_examples import \
    experiment_approach_examples
from uppyyl_state_constructor_experiments.backend.experiments.introduction_example.introduction_example import \
    experiment_introduction_example
from uppyyl_state_constructor_experiments.backend.experiments.review.review_experiments import Experiments

def main():
    """The main function."""
    experiments = Experiments()

    #########################
    # Model-based experiments
    #########################
    # experiments.state_construction_on_models_at_selected_steps()
    # experiments.state_construction_on_models_at_every_step()

    ###################################
    # Random-sequence-based experiments
    ###################################
    # experiments.state_construction_on_random_sequences_with_varying_lengths()
    # experiments.state_construction_on_random_sequences_with_varying_clocks()

    ####################################
    # Introduction and approach examples
    ####################################
    # experiment_introduction_example()
    experiment_approach_examples()

if __name__ == '__main__':
    main()
