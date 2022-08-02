"""This module provides all DBM state construction experiments."""

import pprint

import pytest

from uppyyl_state_constructor_experiments.backend.experiments.review.review_experiments import Experiments

pp = pprint.PrettyPrinter(indent=4, compact=True)
printExpectedResults = False
printActualResults = False


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


@pytest.fixture(scope="module")
def experiments():
    """A fixture for an Experiments instance.

    Returns:
        The Experiments instance.
    """
    return Experiments()


################################################################################
# Experiments #
################################################################################

#####################
# Model Simulations #
#####################
def test_state_construction_on_models_at_selected_steps(experiments):
    print_header("Test SSR on Models at selected steps")
    experiments.state_construction_on_models_at_selected_steps()


def test_state_construction_on_models_at_every_step(experiments):
    print_header("Test SSR on Models at every steps")
    experiments.state_construction_on_models_at_every_step()


####################
# Random Sequences #
####################
def test_state_construction_on_random_sequences_with_varying_clocks(experiments):
    print_header("Test SSR on Random Sequences")
    experiments.state_construction_on_random_sequences_with_varying_clocks()


def test_state_construction_on_random_sequences_with_varying_lengths(experiments):
    print_header("Test SSR on Random Sequences")
    experiments.state_construction_on_random_sequences_with_varying_lengths()
