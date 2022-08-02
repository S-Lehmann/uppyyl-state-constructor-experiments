"""The main entry point of the Uppyyl state constructor experiments module."""
from uppyyl_state_constructor_experiments.cli.cli import UppyylStateConstructorExperimentsCLI


def main():
    """The main function."""
    prompt = UppyylStateConstructorExperimentsCLI()
    prompt.cmdloop()


if __name__ == '__main__':
    main()
