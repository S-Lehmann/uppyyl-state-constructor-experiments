# Uppyyl State Constructor Experiments

The suite of state constructor experiments conducted for the article "Bounded DBM-Based Clock State Construction for Timed Automata in Uppaal" to be published in the "International Journal on Software Tools for Technology Transfer" (STTT).

## Getting Started

In this section, you will find instructions to setup the Uppyyl State Constructor Experiments on your local machine.

### Prerequisites

#### Python

Install Python >=3.8 for this project.

#### Virtual Environment

If you want to install the project in a dedicated virtual environment, first install virtualenv:
```
python3.8 -m pip install virtualenv
```

And create a virtual environment:

```
cd project_folder
virtualenv uppyyl-env
```

Then, activate the virtual environment on macOS and Linux via:

```
source ./uppyyl-env/bin/activate
```

or on Windows via:

```
source .\uppyyl-env\Scripts\activate
```

#### Dependencies

Note that the experiments use the Uppyyl Simulator and Uppyyl State Constructor packages, which need to be installed beforehand.

### Installing

To install the Uppyyl State Constructor directly from GitHub, run the following command:

```
python3.8 -m pip install -e git+https://github.com/S-Lehmann/uppyyl-state-constructor-experiments.git#egg=uppyyl-state-constructor-experiments
```

To install the project from a local directory instead, run:

```
python3.8 -m pip install -e path_to_project_root
```

NOTE: Some model-based experiments use a subset of the benchmark models included in the official [Uppaal](https://www.uppaal.org/) distribution.
For these experiments, copy the `2doors.xml`, `bridge.xml`, `fischer.xml`, `fischer-symmetry.xml`, `train-gate.xml`, and `train-gate-orig.xml` models to `./res/uppaal_demo_models`.
Furthermore, for the case-study-based models, copy the `csmacd2.xml` model (converted from [csma_input_02](https://www.it.uu.se/research/group/darts/uppaal/benchmarks/csma/csma_input_02.ta) to `xml` with Uppaal) and the `tdma.xml` model (described in detail in [[LP97]](https://www.it.uu.se/research/group/darts/papers/texts/lp-prfts97.pdf)) to `./res/uppaal_demo_models/case-study`.

### Usage

To run the experiments CLI tool, first switch to the experiments project directory:

```
cd path_to_uppyyl_state_constructor_experiments
```

Then, execute the following command:

```
python3.8 -m uppyyl_state_constructor_experiments
```

Via the CLI, you can run all experiments via `run`, or specific experiments via `run exp_name` (e.g., `run exp.gen_seq.var_clocks`)

### Experiments

The project suite contains the following experiment files (located in `uppyyl_state_constructor_experiments/experiments`):

1. `approach_examples/approach_examples.py`: Contains the examples of the article sections "Overapproximation Phase (O-phase)" and "Constraint Phase (C-phase)"
    * Inputs:
        - Statically defined operation sequence S=(DF, C(t1, t(0), 5), Cl, R(t1,0), R(t2,0), DF, C(t(0),t2,-3), Cl, R(t1,0), R(t3,0))
    * Outputs:
        - Shortened / generated sequences written to terminal
    * Experiments:
        - `experiment_o_seq_example()`: Applies the O(SEQ) approach to the example sequence
        - `experiment_o_dbm_example()`: Applies the O(DBM) approach to the example sequence
        - `experiment_c_fcs_example()`: Applies the C(FCS) approach after O(SEQ) to the example sequence
        - `experiment_c_mcs_example()`: Applies the C(MCS) approach after O(SEQ) to the example sequence
        - `experiment_c_rcs_example()`: Applies the C(RCS) approach after O(SEQ) to the example sequence
    * Executed in the CLI tool via `run exp.examples.approaches`.

2. `introduction_example/introduction_example.py`: Contains the introductory example first described in the article section "Introduction" and later analyzed with the Trivial, Rinast, and OC approach in the section "DBM-Based Clock State Construction").
    * Inputs:
        - `./res/example_models/introductory_example.xml` (preprocessed by `prepare_adapted_model()`)
        - `reference_data`: Simulated reference sequence of operations for the scenario described in the article (10 times `Execute` cycle, `Off`, `On`, another 10 times `Execute` cycle)
    * Outputs:
        - `./res/example_models/introductory_example[trivial].xml`
        - `./res/example_models/introductory_example[rinast].xml`
        - `./res/example_models/introductory_example[oc].xml`
    * Experiments:
        - `experiment_introduction_example_trivial()`: Applies the Trivial replay approach to the example model
        - `experiment_introduction_example_rinast()`: Applies the graph-based Rinast approach to the example model
        - `experiment_introduction_example_oc()`: Applies the OC approach (in this case, O(SEQ) + C(FCS)) to the example model
    * Executed in the CLI tool via `run exp.examples.introduction`.

3. `state_construction/article_experiments.py`: Contains the experiments on random and simulated sequences for the Trivial, Rinast, and OC approaches.
    * Inputs:
        - Random sequence generation routine (implemented by the `RandomSequenceGenerator` class in `uppyyl_state_constructor/uppyyl_state_constructor/backend/dbm_constructor/random_sequence_generator.py`)
        - The experiment model suite (paths defined in `state_construction/model_data.py`, models located in `./res/uppaal_demo_models`)
            + Included models: `2doors.xml`, `bridge.xml`, `fischer.xml`, `fischer-symmetry.xml`, `train-gate.xml`, `train-gate-orig.xml`, `csmacd2.xml`, `tdma.xml`
    * Outputs:
        - Reconstructed sequences lengths, sequence generation times, and sequence application times are saved to sub-directories of `./res/logs/clock_state_construction/data_logs/` per experiment
    * Experiments:
        1. `gather_model_details()`:
        Gathers component details (e.g., numbers of clocks, locations, edges) of each model in the model suite
        2. `state_construction_on_models_at_selected_steps()`:
        Performs state construction for each model in the model suite at selected simulation steps (e.g., after 1, 10, 20, 50, and 100 transitions).
        3. `state_construction_on_models_at_every_step()`:
        Performs state construction for each model in the model suite at every simulation step (e.g., up to n steps).
        Executed in the CLI tool via `run exp.sim_seq.every_step`.
        4. `state_construction_on_random_sequences_with_varying_clocks()`:
        Performs state construction for randomly generated operation sequences with varying clock counts.
        Executed in the CLI tool via `run exp.gen_seq.var_clocks`.
        5. `state_construction_on_random_sequences_with_varying_lengths()`:
        Performs state construction for randomly generated operation sequences of varying lengths.
        Executed in the CLI tool via `run exp.gen_seq.var_lengths`.

## Authors

* **Sascha Lehmann** - *Initial work* - [S-Lehmann](https://github.com/S-Lehmann)

See also the list of [contributors](https://github.com/S-Lehmann/uppyyl-state-constructor-experiments/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* The Uppaal model checking tool can be found at https://www.uppaal.org/.
* The project is associated with the [Institute for Software Systems](https://www.tuhh.de/sts) at Hamburg University of Technology.
