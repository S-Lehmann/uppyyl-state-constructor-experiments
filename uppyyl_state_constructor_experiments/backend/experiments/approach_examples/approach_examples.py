"""This module contains examples for the different O- and C-phase approaches."""
import os

from uppyyl_simulator.backend.data_structures.dbm.dbm import DBM
from uppyyl_simulator.backend.data_structures.dbm.dbm_operations.dbm_operations import (
    DBMOperationSequence, Constraint, Reset, Close, DelayFuture
)

from uppyyl_state_constructor.backend.dbm_constructor.oc.dbm_constructor_oc import DBMReconstructorOC, \
    ApproximationStrategy, ConstraintStrategy

################
# Example Data #
################
clocks = ["t1", "t2", "t3"]
sequence = DBMOperationSequence()
sequence.extend([
    DelayFuture(),
    Constraint(clock1="t1", clock2="T0_REF", rel="<=", val=5),
    Close(),
    Reset(clock="t1", val=0),
    Reset(clock="t2", val=0),
    DelayFuture(),
    Constraint(clock1="t2", clock2="T0_REF", rel=">=", val=3),
    Close(),
    Reset(clock="t1", val=0),
    Reset(clock="t3", val=0),
])
dbm_init = DBM(clocks=clocks, zero_init=True)
dbm_target = sequence.apply(dbm=dbm_init.copy())


##########
# Helper #
##########
def short_string_operation(op):
    """Generate a short string representation for a single DBM operation (e.g., R(...), C(...), Cl, DF)

    Args:
        op: The DBM operation.

    Returns:
        The short string representation.
    """
    if isinstance(op, Constraint):
        return f'C({op.clock1},{op.clock2},{op.val})'
    elif isinstance(op, Reset):
        return f'R({op.clock},{op.val})'
    elif isinstance(op, Close):
        return f'Cl'
    elif isinstance(op, DelayFuture):
        return f'DF'


###############
# Experiments #
###############
def experiment_approach_examples():
    """Executes the experiments for the approach examples."""
    print(f'\n============\n= Orig Seq =\n============')
    seq_str = "(" + ",".join(map(lambda op: short_string_operation(op), sequence)) + ")"
    print(seq_str)
    print(f'\n==========\n= Target =\n==========')
    print(dbm_target)

    experiment_o_dbm_example()
    experiment_o_seq_example()

    experiment_c_fcs_example()
    experiment_c_mcs_example()
    experiment_c_rcs_example()


def experiment_o_dbm_example():
    """Executes the example of the O(DBM) approach."""
    dbm_constructor_oc = DBMReconstructorOC(dbm_init=dbm_init.copy())
    dbm_constructor_oc.set_strategies(approximation=ApproximationStrategy.DBM, constraint=ConstraintStrategy.FCS)
    data_for_rec = {
        "dbm_init": dbm_init,
        "dbm_target": dbm_target.copy(),
        "seq_full": sequence,
        "seq_incr": sequence
    }
    res_csc = dbm_constructor_oc.time_measured_approximation(data_for_rec=data_for_rec)
    seq_approx = res_csc["seq_approx"]
    dbm_approx = seq_approx.apply(dbm_init.copy())

    print(f'\n==========\n= O(DBM) =\n==========')
    print(f'\n== DBM approximation sequence (len: {len(seq_approx)}) ==')
    seq_str = "(" + ",".join(map(lambda op: short_string_operation(op), seq_approx)) + ")"
    print(seq_str)
    print(f'\n== DBM ==')
    print(dbm_approx)
    assert dbm_approx.includes(dbm_target)


def experiment_o_seq_example():
    """Executes the example of the O(SEQ) approach."""
    dbm_constructor_oc = DBMReconstructorOC(dbm_init=dbm_init.copy())
    dbm_constructor_oc.set_strategies(approximation=ApproximationStrategy.SEQ, constraint=ConstraintStrategy.FCS)
    data_for_rec = {
        "dbm_init": dbm_init,
        "dbm_target": dbm_target.copy(),
        "seq_full": sequence,
        "seq_incr": sequence
    }
    res_csc = dbm_constructor_oc.time_measured_approximation(data_for_rec=data_for_rec)
    seq_approx = res_csc["seq_approx"]
    dbm_approx = seq_approx.apply(dbm_init.copy())

    print(f'\n==========\n= O(SEQ) =\n==========')
    print(f'\n== DBM approximation sequence (len: {len(seq_approx)}) ==')
    seq_str = "(" + ",".join(map(lambda op: short_string_operation(op), seq_approx)) + ")"
    print(seq_str)
    print(f'\n== DBM ==')
    print(dbm_approx)
    assert dbm_approx.includes(dbm_target)


def experiment_c_fcs_example():
    """Executes the example of the C(FCS) approach."""
    dbm_constructor_oc = DBMReconstructorOC(dbm_init=dbm_init.copy())
    dbm_constructor_oc.set_strategies(approximation=ApproximationStrategy.SEQ, constraint=ConstraintStrategy.FCS)
    data_for_rec = {
        "dbm_init": dbm_init,
        "dbm_target": dbm_target.copy(),
        "seq_full": sequence,
        "seq_incr": sequence
    }
    res_csc = dbm_constructor_oc.time_measured_reconstruction(data_for_rec=data_for_rec)
    seq_constr = res_csc["seq_constr"]
    dbm_constr = res_csc["dbm_rec"]

    print(f'\n==========\n= C(FCS) =\n==========')
    print(f'\n== DBM constraining sequence (len: {len(seq_constr)}) ==')
    seq_str = "(" + ",".join(map(lambda op: short_string_operation(op), seq_constr)) + ")"
    print(seq_str)
    print(f'\n== DBM ==')
    print(dbm_constr)

    assert dbm_constr == dbm_target


def experiment_c_mcs_example():
    """Executes the example of the C(MCS) approach."""
    dbm_constructor_oc = DBMReconstructorOC(dbm_init=dbm_init.copy())
    dbm_constructor_oc.set_strategies(approximation=ApproximationStrategy.SEQ, constraint=ConstraintStrategy.MCS)
    data_for_rec = {
        "dbm_init": dbm_init,
        "dbm_target": dbm_target.copy(),
        "seq_full": sequence,
        "seq_incr": sequence
    }
    res_csc = dbm_constructor_oc.time_measured_reconstruction(data_for_rec=data_for_rec)
    seq_constr = res_csc["seq_constr"]
    dbm_constr = res_csc["dbm_rec"]

    print(f'\n==========\n= C(MCS) =\n==========')
    print(f'\n== DBM constraining sequence (len: {len(seq_constr)}) ==')
    seq_str = "(" + ",".join(map(lambda op: short_string_operation(op), seq_constr)) + ")"
    print(seq_str)
    print(f'\n== DBM ==')
    print(dbm_constr)
    assert dbm_constr == dbm_target


def experiment_c_rcs_example():
    """Executes the example from the C(RCS) approach."""
    dbm_constructor_oc = DBMReconstructorOC(dbm_init=dbm_init.copy())
    dbm_constructor_oc.set_strategies(approximation=ApproximationStrategy.SEQ, constraint=ConstraintStrategy.RCS)
    data_for_rec = {
        "dbm_init": dbm_init,
        "dbm_target": dbm_target.copy(),
        "seq_full": sequence,
        "seq_incr": sequence
    }
    res_csc = dbm_constructor_oc.time_measured_reconstruction(data_for_rec=data_for_rec)
    seq_constr = res_csc["seq_constr"]
    dbm_constr = res_csc["dbm_rec"]

    print(f'\n==========\n= C(RCS) =\n==========')
    print(f'\n== DBM constraining sequence (len: {len(seq_constr)}) ==')
    seq_str = "(" + ",".join(map(lambda op: short_string_operation(op), seq_constr)) + ")"
    print(seq_str)
    print(f'\n== DBM ==')
    print(dbm_constr)
    assert dbm_constr == dbm_target
