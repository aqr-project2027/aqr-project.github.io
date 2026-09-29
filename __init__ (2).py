"""Physics contracts for action-queried candidate trajectories (lazy imports)."""

__all__ = ["execution_slot_indices"]


def __getattr__(name):
    if name == "execution_slot_indices":
        from contactflow.dp3.action_query.trajectory import execution_slot_indices
        return execution_slot_indices
    raise AttributeError(name)
