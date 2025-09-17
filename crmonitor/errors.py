from crmonitor.common.world import World


class StlMonitorError(Exception): ...


class PredicateEvaluationError(StlMonitorError):
    def __init__(
        self,
        predicate_name: str,
        world: World,
        time_step: int,
        vehicle_ids: tuple[int, ...],
        reason: str | None = None,
    ) -> None:
        msg = f"Failed to evaluate predicate {predicate_name} in scenario {world.scenario_id} at time step {time_step} for vehicle(s) {','.join([str(vehicle_id) for vehicle_id in vehicle_ids])}"

        if reason is not None:
            msg += f": {reason}"

        super().__init__(msg)
