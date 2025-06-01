"""
Scenario type classification module for determining whether a scenario represents
an interstate/highway or intersection driving scenario.
"""

import logging
from enum import Enum, auto

from commonroad.scenario.lanelet import LaneletType
from commonroad.scenario.scenario import Scenario, Tag

_LOGGER = logging.getLogger(__name__)


class ScenarioType(Enum):
    INTERSTATE = auto()
    INTERSECTION = auto()

    def __str__(self) -> str:
        if self == ScenarioType.INTERSTATE:
            return "interstate"
        else:
            return "intersection"


def determine_scenario_type(
    scenario: Scenario, default: ScenarioType = ScenarioType.INTERSTATE
) -> ScenarioType:
    """
    Determine the type of a CommonRoad scenario using multiple classification methods.

    This function prioritizes tag-based classification over lanelet network analysis.
    If a scenario has conflicting indicators (both interstate and intersection
    characteristics) for all crierions, the default scenario type is used.

    Args:
        scenario: The CommonRoad scenario to classify.
        default: The fallback scenario type if classification fails.

    Returns:
        ScenarioType: The determined scenario type, or default if classification fails
    """
    scenario_type = _try_determine_scenario_type_from_scenario_tags(scenario)
    if scenario_type is not None:
        _LOGGER.debug(
            f"Determined scenario type of scenario {scenario.scenario_id} from its tags to be {scenario_type}"
        )

    scenario_type = _try_determine_scenario_type_from_lanelet_network(scenario)
    if scenario_type is not None:
        _LOGGER.debug(
            f"Determined scenario type of scenario {scenario.scenario_id} from its lanelet network to be {scenario_type}"
        )

    _LOGGER.debug(
        f"Failed to determine scenario type for scenario {scenario.scenario_id}. "
        + f"Falling back to default scenario type {default}"
    )
    return default


_QUALIFING_INTERSECTION_TAGS = {Tag.INTERSECTION}
_QUALIFING_INTERSTATE_TAGS = {Tag.INTERSTATE, Tag.HIGHWAY}


def _try_determine_scenario_type_from_scenario_tags(scenario: Scenario) -> ScenarioType | None:
    """
    Attempt to determine scenario type based on scenario tags.

    Args:
        scenario: The scenario whose tags should be analyzed

    Returns:
        The determined scenario type, or None if it cannot be determined.
    """
    if scenario.tags is None:
        return None

    matching_intersection_tags = scenario.tags.intersection(_QUALIFING_INTERSECTION_TAGS)
    matching_interstate_tags = scenario.tags.intersection(_QUALIFING_INTERSTATE_TAGS)

    has_intersection_tags = len(matching_intersection_tags) > 0
    has_interstate_tags = len(matching_interstate_tags) > 0

    if has_intersection_tags and has_interstate_tags:
        # Conflicting tags - cannot determine type reliably.
        return None
    elif has_intersection_tags:
        return ScenarioType.INTERSECTION
    elif has_interstate_tags:
        return ScenarioType.INTERSTATE
    else:
        # No qualifying tags found.
        return None


_QUALIFYING_INTERSECTION_LANELET_TYPES = {
    LaneletType.INTERSECTION,
    LaneletType.CROSSWALK,
    LaneletType.SIDEWALK,
}
_QUALIFYING_INTERSTATE_LANELET_TYPES = {
    LaneletType.HIGHWAY,
    LaneletType.INTERSTATE,
    LaneletType.MAIN_CARRIAGE_WAY,
    LaneletType.ACCESS_RAMP,
    LaneletType.EXIT_RAMP,
}


def _try_determine_scenario_type_from_lanelet_network(scenario: Scenario) -> ScenarioType | None:
    """
    Attempt to determine scenario type based on lanelet network.

    Args:
        scenario: The scenario whose lanelet network should be analyzed

    Returns:
        ScenarioType: The determined scenario type, or None if undecidable.
    """

    matching_intersection_lanelets = {
        lanelet
        for lanelet in scenario.lanelet_network.lanelets
        if len(lanelet.lanelet_type.intersection(_QUALIFYING_INTERSECTION_LANELET_TYPES)) > 0
    }

    matching_interstate_lanelets = {
        lanelet
        for lanelet in scenario.lanelet_network.lanelets
        if len(lanelet.lanelet_type.intersection(_QUALIFYING_INTERSTATE_LANELET_TYPES)) > 0
    }

    has_intersection_lanelets = len(matching_intersection_lanelets) > 0
    has_interstate_lanelets = len(matching_interstate_lanelets) > 0

    if has_intersection_lanelets and has_interstate_lanelets:
        # Mixed scenario - cannot classify reliably.
        return None
    elif has_intersection_lanelets:
        return ScenarioType.INTERSECTION
    elif has_interstate_lanelets:
        return ScenarioType.INTERSTATE
    else:
        # No qualifying lanelets found.
        return None
