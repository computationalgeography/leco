"""Functions for movement of agents across continuous space."""

import geopandas as gpd
import numpy as np
import numpy.typing as npt
from shapely import LineString
from shapely.geometry.base import BaseGeometry


def movement_direction(nr_agents: int, rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """Get random movement directions in radians for all agents."""
    return rng.uniform(
        0,
        2 * np.pi,
        size=nr_agents,
    )  # Get a random angle in radians between 0 and 2π for each agent


def change_position(
    speed: float,
    angle: npt.NDArray[np.float64],
    axis: str,
) -> npt.NDArray[np.float64]:
    """Calculate the change in position based on speed and angle for a specified axis."""
    if axis == "x":
        return speed * np.cos(angle)  # Change in x position
    if axis == "y":
        return speed * np.sin(angle)  # Change in y position
    raise ValueError(f"Invalid axis: {axis}. Must be 'x' or 'y".format(axis))


def move_axis(
    current_pos: npt.NDArray[np.float64],
    delta_pos: npt.NDArray[np.float64],
    max_pos: float,
) -> npt.NDArray[np.float64]:
    """Move across a specified axis."""
    return np.clip(
        current_pos + delta_pos,
        a_min=0.0,
        a_max=max_pos,
    )  # Values outside the interval are clipped to the interval edges


def find_intersecting_movements(
    old_coordinates: gpd.GeoSeries,
    new_coordinates: gpd.GeoSeries,
    barrier: BaseGeometry,
) -> npt.NDArray[np.bool]:
    """Find which agent movements intersect with the barrier."""
    # Create movement lines from old to new positions
    movement_lines = gpd.GeoSeries(
        # Gives error if old_coordinates and new_coordinates have different lengths
        [LineString([p1, p2]) for p1, p2 in zip(old_coordinates, new_coordinates, strict=True)],
        index=old_coordinates.index,
    )

    # Find which movement lines intersect the barrier
    crosses_barrier = movement_lines.intersects(barrier)

    return crosses_barrier.to_numpy()


def movement_across_barrier(
    position: gpd.GeoSeries,
    new_position: gpd.GeoSeries,
    barrier: BaseGeometry,
    impermeability: float,
    rng: np.random.Generator,
) -> gpd.GeoSeries:
    """Handle movement across a barrier by stopping agents at the barrier."""
    # Check if the barrier impermeability falls within the range of [0,1]
    if impermeability < 0 or impermeability > 1:
        raise ValueError(
            f"The barrier impermeability value {impermeability} must be between 0 and 1.".format(
                impermeability,
            ),
        )

    # Find agents which migration routes intersect with the barrier
    intersecting = find_intersecting_movements(position, new_position, barrier)

    # If none of the routes are intersecting with the barrier, return new positions
    if not np.any(intersecting):
        return new_position

    # Get the agent ids that intersect with the barrier
    intersecting_indices = np.where(intersecting)[0]

    # Generate impermeability masks for the intersecting agents based on the barrier impermeability
    impeded_probabilities = rng.random(len(intersecting_indices))
    impeded_agents = intersecting_indices[impeded_probabilities < impermeability]

    # If none of the agents is impeded by the barrier, return new positions
    if not np.any(impeded_agents):
        return new_position

    # Get the x and y coordinates from the current positions of the impeded agents
    impeded_x = position.x.iloc[impeded_agents].values
    impeded_y = position.y.iloc[impeded_agents].values

    # Impeded agents stay at their old position
    updated_position = new_position.copy()
    updated_position.iloc[impeded_agents] = gpd.GeoSeries(
        gpd.points_from_xy(impeded_x, impeded_y),
        index=updated_position.iloc[impeded_agents].index,
        crs=updated_position.crs,
    )

    return updated_position


def move(
    position: gpd.GeoSeries,
    barrier: BaseGeometry | None,
    impermeability: float,
    speed: float,
    space: list[float],
    rng: np.random.Generator,
) -> gpd.GeoSeries:
    """Move agents across a continuous space with a certain speed."""
    angle = movement_direction(len(position), rng)  # Get random angles for all agents

    new_x = move_axis(
        position.x.to_numpy(dtype=np.float64),
        change_position(speed, angle, "x"),
        space[0],
    )  # Calculate the change in position along x axis and move

    new_y = move_axis(
        position.y.to_numpy(dtype=np.float64),
        change_position(speed, angle, "y"),
        space[1],
    )  # Calculate the change in position along y axis and move

    new_position = gpd.GeoSeries(
        gpd.points_from_xy(new_x, new_y),
        index=position.index,
        crs=position.crs,
    )

    # If there is no barrier or no impermeability from the barrier, return the new positions
    if barrier is None or impermeability == 0:
        return new_position

    return movement_across_barrier(
        position,
        new_position,
        barrier,
        impermeability,
        rng,
    )
