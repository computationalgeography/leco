import numpy as np
import geopandas as gpd
from shapely import (
    Polygon,
    LineString,
)


def movement_direction(nr_agents: int, rng: np.random.default_rng) -> np.ndarray[float]:
    """Get random movement directions in radians for all agents"""
    angle = rng.uniform(
        0, 2 * np.pi, size=nr_agents
    )  # Get a random angle in radians between 0 and 2π for each agent

    return angle


def change_pos(speed: float, angle: float, axis: str) -> np.ndarray[float]:
    """Calculate the change in position based on speed and angle for a specified axis"""
    if axis == "x":
        return speed * np.cos(angle)  # Change in x position
    elif axis == "y":
        return speed * np.sin(angle)  # Change in y position
    else:
        raise ValueError("Axis must be 'x' or 'y'")


def move_axis(
    current_pos: np.ndarray[float], delta_pos: np.ndarray[float], max_pos: int
) -> np.ndarray[float]:
    """Move across a specified axis"""
    new_pos = np.clip(
        current_pos + delta_pos, a_min=0.0, a_max=max_pos
    )  # Values outside the interval are clipped to the interval edges

    return new_pos


def find_intersecting_movements(
    old_coor: gpd.GeoSeries.geometry,
    new_coor: gpd.GeoSeries.geometry,
    barrier: Polygon,
) -> np.ndarray[bool]:
    """Find which agent movements intersect with the barrier"""

    # Create movement lines from old to new positions
    movement_lines = gpd.GeoSeries(
        [LineString([p1, p2]) for p1, p2 in zip(old_coor, new_coor)],
        index=old_coor.index,
    )

    # Find which movement lines intersect the barrier
    crosses_barrier = movement_lines.intersects(barrier)

    return crosses_barrier.to_numpy()


def movement_across_barrier(
    position: gpd.GeoSeries.geometry,
    new_position: gpd.GeoSeries.geometry,
    barrier: Polygon,
    bar_impermeability: float,
    rng: np.random.default_rng,
) -> gpd.GeoSeries.geometry:
    """Handle movement across a barrier by stopping agents at the barrier"""

    # Check if the barrier impermeability falls within the range of [0,1]
    if bar_impermeability < 0 or bar_impermeability > 1:
        raise ValueError(
            f"The barrier impermeability value {bar_impermeability} must be between 0 and 1."
        )

    # Find agents which migration routes intersect with the barrier
    intersecting = find_intersecting_movements(position, new_position, barrier)

    # If none of the routes are intersecting with the barrier, return new positions
    if not np.any(intersecting):
        return new_position

    # Get the agent ids that intersect with the barrier
    intersecting_indices = np.where(intersecting)[0]

    # Generate impermeability masks for the intersecting agents based on the barrier impermeability
    impeded_prob = rng.random(len(intersecting_indices))
    impeded_agents = intersecting_indices[impeded_prob < bar_impermeability]

    # If none of the agents is impeded by the barrier, return new positions
    if not np.any(impeded_agents):
        return new_position

    # Get the x and y coordinates from the current positions of the impeded agents
    impeded_x = position.x.iloc[impeded_agents].values
    impeded_y = position.y.iloc[impeded_agents].values

    # Impeded agents stay at their old position
    new_position[impeded_agents] = gpd.points_from_xy(impeded_x, impeded_y)

    return new_position


def move(
    position: gpd.GeoSeries.geometry,
    barrier: Polygon | None,
    bar_impermeability: float,
    speed: float,
    x_max: int,
    y_max: int,
    rng: np.random.default_rng,
) -> gpd.GeoSeries.geometry:
    """Move agents across a continuous space with a certain speed"""

    angle = movement_direction(len(position), rng)  # Get random angles for all agents
    new_x = move_axis(
        position.x.values, change_pos(speed, angle, "x"), x_max
    )  # Calculate the change in position along x axis and move
    new_y = move_axis(
        position.y.values, change_pos(speed, angle, "y"), y_max
    )  # Calculate the change in position along y axis and move
    new_position = gpd.points_from_xy(new_x, new_y)

    # If there is no barrier or no impermeability from the barrier, return the new positions
    if barrier is None or bar_impermeability == 0:
        return new_position
    else:
        new_position_barcorrected = movement_across_barrier(
            position, new_position, barrier, bar_impermeability, rng
        )
        return new_position_barcorrected
