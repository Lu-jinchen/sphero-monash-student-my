import numpy as np

def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def dynamics(state, action):

    x, y, heading, speed = state
    speed_cmd, heading_cmd = action

    if speed_cmd >= 0:
        speed_new = speed_cmd
        heading_new = wrap_angle(heading_cmd)
    else:
        speed_new = -speed_cmd
        heading_new = wrap_angle(heading_cmd + np.pi)

    # Calibration from real robot data
    distance_scale = 4.0

    x_new = (
        x
        + distance_scale
        * speed_new
        * np.sin(heading_new)
        * 0.1
    )

    y_new = (
        y
        + distance_scale
        * speed_new
        * np.cos(heading_new)
        * 0.1
    )

    return np.array(
        [x_new, y_new, heading_new, speed_new],
        dtype=np.float32
    )