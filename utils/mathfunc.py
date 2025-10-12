import numpy as np
from scipy.spatial.transform import Rotation as R

def to_se3(position, quaternion):
    """
    Converts a position vector and a quaternion into a 4x4 SE(3) matrix.
    Args:
        position (np.ndarray): A 3-element array [x, y, z].
        quaternion (np.ndarray): A 4-element array [w, x, y, z].
    Returns:
        np.ndarray: A 4x4 homogeneous transformation matrix.
    """
    # Create a rotation object from the quaternion
    # Note: scipy's Rotation expects (x, y, z, w) format.
    rotation_scipy = R.from_quat(np.roll(quaternion, -1))

    # Get the 3x3 rotation matrix
    rotation_matrix = rotation_scipy.as_matrix()

    # Create the 4x4 identity matrix
    se3_matrix = np.identity(4)

    # Place the rotation matrix into the top-left 3x3 block
    se3_matrix[:3, :3] = rotation_matrix

    # Place the position vector into the top-right 3x1 block
    se3_matrix[:3, 3] = position

    return se3_matrix


def pose_error(T_estimate, T_truth = None):
    """
    Calculates and prints pose error, the translation error and angle difference between z-axes.
    Args:
        T_estimate (np.ndarray): Estimated SE(3) pose matrix.
        T_truth (np.ndarray): Ground Truth SE(3) pose matrix.
    """
    if T_truth is None: return
    t1 = T_truth[:3, 3]
    t2 = T_estimate[:3, 3]
    translation_error =  np.linalg.norm(t2 - t1)
    z1 = T_truth[:3, 2]
    z2 = T_estimate[:3, 2]

    print(f"t1: ", t1, "t2: ", t2)
    print(f"z1: ", z1, "z2: ", z2)

    angle_rad = np.arccos(np.dot(z1, z2))

    print(f"translation error: {translation_error:.3f}")
    print(f"rotation error in rad: {angle_rad:.3f}")
    print(f"rotation error in deg: {np.degrees(angle_rad):.3f}")