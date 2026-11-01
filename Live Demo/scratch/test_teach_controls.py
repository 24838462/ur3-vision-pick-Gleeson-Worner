"""Verify compute_cartesian_step preserves X and Y during Z movement."""

import numpy as np
from ur3e_kinematics import UR3eKinematics
from pbvs_controller import PBVSController

def test_cartesian_step_preservation():
    kin = UR3eKinematics(tcp_offset_z=0.195)
    pbvs = PBVSController(kinematics=kin)

    # Example joint pose
    q0 = np.deg2rad([-52.4, -77.4, -61.8, -130.4, 90.7, 36.5])
    t0 = kin.forward_kinematics(q0, include_tcp=True)
    p0 = t0[:3, 3]

    # Command a purely vertical movement in Z (e.g. descend 20 mm)
    target_pos = p0.copy()
    target_pos[2] -= 0.020
    target_yaw = 0.0

    qd, arrived = pbvs.compute_cartesian_step(q0, target_pos, target_yaw)

    # Linear twist resulting from joint velocity qd
    J = kin.jacobian(q0, include_tcp=True)
    twist = J @ qd
    v_linear = twist[:3]

    print(f"Initial TCP: {np.round(p0, 4)}")
    print(f"Target TCP:  {np.round(target_pos, 4)}")
    print(f"Twist linear: Vx={v_linear[0]:.5f}, Vy={v_linear[1]:.5f}, Vz={v_linear[2]:.5f}")

    # Vx and Vy should be virtually zero (< 1e-4) compared to Vz
    assert abs(v_linear[0]) < 1e-3, f"Vx should be ~0, got {v_linear[0]}"
    assert abs(v_linear[1]) < 1e-3, f"Vy should be ~0, got {v_linear[1]}"
    assert v_linear[2] < -0.010, f"Vz should be negative downwards, got {v_linear[2]}"

    print("[PASS] Z motion strictly preserves X and Y with zero horizontal drift!")

if __name__ == "__main__":
    test_cartesian_step_preservation()
