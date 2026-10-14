"""Camera-frame PBVS for pickup and placement alignment."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
GAIN = 0.8
MAX_CAMERA_SPEED = 0.04
MAX_CAMERA_ANGULAR_SPEED = 0.35
MAX_JOINT_SPEED = 0.6
PICK_APPROACH_CAMERA_SPEED = 0.025
PICK_APPROACH_ANGULAR_SPEED = 0.15
PICK_APPROACH_JOINT_SPEED = 0.35
PLACEMENT_GAIN = 0.5
PLACEMENT_CAMERA_SPEED = 0.02
PLACEMENT_ANGULAR_SPEED = 0.15
PLACEMENT_JOINT_SPEED = 0.25
DAMPING = 0.005
ROTATION_WEIGHT = 0.15
APPROACH_CLEARANCE = 0.035
PLACEMENT_ALIGN_Z = 0.22
MIN_PLACEMENT_CLEARANCE = 0.08
GRASP_OFFSET_Z = 0.008
RELEASE_GAP = 0.003
LOST_TIMEOUT = 0.6
POSITION_TOLERANCE = 0.003
ROTATION_TOLERANCE = np.deg2rad(3)

class PoseServo:

    def __init__(
        self,
        model,
        frames,
        v_addresses,
        target,
        phase='pick_approach',
        held_size=None,
        transport_height=None,
    ):
        if phase not in ('pick_approach', 'pick_descend', 'place_approach', 'place_transport'):
            raise ValueError(f'Unknown PBVS phase: {phase}')
        self.model = model
        self.frames = frames
        self.va = np.asarray(v_addresses)
        self.phase = phase
        self.placing = phase.startswith('place')
        self.descending = phase == 'pick_descend'
        self.name = target['name']
        self.size = float(target['size'])
        self.reference_yaw = float(target['yaw'])
        self.held_size = held_size
        self.transport_height = transport_height
        if phase == 'place_transport' and transport_height is None:
            raise ValueError('PBVS transport requires a transport height.')
        if self.placing and (held_size is None or held_size <= 0):
            raise ValueError('Placement requires a measured held-cube size.')
        self.last_target = dict(target)
        self.last_seen = None
        self.command = np.zeros(6)
        self.valid = False
        self.rejection_reason = 'Waiting for a fresh RGB-D measurement.'
        self.converged = False
        self.stable_frames = 0
        self.translation_error = float('nan')
        self.rotation_error = float('nan')
        self.grasp_world = None
        self.grasp_rotation_world = None
        self.mount_rotation = None
        self.mount_translation = None
        self.pinch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'gripper_pinch')
        if self.pinch_id < 0:
            raise ValueError('Missing gripper_pinch')
        self.gain = PLACEMENT_GAIN if self.placing else GAIN
        self.speed_limit = (PLACEMENT_CAMERA_SPEED if self.placing else
                            0.015 if self.descending else PICK_APPROACH_CAMERA_SPEED)
        self.angular_speed_limit = (PLACEMENT_ANGULAR_SPEED if self.placing else
                                    MAX_CAMERA_ANGULAR_SPEED if self.descending else PICK_APPROACH_ANGULAR_SPEED)
        self.joint_speed_limit = (PLACEMENT_JOINT_SPEED if self.placing else
                                  MAX_JOINT_SPEED if self.descending else PICK_APPROACH_JOINT_SPEED)
        self.position_tolerance = 0.0015 if self.descending else POSITION_TOLERANCE
        self.rotation_tolerance = np.deg2rad(1) if self.descending or self.placing else ROTATION_TOLERANCE

    def reject_measurement(self, reason):
        self.stable_frames = 0
        self.converged = False
        self.rejection_reason = reason

    def update(self, measurement, now):
        self.command[:] = 0.0
        self.valid = False
        if measurement is None:
            self.reject_measurement('No accepted RGB-D cube/surface estimate: check visibility, occlusion and top-face fitting.')
            return
        if abs(measurement['size'] - self.size) > 0.003:
            self.reject_measurement('Measured size differs from the recorded size by more than 3 mm.')
            return
        centre = np.asarray(measurement['centre_base'], dtype=float)
        old_centre = np.asarray(self.last_target['centre_base'])
        if (
            not np.all(np.isfinite(centre))
            or not np.isfinite(measurement['yaw'])
            or np.linalg.norm(centre - old_centre) > 0.02
        ):
            self.reject_measurement('Measurement is non-finite or its centre jumped by more than 20 mm.')
            return
        delta = (measurement['yaw'] - self.reference_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        yaw = self.reference_yaw + delta
        frames = self.frames
        data = frames.kinematics
        base_rotation_world = data.xmat[frames.base_id].reshape(3, 3)
        base_position_world = data.xpos[frames.base_id]
        pinch_rotation_base = base_rotation_world.T @ data.site_xmat[self.pinch_id].reshape(3, 3)
        pinch_position_base = base_rotation_world.T @ (data.site_xpos[self.pinch_id] - base_position_world)
        camera_rotation_base = frames.rotation_base_camera
        camera_position_base = frames.position_base_camera
        if self.mount_rotation is None:
            self.mount_rotation = pinch_rotation_base.T @ camera_rotation_base
            self.mount_translation = pinch_rotation_base.T @ (camera_position_base - pinch_position_base)
        object_rotation_base = Rotation.from_euler('z', yaw).as_matrix()
        grasp_rotation_base = object_rotation_base @ np.diag([1.0, -1.0, -1.0])
        height_offset = GRASP_OFFSET_Z
        if self.placing:
            height_offset += self.held_size / 2 + RELEASE_GAP
        final_pinch = centre + np.array([0.0, 0.0, height_offset])
        desired_pinch = final_pinch.copy()
        if self.phase == 'place_transport':
            desired_pinch[2] = self.transport_height
        elif self.placing and (not self.descending):
            desired_pinch[2] = max(PLACEMENT_ALIGN_Z, final_pinch[2] + MIN_PLACEMENT_CLEARANCE)
        elif not self.descending:
            desired_pinch[2] += APPROACH_CLEARANCE
        desired_camera_rotation = grasp_rotation_base @ self.mount_rotation
        desired_camera_position = desired_pinch + grasp_rotation_base @ self.mount_translation
        target_translation_camera = camera_rotation_base.T @ (centre - camera_position_base)
        desired_translation_camera = desired_camera_rotation.T @ (centre - desired_camera_position)
        target_rotation_camera = camera_rotation_base.T @ object_rotation_base
        desired_rotation_camera = desired_camera_rotation.T @ object_rotation_base
        translation_error = target_translation_camera - desired_translation_camera
        rotation_error = Rotation.from_matrix(target_rotation_camera @ desired_rotation_camera.T).as_rotvec()
        omega = self.gain * rotation_error
        velocity = self.gain * translation_error + np.cross(target_translation_camera, omega)
        scale = max(
            1.0,
            np.linalg.norm(velocity) / self.speed_limit,
            np.linalg.norm(omega) / self.angular_speed_limit,
        )
        twist = np.r_[velocity, omega] / scale
        jac_position = np.zeros((3, self.model.nv))
        jac_rotation = np.zeros((3, self.model.nv))
        mujoco.mj_jac(
            self.model,
            data,
            jac_position,
            jac_rotation,
            data.cam_xpos[frames.camera_id],
            int(self.model.cam_bodyid[frames.camera_id]),
        )
        world_to_optical = (base_rotation_world @ camera_rotation_base).T
        jacobian = np.vstack((
            world_to_optical @ jac_position[:, self.va],
            ROTATION_WEIGHT * world_to_optical @ jac_rotation[:, self.va],
        ))
        weighted_twist = twist.copy()
        weighted_twist[3:] *= ROTATION_WEIGHT
        rates = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + DAMPING ** 2 * np.eye(6),
            weighted_twist,
        )
        rates /= max(1.0, np.max(np.abs(rates)) / self.joint_speed_limit)
        if not np.all(np.isfinite(rates)):
            raise RuntimeError('Invalid PBVS joint command.')
        self.translation_error = float(np.linalg.norm(translation_error))
        self.rotation_error = float(np.linalg.norm(rotation_error))
        self.valid = True
        self.rejection_reason = None
        self.last_seen = float(now)
        self.last_target = dict(measurement, name=self.name, yaw=yaw)
        self.grasp_world = frames.base_to_world(final_pinch)
        self.grasp_rotation_world = base_rotation_world @ grasp_rotation_base
        self.pinch_error = float(np.linalg.norm(pinch_position_base - desired_pinch))
        close = self.translation_error < self.position_tolerance and self.pinch_error < self.position_tolerance and (self.rotation_error < self.rotation_tolerance)
        self.stable_frames = self.stable_frames + 1 if close else 0
        self.converged = self.stable_frames >= 5
        self.command[:] = 0.0 if self.converged else rates
