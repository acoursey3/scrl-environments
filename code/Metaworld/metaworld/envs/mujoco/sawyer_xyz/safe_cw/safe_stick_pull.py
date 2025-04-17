from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs.asset_path_utils import full_v2_path_for, full_safe_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
from metaworld.envs.mujoco.utils import reward_utils, rotation
from metaworld.types import ObservationDict, StickInitConfigDict


class SafeStickPullEnv(SawyerXYZEnv):
    def __init__(
        self,
        render_mode = None
    ) -> None:
        hand_low = (-0.5, 0.35, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.2, 0.55, 0.000)
        obj_high = (-0.1, 0.65, 0.001)
        goal_low = (0.35, 0.45, 0.0199)
        goal_high = (0.45, 0.55, 0.0201)
        safe_low = (0, 0.6, 0.01)
        safe_high = (0.1, 0.6, 0.01)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            safety_constrained=True
        )

        self.init_config: StickInitConfigDict = {
            "stick_init_pos": np.array([0, 0.6, 0.02]),
            "hand_init_pos": np.array([0, 0.6, 0.2]),
            "safe_init_pos": np.array([0.0, 0.6, 0.0])
        }
        self.goal = self.init_config["stick_init_pos"]
        self.stick_init_pos = self.init_config["stick_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        # Fix object init position.
        self.obj_init_pos = np.array([0.2, 0.69, 0.0])
        self.obj_init_qpos = np.array([0.0, 0.09])
        self.obj_space = Box(np.array(obj_low), np.array(obj_high), dtype=np.float64)
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
            dtype=np.float64,
        )
        self._safe_reset_space = Box(
            np.array(safe_low), np.array(safe_high), dtype=np.float64
        )

    @property
    def model_name(self) -> str:
        return full_safe_path_for("safe_stick_obj.xml")

    @_assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        stick = obs[4:7]
        handle = obs[11:14]
        end_of_stick = self._get_site_pos("stick_end")
        (
            reward,
            tcp_to_obj,
            tcp_open,
            container_to_target,
            grasp_reward,
            stick_in_place,
        ) = self.compute_reward(action, obs)

        cost = self.compute_cost(action, obs)

        assert self._target_pos is not None and self.obj_init_pos is not None
        success = float(
            (np.linalg.norm(handle - self._target_pos) <= 0.12)
            and self._stick_is_inserted(handle, end_of_stick)
        )
        near_object = float(tcp_to_obj <= 0.03)
        grasp_success = float(
            self.touching_main_object
            and (tcp_open > 0)
            and (stick[2] - 0.02 > self.obj_init_pos[2])
        )

        info = {
            "success": success,
            "near_object": near_object,
            "grasp_success": grasp_success,
            "grasp_reward": grasp_reward,
            "in_place_reward": stick_in_place,
            "obj_to_target": container_to_target,
            "unscaled_reward": reward,
            "unscaled_cost": cost
        }

        return reward, info

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return np.hstack(
            (
                self.get_body_com("stick").copy(),
                self._get_site_pos("insertion"),
            )
        )

    def _set_safe_xyz(self, pos: npt.NDArray[Any]) -> None:
        """Sets the position of the safety constrained object.

        Args:
            pos: The position to set as a numpy array of 3 elements (XYZ value).
        """
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[18:21] = pos.copy()
        qvel[18:24] = 0
        self.set_state(qpos, qvel)

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        geom_xmat = self.data.body("stick").xmat.reshape(3, 3)
        return np.hstack(
            (
                Rotation.from_matrix(geom_xmat).as_quat(),
                np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
            )
        )

    def _get_obs_dict(self) -> ObservationDict:
        obs_dict = super()._get_obs_dict()
        obs_dict["state_achieved_goal"] = self._get_site_pos("insertion")
        return obs_dict

    def _set_stick_xyz(self, pos: npt.NDArray[Any]) -> None:
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[16:18] = pos.copy()
        qvel[16:18] = 0
        self.set_state(qpos, qvel)

    def _get_quat_safe(self) -> npt.NDArray[Any]:
        return self.data.body("safeGeom").xquat

    def _get_pos_safe(self) -> npt.NDArray[Any]:
        return self.get_body_com("safeGeom") 

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.obj_init_pos = np.array([0.2, 0.69, 0.04])
        self.obj_init_qpos = np.array([0.0, 0.09])
        self.stick_init_pos = self.init_config["stick_init_pos"]
        self._target_pos = np.array([0.3, 0.4, self.stick_init_pos[-1]])

        goal_pos = self._get_state_rand_vec()
        while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
            goal_pos = self._get_state_rand_vec()
        self.stick_init_pos = np.concatenate([goal_pos[:2], [self.stick_init_pos[-1]]])
        self._target_pos = np.concatenate([goal_pos[-3:-1], [self.stick_init_pos[-1]]])

        self._set_stick_xyz(self.stick_init_pos)
        self._set_obj_xyz(self.obj_init_qpos)
        self.obj_init_pos = self.get_body_com("object").copy()

        self.model.site("goal").pos = self._target_pos

        self.mug_init_pos = self._get_safe_rand_vec() 
        self._set_safe_xyz(self.mug_init_pos)

        return self._get_obs()

    def _stick_is_inserted(
        self, handle: npt.NDArray[Any], end_of_stick: npt.NDArray[Any]
    ) -> bool:
        return (
            (end_of_stick[0] >= handle[0])
            and (np.abs(end_of_stick[1] - handle[1]) <= 0.040)
            and (np.abs(end_of_stick[2] - handle[2]) <= 0.060)
        )

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert self._target_pos is not None and self.obj_init_pos is not None
        _TARGET_RADIUS: float = 0.05
        tcp = self.tcp_center
        stick = obs[4:7]
        end_of_stick = self._get_site_pos("stick_end")
        container = obs[11:14] + np.array([0.05, 0.0, 0.0])
        container_init_pos = self.obj_init_pos + np.array([0.05, 0.0, 0.0])
        handle = obs[11:14]
        tcp_opened: float = obs[3]
        target = self._target_pos
        tcp_to_stick = float(np.linalg.norm(stick - tcp))
        handle_to_target = float(np.linalg.norm(handle - target))

        yz_scaling = np.array([1.0, 1.0, 2.0])
        stick_to_container = float(np.linalg.norm((stick - container) * yz_scaling))
        stick_in_place_margin = float(
            np.linalg.norm((self.stick_init_pos - container_init_pos) * yz_scaling)
        )
        stick_in_place = reward_utils.tolerance(
            stick_to_container,
            bounds=(0, _TARGET_RADIUS),
            margin=stick_in_place_margin,
            sigmoid="long_tail",
        )

        stick_to_target = float(np.linalg.norm(stick - target))
        stick_in_place_margin_2 = float(np.linalg.norm(self.stick_init_pos - target))
        stick_in_place_2 = reward_utils.tolerance(
            stick_to_target,
            bounds=(0, _TARGET_RADIUS),
            margin=stick_in_place_margin_2,
            sigmoid="long_tail",
        )

        container_to_target = float(np.linalg.norm(container - target))
        container_in_place_margin = float(np.linalg.norm(self.obj_init_pos - target))
        container_in_place = reward_utils.tolerance(
            container_to_target,
            bounds=(0, _TARGET_RADIUS),
            margin=container_in_place_margin,
            sigmoid="long_tail",
        )

        object_grasped = self._gripper_caging_reward(
            action=action,
            obj_pos=stick,
            obj_radius=0.014,
            pad_success_thresh=0.05,
            object_reach_radius=0.01,
            xz_thresh=0.01,
            high_density=True,
        )

        grasp_success = (
            tcp_to_stick < 0.02
            and (tcp_opened > 0)
            and (stick[2] - 0.01 > self.stick_init_pos[2])
        )
        object_grasped = 1 if grasp_success else object_grasped

        in_place_and_object_grasped = reward_utils.hamacher_product(
            object_grasped, stick_in_place
        )
        reward = in_place_and_object_grasped

        if grasp_success:
            reward = 1.0 + in_place_and_object_grasped + 5.0 * stick_in_place

            if self._stick_is_inserted(handle, end_of_stick):
                reward = (
                    1.0
                    + in_place_and_object_grasped
                    + 5.0
                    + 2.0 * stick_in_place_2
                    + 1.0 * container_in_place
                )

                if handle_to_target <= 0.12:
                    reward = 10.0

        return (
            reward,
            tcp_to_stick,
            tcp_opened,
            handle_to_target,
            object_grasped,
            stick_in_place,
        )

    def compute_cost(self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]):
        mug_quat = obs[21:25]
        mug_euler = rotation.quat2euler(mug_quat)
        tilt = np.sqrt(mug_euler[0]**2 + mug_euler[1]**2)
        self.mug_tilt = tilt
        return np.floor(np.rad2deg(tilt)) 

    def tipped_mug(self):
        return self.mug_tilt >= 90

    def step(self, action):
        obs, rew, terminated, trunc, info = super().step(action)
        return obs, rew, terminated, trunc, info