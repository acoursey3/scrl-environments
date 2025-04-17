from __future__ import annotations

from typing import Any

import mujoco
import numpy as np 
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.envs.asset_path_utils import full_safe_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
from metaworld.envs.mujoco.utils import reward_utils, rotation
from metaworld.types import InitConfigDict

class SafeSawyerFaucetCloseEnv(SawyerXYZEnv):
    def __init__(
        self,
        render_mode = None
    ) -> None:
        hand_low = (-0.5, 0.40, -0.15)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.8, 0.0)
        obj_high = (0.1, 0.85, 0.0)

        # Use this to configure the bounds of the mug
        safe_low = (-0.15, 0.6, 0.01)
        safe_high = (0.15, 0.65, 0.01)

        self._handle_length = 0.175
        self._target_radius: float = 0.07

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            safety_constrained=True
        )

        self.init_config: InitConfigDict = {
            "obj_init_pos": np.array([0, 0.8, 0.0]),
            "hand_init_pos": np.array([0.0, 0.4, 0.2]),
            "safe_init_pos": np.array([0.0, 0.6, 0.0])
        }
        self.hand_init_pos = self.init_config["hand_init_pos"]
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.safe_init_pos = self.init_config["safe_init_pos"]

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self._safe_reset_space = Box(
            np.array(safe_low), np.array(safe_high), dtype=np.float64
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

    @property
    def model_name(self) -> str:
        return full_safe_path_for('safe_faucet.xml')

    @_assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            tcp_to_obj,
            _,
            target_to_obj,
            object_grasped,
            in_place,
        ) = self.compute_reward(action, obs)

        cost = self.compute_cost(action, obs)

        info = {
            "success": float(target_to_obj <= 0.07),
            "near_object": float(tcp_to_obj <= 0.01),
            "grasp_success": 1.0,
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
            "unscaled_reward": reward, 
            "unscaled_cost": cost
        }

        return reward, info

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `_target_site_config`."
        return [
            ("goal_close", self._target_pos),
            ("goal_open", np.array([10.0, 10.0, 10.0])),
        ]

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("faucetBase").xquat

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self._get_site_pos("handleStartClose") + np.array([0.0, 0.0, -0.01])

    def _get_quat_safe(self) -> npt.NDArray[Any]:
        return self.data.body("safeGeom").xquat

    def _get_pos_safe(self) -> npt.NDArray[Any]:
        return self.get_body_com("safeGeom") 

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()

        # Compute faucet position
        self.obj_init_pos = self._get_state_rand_vec()
        # Set mujoco body to computed position
        self.model.body("faucetBase").pos = self.obj_init_pos

        self.mug_init_pos = self._get_safe_rand_vec()
        self._set_safe_xyz(self.mug_init_pos)
        # self.model.body("safeGeom").qpos = self.mug_init_pos
        # self.model.body("safeGeom").pos = np.array([-1, -1, 0])

        self._target_pos = self.obj_init_pos + np.array(
            [-self._handle_length, 0.0, 0.125]
        )
        mujoco.mj_forward(self.model, self.data)
        self.model.site("goal_close").pos = self._target_pos
        return self._get_obs()

    def _reset_hand(self, steps: int = 50) -> None:
        super()._reset_hand(steps=steps)
        self.reachCompleted = False

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."
        obj = obs[4:7]
        tcp = self.tcp_center
        target = self._target_pos.copy()

        target_to_obj = obj - target
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = self.obj_init_pos - target
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self._target_radius),
            margin=abs(target_to_obj_init - self._target_radius),
            sigmoid="long_tail",
        )

        faucet_reach_radius = 0.01
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        tcp_to_obj_init = np.linalg.norm(self.obj_init_pos - self.init_tcp)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, faucet_reach_radius),
            margin=abs(tcp_to_obj_init - faucet_reach_radius),
            sigmoid="gaussian",
        )

        tcp_opened = 0
        object_grasped = reach

        reward = 2 * reach + 3 * in_place
        reward *= 2
        reward = 10 if target_to_obj <= self._target_radius else reward

        return (reward, tcp_to_obj, tcp_opened, target_to_obj, object_grasped, in_place)

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
