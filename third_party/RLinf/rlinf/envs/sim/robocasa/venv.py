# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Subprocess vectorized environment for Robocasa.

Based on metaworld/venv.py implementation, adapted for Robocasa/Robosuite environments.
"""

import os
import signal
from collections import OrderedDict
from multiprocessing import connection, get_context
from typing import Any, Callable, Optional, Union

import gymnasium as gym
import numpy as np

from rlinf.envs.venv import (
    BaseVectorEnv,
    CloudpickleWrapper,
    EnvWorker,
    ShArray,
    SubprocEnvWorker,
    SubprocVectorEnv,
    _setup_buf,
)


def _build_shared_buffer(value: Any, context=None) -> Any:
    """Build shared buffers for ndarray leaves and omit metadata leaves."""
    if isinstance(value, dict):
        return {
            key: buffer
            for key, item in value.items()
            if (buffer := _build_shared_buffer(item, context)) is not None
        }
    if isinstance(value, tuple):
        return tuple(_build_shared_buffer(item, context) for item in value)
    array = np.asarray(value)
    if array.dtype == np.dtype("O"):
        return None
    try:
        return ShArray(array.dtype, tuple(array.shape), context=context)
    except (KeyError, AttributeError, TypeError):
        return None


def _worker(
    parent: connection.Connection,
    p: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    obs_bufs: Optional[Union[dict, tuple, ShArray]] = None,
) -> None:
    """Worker function for robocasa subprocess environment.

    Based on metaworld's _worker function, adapted for robosuite environments.
    """

    def _encode_obs(
        obs: Union[dict, tuple, np.ndarray], buffer: Union[dict, tuple, ShArray]
    ) -> None:
        if isinstance(obs, np.ndarray) and isinstance(buffer, ShArray):
            buffer.save(obs)
        elif isinstance(obs, tuple) and isinstance(buffer, tuple):
            for o, b in zip(obs, buffer):
                _encode_obs(o, b)
        elif isinstance(obs, dict) and isinstance(buffer, dict):
            for k in obs.keys():
                if k in buffer:
                    _encode_obs(obs[k], buffer[k])
        return None

    def _check_success(env, env_return):
        success = env._check_success()
        env_return = list(env_return)
        info = env_return[-1]
        info["success"] = success
        env_return[-1] = info
        env_return = tuple(env_return)
        return env_return

    def get_ep_meta(env, env_return):
        ep_meta = env.get_ep_meta()
        env_return = list(env_return)
        info = env_return[-1]
        info["ep_meta"] = ep_meta
        env_return[-1] = info
        env_return = tuple(env_return)
        return env_return

    parent.close()
    env_fn = env_fn_wrapper.data
    env = env_fn()
    print(
        "[RoboCasa] worker ready; "
        f"pid={os.getpid()} "
        f"task_id={getattr(env_fn, '_rlinf_task_id', None)} "
        f"task={getattr(env_fn, '_rlinf_task_name', None)}",
        flush=True,
    )
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:  # the pipe has been closed
                p.close()
                break
            if cmd == "step":
                env_return = env.step(data)
                # Standard RoboCasa Gym environments use the Gymnasium
                # five-value API; normalize it to RLinf's legacy four-value
                # vector-env contract.
                if isinstance(env_return, tuple) and len(env_return) == 5:
                    obs, reward, terminated, truncated, info = env_return
                    env_return = (obs, reward, bool(terminated or truncated), info)
                if obs_bufs is not None:
                    _encode_obs(env_return[0], obs_bufs)
                    env_return = (None, *env_return[1:])
                # RoboCasa step can't record success in info, _check_success() must be called
                if hasattr(env, "_check_success"):
                    env_return = _check_success(env, env_return)
                # call get_ep_meta() to get the RoboCasa env meta, includes prompt & layout_id, etcs
                if hasattr(env, "get_ep_meta"):
                    env_return = get_ep_meta(env, env_return)
                p.send(env_return)
            elif cmd == "reset":
                # Robosuite reset can return just obs or (obs, info)
                retval = env.reset(**data)
                reset_returns_info = (
                    isinstance(retval, (tuple, list))
                    and len(retval) == 2
                    and isinstance(retval[1], dict)
                )
                if reset_returns_info:
                    obs, info = retval
                else:
                    obs = retval
                    info = {}
                if obs_bufs is not None:
                    _encode_obs(obs, obs_bufs)
                    obs = None
                # call get_ep_meta() to get the RoboCasa env meta, includes prompt & layout_id, etcs
                if hasattr(env, "get_ep_meta"):
                    info = get_ep_meta(env, (info,))[-1]
                # return obs + info other than mere obs
                p.send((obs, info))
            elif cmd == "close":
                p.send(env.close())
                p.close()
                break
            elif cmd == "render":
                p.send(env.render(**data) if hasattr(env, "render") else None)
            elif cmd == "seed":
                if hasattr(env, "seed"):
                    p.send(env.seed(data))
                else:
                    env.reset(seed=data)
                    p.send(None)
            elif cmd == "getattr":
                p.send(getattr(env, data) if hasattr(env, data) else None)
            elif cmd == "setattr":
                setattr(env.unwrapped, data["key"], data["value"])
            elif cmd == "reconfigure":
                env.close()
                env = data.data()
                p.send(None)
            else:
                p.close()
                raise NotImplementedError(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        p.close()


class RobocasaSubprocEnvWorker(SubprocEnvWorker):
    """Subprocess environment worker for Robocasa.

    Based on metaworld's ReconfigureSubprocEnvWorker, but without the reconfigure
    functionality since robocasa doesn't need it.
    """

    def __init__(self, env_fn: Callable[[], gym.Env], share_memory: bool = False):
        # MuJoCo's EGL context is not fork-safe.  The previous default fork
        # context inherited Ray/OpenGL state and could abort in read_pixels
        # after several environment resets.  Spawn starts a clean interpreter
        # for each simulator worker while retaining an escape hatch for
        # controlled legacy experiments.
        start_method = os.environ.get("RLINF_ROBOCASA_START_METHOD", "spawn").lower()
        if start_method not in {"spawn", "fork", "forkserver"}:
            raise ValueError(
                "RLINF_ROBOCASA_START_METHOD must be one of spawn, fork, forkserver; "
                f"got {start_method!r}"
            )
        mp_context = get_context(start_method)
        self._mp_context = mp_context
        self._env_fn = env_fn
        self._restart_count = 0
        self._max_restarts = int(os.environ.get("RLINF_ROBOCASA_MAX_RESTARTS", "8"))
        self._eof_policy = os.environ.get(
            "RLINF_ROBOCASA_EOF_POLICY", "raise"
        ).lower()
        if self._eof_policy not in {"raise", "restart"}:
            raise ValueError(
                "RLINF_ROBOCASA_EOF_POLICY must be one of raise, restart; "
                f"got {self._eof_policy!r}"
            )
        self.parent_remote, self.child_remote = mp_context.Pipe()
        self.share_memory = share_memory
        self.buffer: Optional[Union[dict, tuple, ShArray]] = None
        if self.share_memory:
            dummy = env_fn()
            # robosuite task instances return an OrderedDict from reset() but
            # do not expose Gym's observation_space attribute.  Infer buffers
            # directly from a real observation and omit string/object metadata.
            reset_result = dummy.reset()
            obs_sample = (
                reset_result[0]
                if isinstance(reset_result, (tuple, list))
                and len(reset_result) == 2
                and isinstance(reset_result[1], dict)
                else reset_result
            )
            self.buffer = _build_shared_buffer(obs_sample, mp_context)
            dummy.close()
            del dummy
        args = (
            self.parent_remote,
            self.child_remote,
            CloudpickleWrapper(env_fn),
            self.buffer,
        )
        # Use our custom _worker function
        self.process = mp_context.Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()
        EnvWorker.__init__(self, env_fn)

    @staticmethod
    def _format_exitcode(exitcode: Optional[int]) -> str:
        if exitcode is None or exitcode >= 0:
            return str(exitcode)
        signum = -exitcode
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = f"SIG{signum}"
        return f"{exitcode} ({signal_name})"

    def _retire_process(self, *, graceful: bool) -> tuple[int, Optional[int]]:
        """Stop the current child and return its PID and final exit code."""
        process = self.process
        pid = int(process.pid or -1)
        if graceful and process.is_alive():
            try:
                self.parent_remote.send(["close", None])
                if self.parent_remote.poll(5.0):
                    self.parent_remote.recv()
            except (EOFError, BrokenPipeError, ConnectionResetError, OSError):
                pass
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        return pid, process.exitcode

    def _start_process(self) -> None:
        self.parent_remote, self.child_remote = self._mp_context.Pipe()
        args = (
            self.parent_remote,
            self.child_remote,
            CloudpickleWrapper(self._env_fn),
            self.buffer,
        )
        self.process = self._mp_context.Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()

    def _replace_process(self, *, reason: str, graceful: bool) -> None:
        """Replace a simulator process while retaining its real exit status."""
        old_pid, old_exitcode = self._retire_process(graceful=graceful)
        try:
            self.parent_remote.close()
        except OSError:
            pass
        try:
            self.child_remote.close()
        except OSError:
            pass
        self._start_process()
        print(
            "[RoboCasa] replaced worker; "
            f"reason={reason} old_pid={old_pid} "
            f"old_exitcode={self._format_exitcode(old_exitcode)} "
            f"new_pid={self.process.pid}",
            flush=True,
        )

    def _restart_after_eof(self) -> tuple[int, Optional[int]]:
        """Recreate a native simulator worker after an unexpected pipe close."""
        self._restart_count += 1
        old_pid, old_exitcode = self._retire_process(graceful=False)
        if self._restart_count > self._max_restarts:
            raise RuntimeError(
                "RoboCasa worker exceeded restart limit "
                f"({self._max_restarts}); old_pid={old_pid} "
                f"old_exitcode={self._format_exitcode(old_exitcode)}"
            )
        try:
            self.parent_remote.close()
        except Exception:
            pass
        try:
            self.child_remote.close()
        except Exception:
            pass
        self._start_process()
        print(
            "[RoboCasa] restarted worker after EOF; "
            f"count={self._restart_count} old_pid={old_pid} "
            f"old_exitcode={self._format_exitcode(old_exitcode)} "
            f"new_pid={self.process.pid}",
            flush=True,
        )
        return old_pid, old_exitcode

    def _recover_or_raise(self, operation: str):
        old_pid, old_exitcode = self._restart_after_eof()
        if self._eof_policy == "raise":
            raise RuntimeError(
                "RoboCasa simulator worker died during "
                f"{operation}; old_pid={old_pid} "
                f"old_exitcode={self._format_exitcode(old_exitcode)}. "
                "The replacement worker is ready, but the lost transition "
                "cannot be reconstructed safely."
            )

    def recv(self):
        try:
            return super().recv()
        except (EOFError, BrokenPipeError, ConnectionResetError):
            self._recover_or_raise("step")
            obs, _info = super().reset()
            return (
                obs,
                0.0,
                True,
                {"worker_restarted": True, "transition_valid": False},
            )

    def reset(self, **kwargs: Any):
        try:
            return super().reset(**kwargs)
        except (EOFError, BrokenPipeError, ConnectionResetError):
            self._recover_or_raise("reset")
            return super().reset(**kwargs)

    def reconfigure_env_fn(self, env_fn: Callable[[], gym.Env]) -> None:
        self._env_fn = env_fn
        restart_process = os.environ.get(
            "RLINF_ROBOCASA_RECONFIGURE_RESTART", "1"
        ).lower() not in {"0", "false", "no"}
        if restart_process:
            self._replace_process(reason="task_reconfigure", graceful=True)
            return None
        try:
            self.parent_remote.send(["reconfigure", CloudpickleWrapper(env_fn)])
            return self.parent_remote.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError):
            self._recover_or_raise("reconfigure")
            return None


class RobocasaSubprocEnv(SubprocVectorEnv):
    """Subprocess vectorized environment for Robocasa/Robosuite.

    Based on metaworld's ReconfigureSubprocEnv, adapted for robocasa environments.
    Uses subprocess isolation to avoid OpenGL context sharing issues in MuJoCo.
    """

    def __init__(self, env_fns: list[Callable[[], gym.Env]], **kwargs: Any) -> None:
        def worker_fn(fn: Callable[[], gym.Env]) -> RobocasaSubprocEnvWorker:
            # Camera observations are large and frequent.  Shared-memory
            # buffers avoid repeatedly pickling three 224x224 RGB frames over
            # the control pipe, which can grow Ray worker RSS over long PPO
            # rollouts.  Keep an opt-out for debugging incompatible spaces.
            share_memory = os.environ.get(
                "RLINF_ROBOCASA_SHARE_MEMORY", "1"
            ).lower() not in {"0", "false", "no"}
            return RobocasaSubprocEnvWorker(fn, share_memory=share_memory)

        BaseVectorEnv.__init__(self, env_fns, worker_fn, **kwargs)

    def reconfigure_env_fns(self, env_fns, id=None):
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        for j, i in enumerate(id):
            self.workers[i].reconfigure_env_fn(env_fns[j])
