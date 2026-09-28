"""Regression checks for the standalone RoboCasa365 evaluator."""

import sys
import types
from pathlib import Path

import numpy as np
import torch

from deploy.server import Server
from eval_robocasa365.entry import (
    configure_direct_gl_readback,
    episode_index,
    parse_args,
    validate_args,
)


def test_direct_gl_readback_is_opt_in():
    assert parse_args([]).direct_gl_readback is False


def test_seed_stride_keeps_short_screen_as_full_manifest_prefix():
    short = parse_args(["--num-trials", "5", "--seed-stride", "20"])
    full = parse_args(["--num-trials", "20", "--seed-stride", "20"])
    validate_args(short)
    validate_args(full)

    for task_index in (0, 1, 17, 49):
        assert [episode_index(short, task_index, episode) for episode in range(5)] == [
            episode_index(full, task_index, episode) for episode in range(5)
        ]


def test_isolated_eval_exposes_explicit_readback_switch():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_robocasa365_policy_eval_isolated.sbatch"
    ).read_text()

    assert "EVAL_DIRECT_GL_READBACK:-0" in script
    assert "readback_arg=--no-direct-gl-readback" in script
    assert "readback_arg=--direct-gl-readback" in script


def test_step1000_gated_eval_keeps_gate_and_full_run_on_one_allocation():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_step1000_remaining_eval_gated.sbatch"
    ).read_text()

    assert "#SBATCH --exclude=gnho011,gnho020" in script
    assert "EVAL_TASKS=StoreLeftoversInBowl" in script
    assert script.count("EVAL_DIRECT_GL_READBACK=0") == 2
    assert "bash scripts/run_robocasa365_policy_eval_isolated.sbatch" in script
    assert script.index("NATIVE_READBACK_GATE_PASSED") < script.index("POLICY_NAME=step1000_fill3")


def test_server_preserves_explicit_job_local_device(monkeypatch):
    model = types.SimpleNamespace(
        to=lambda *, device, dtype: (setattr(model, "device", device) or model)
    )
    monkeypatch.setattr(
        "deploy.server.AutoModel.from_pretrained", lambda *args, **kwargs: model
    )

    server = Server("model", "127.0.0.1", 10086, device="cuda:3")

    assert server.device == torch.device("cuda:3")
    assert model.device == torch.device("cuda:3")


def test_direct_gl_readback_restores_state_and_is_not_wrapped_twice(monkeypatch):
    class FakeGL:
        GL_NO_ERROR = 0
        GL_READ_FRAMEBUFFER = 1
        GL_DRAW_FRAMEBUFFER = 2
        GL_READ_FRAMEBUFFER_BINDING = 3
        GL_DRAW_FRAMEBUFFER_BINDING = 4
        GL_READ_BUFFER = 5
        GL_PACK_ALIGNMENT = 6
        GL_PIXEL_PACK_BUFFER_BINDING = 7
        GL_PIXEL_PACK_BUFFER = 8
        GL_COLOR_ATTACHMENT0 = 9
        GL_COLOR_BUFFER_BIT = 10
        GL_NEAREST = 11
        GL_STREAM_READ = 12
        GL_RGB = 13
        GL_UNSIGNED_BYTE = 14

        def __init__(self):
            self.values = {
                self.GL_READ_FRAMEBUFFER_BINDING: 101,
                self.GL_DRAW_FRAMEBUFFER_BINDING: 102,
                self.GL_READ_BUFFER: 103,
                self.GL_PACK_ALIGNMENT: 4,
                self.GL_PIXEL_PACK_BUFFER_BINDING: 105,
            }

        def glGetError(self):
            return self.GL_NO_ERROR

        def glGetIntegerv(self, name):
            return self.values[name]

        def glBindFramebuffer(self, target, value):
            binding = (
                self.GL_READ_FRAMEBUFFER_BINDING
                if target == self.GL_READ_FRAMEBUFFER
                else self.GL_DRAW_FRAMEBUFFER_BINDING
            )
            self.values[binding] = value

        def glReadBuffer(self, value):
            self.values[self.GL_READ_BUFFER] = value

        def glPixelStorei(self, name, value):
            self.values[name] = value

        def glBindBuffer(self, target, value):
            assert target == self.GL_PIXEL_PACK_BUFFER
            self.values[self.GL_PIXEL_PACK_BUFFER_BINDING] = value

        def glGenBuffers(self, count):
            assert count == 1
            return 999

        def glBufferData(self, *args):
            pass

        def glReadPixels(self, *args):
            pass

        def glGetBufferSubData(self, target, offset, count):
            return bytes(count)

        def glDeleteBuffers(self, count, buffers):
            assert count == 1 and buffers == [999]

        def glBlitFramebuffer(self, *args):
            pass

    fake_gl = FakeGL()
    monkeypatch.setitem(sys.modules, "OpenGL", types.SimpleNamespace(GL=fake_gl))

    context = types.SimpleNamespace(
        gl_ctx=types.SimpleNamespace(make_current=lambda: None),
        con=types.SimpleNamespace(
            offSamples=4,
            offFBO=201,
            offFBO_r=202,
            offWidth=2,
            offHeight=2,
        ),
        read_pixels=lambda width, height, depth=False, segmentation=False: np.zeros(
            (height, width, 3), dtype=np.uint8
        ),
    )
    env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            env=types.SimpleNamespace(
                sim=types.SimpleNamespace(_render_context_offscreen=context)
            )
        )
    )
    original_state = dict(fake_gl.values)

    configure_direct_gl_readback(env, True)
    installed_method = context.read_pixels
    configure_direct_gl_readback(env, True)
    image = context.read_pixels(2, 2)

    assert context.read_pixels is installed_method
    assert np.array_equal(image, np.zeros((2, 2, 3), dtype=np.uint8))
    assert fake_gl.values == original_state
