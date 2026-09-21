"""Regression checks for the standalone RoboCasa365 evaluator."""

import sys
import types

import numpy as np
import torch

from deploy.server import Server
from eval_robocasa365.entry import configure_direct_gl_readback, parse_args


def test_direct_gl_readback_is_opt_in():
    assert parse_args([]).direct_gl_readback is False


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
