# Copyright (C) 2026 Xiaomi Corporation.
import pickle
import socket
import struct
import time
import argparse
import traceback

import torch
from tqdm import tqdm
from transformers import AutoModel


def attention_backend():
    """Use FlashAttention when installed, otherwise keep deployment portable."""
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return "eager"
    return "flash_attention_2"


class Server:
    def __init__(self, model_path, host, port, policy_checkpoint=None, device="cuda:0"):
        self.host = host
        self.port = port
        self.device = torch.device(device)

        print("Loading model...")
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation=attention_backend(),
            dtype=torch.bfloat16,
        ).to(device=self.device, dtype=torch.bfloat16)
        if policy_checkpoint:
            print(f"Loading PPO policy checkpoint: {policy_checkpoint}")
            state = torch.load(policy_checkpoint, map_location="cpu", weights_only=False)
            if isinstance(state, dict):
                state = state.get("module", state.get("state_dict", state))
            native_state = {
                key[len("xr1_model."):]: value
                for key, value in state.items()
                if key.startswith("xr1_model.")
            }
            if not native_state:
                raise ValueError(
                    "PPO checkpoint has no xr1_model.* keys: "
                    f"{policy_checkpoint}"
                )
            missing, unexpected = self.model.load_state_dict(native_state, strict=False)
            print(
                "PPO checkpoint loaded: "
                f"native={len(native_state)} missing={len(missing)} unexpected={len(unexpected)}"
            )
            del state, native_state
        print("Model loaded.")

    def _recv_all(self, conn, length):
        data = b""
        while len(data) < length:
            packet = conn.recv(length - len(data))
            if not packet:
                return None
            data += packet
        return data

    def serve(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(1)
            print(f"Server running on {self.host}:{self.port}...")

            while True:
                conn, addr = server_socket.accept()

                try:
                    request_count = 0
                    with tqdm(desc="Processing Requests", unit=" req") as pbar:
                        while True:
                            data_len_bytes = self._recv_all(conn, 4)
                            if not data_len_bytes:
                                break
                            data_len = struct.unpack(">I", data_len_bytes)[0]

                            data = self._recv_all(conn, data_len)
                            if not data:
                                break

                            tic = time.time()

                            input_data = pickle.loads(data)
                            robot_type = input_data["task_id"]
                            data = {key: (value.to(device=self.model.device, dtype=self.model.dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value.to(device=self.model.device) if isinstance(value, torch.Tensor) else value) for key, value in input_data.items()}

                            outputs = self.model(**data)

                            response = pickle.dumps(outputs.actions.cpu())
                            conn.sendall(struct.pack(">I", len(response)) + response)

                            toc = time.time()
                            request_count += 1
                            pbar.update(1)
                            pbar.set_postfix(
                                {
                                    "avg_time": f"{(toc - tic) * 1000:.2f}ms",
                                }
                            )

                except Exception as e:
                    print(f"Error handling connection: {e}")
                    traceback.print_exc()
                finally:
                    conn.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model dir.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10086,
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=str,
        default=None,
        help="Optional RLinf full_weights.pt; loads native xr1_model.* weights.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device in the job-local CUDA namespace.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    server = Server(
        model_path=args.model,
        host=args.host,
        port=args.port,
        policy_checkpoint=args.policy_checkpoint,
        device=args.device,
    )
    server.serve()
