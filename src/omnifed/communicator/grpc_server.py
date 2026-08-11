# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
from collections import defaultdict
from typing import Any, Dict
import warnings
import traceback

import rich.repr
import torch

from ..utils import print
from . import grpc_pb2, grpc_pb2_grpc
from .base import AggregationOp
from .utils import get_msg_info, proto_to_tensordict, tensordict_to_proto, proto_to_tensordict_extended
from .utils import (
    aggregation_metric_for_communicate_params,
    compress_message_tensors,
    compressor_proto_name,
    extract_tensordict,
)


@rich.repr.auto
class GrpcServer(grpc_pb2_grpc.GrpcServerServicer):
    """
    gRPC server implementation for federated learning communication coordination.

    Coordinates broadcast and aggregation operations across multiple clients.
    Handles session management, client synchronization, and tensor aggregation
    with support for multiple concurrent FL rounds.
    """

    def __init__(
        self,
        world_size: int,
        compressor=None,
        communicate_params: bool = True,
        normalize_by_total_samples: bool = False,
    ):
        """
        Initialize gRPC server for federated learning coordination.

        Args:
            world_size: Total number of FL participants (including server)
            communicate_params: Aggregate model parameters when True, else gradients
            normalize_by_total_samples: After SUM, divide by summed client ``num_samples``
                (Path B sample-weighted grads). Path A pre-scales on clients and leaves False.
        """
        print(f"world_size={world_size}")

        # Core configuration
        self.world_size = world_size
        self.communicate_params = bool(communicate_params)
        self.normalize_by_total_samples = bool(normalize_by_total_samples)
        self.registered_clients = set()
        # self.compressor = QSGDQuantCompression()
        self.lock = threading.Lock()
        self.model = None

        # Aggregation session management
        self.current_aggregation_session = 0
        self.aggregation_state: dict[int, dict[str, Any]] = defaultdict(
            self._new_aggregation_session_state
        )

        self.compressor = compressor
        # self.compressor = None

    @property
    def aggregation_metric(self) -> str:
        return aggregation_metric_for_communicate_params(self.communicate_params)

    @staticmethod
    def _new_aggregation_session_state() -> dict[str, Any]:
        return {
            "data": {},
            "result": None,
            "event": threading.Event(),
            "reduction_type": None,
            "total_samples": 0,
            "participants": set(),
            "results_delivered": set(),
        }

    def _session_has_participant(
        self, session_state: dict[str, Any], participant_id: str
    ) -> bool:
        if participant_id in session_state["data"]:
            return True
        return participant_id in session_state.get("participants", ())

    def _release_aggregation_inputs(self, session_state: dict[str, Any]) -> None:
        """Drop submitted client/server tensors once aggregation result is ready."""
        session_state["participants"] = set(session_state["data"].keys())
        session_state["data"].clear()

    def mark_aggregation_result_delivered(
        self, session_id: int, participant_id: str
    ) -> None:
        """Record that a participant fetched the result; drop session when all have."""
        with self.lock:
            self._mark_aggregation_result_delivered_locked(session_id, participant_id)

    def _mark_aggregation_result_delivered_locked(
        self, session_id: int, participant_id: str
    ) -> None:
        session_state = self.aggregation_state.get(session_id)
        if session_state is None:
            return
        delivered = session_state.setdefault("results_delivered", set())
        delivered.add(participant_id)
        participants = session_state.get("participants") or set()
        if participants and delivered >= participants:
            self._drop_aggregation_session(session_id)

    def _drop_aggregation_session(self, session_id: int) -> None:
        """Remove a completed aggregation session and free retained tensors."""
        session_state = self.aggregation_state.pop(session_id, None)
        if session_state is None:
            return
        session_state["data"].clear()
        session_state["result"] = None
        session_state["participants"] = set()
        session_state["results_delivered"] = set()
        session_state["event"].clear()
        print(f"Dropped aggregation session {session_id}")

    def _reset_aggregation_session(self, session_id: int) -> None:
        """Clear a stuck/partial aggregation session so the next op can proceed."""
        session_state = self.aggregation_state[session_id]
        session_state["data"].clear()
        session_state["reduction_type"] = None
        session_state["total_samples"] = 0
        session_state["result"] = None
        session_state["participants"] = set()
        session_state["results_delivered"] = set()
        session_state["event"].clear()

    def _abandon_current_session(self) -> None:
        """Drop the active session and advance (partial MAX/SUM mix or failed submit)."""
        sid = int(self.current_aggregation_session)
        self._reset_aggregation_session(sid)
        self.current_aggregation_session = sid + 1
        print(f"Abandoned aggregation session {sid}; next session={self.current_aggregation_session}")


        # Broadcast state storage
        self._broadcast_state = {}

    def get_broadcast_state(self) -> Dict[str, torch.Tensor]:
        """
        Get current broadcast state with thread-safe tensor cloning.

        Returns:
            Deep copy of broadcast tensors to prevent race conditions
        """
        with self.lock:
            return {
                key: tensor.clone() for key, tensor in self._broadcast_state.items()
            }

    def set_broadcast_state(self, tensordict: Dict[str, torch.Tensor]):
        """
        Store broadcast state for distribution to clients.

        Args:
            tensordict: Tensors to broadcast (typically global model)
        """
        with self.lock:
            self._broadcast_state = tensordict
            self.model = tensordict
        print(get_msg_info(tensordict))

    # def store_model(self, tensordict: Dict[str, torch.Tensor]):
    #     """
    #     Store broadcast state for distribution to clients.

    #     Args:
    #         tensordict: Tensors to broadcast (typically global model)
    #     """
    #     with self.lock:
    #         self.model = tensordict
    #     print("Server storing the model in itself")
    #     print(get_msg_info(tensordict))

    def perform_aggregation_if_ready(
        self, session_state: Dict, current_session: int, is_model_communicated=False
    ) -> bool:
        """
        Execute aggregation when all clients have submitted data.

        Args:
            session_state: Current aggregation session data
            current_session: Session identifier

        Returns:
            True if aggregation was performed, False if still waiting
        """
        submitted_count = len(session_state["data"])

        print(f"Waiting for clients ({submitted_count}/{self.world_size} ready)")

        if submitted_count == self.world_size:
            print(f"All {self.world_size} clients ready - beginning aggregation")

            first_data = next(iter(session_state["data"].values()))
            aggregated_tensors = {}

            with torch.no_grad():
                reduction_type = session_state["reduction_type"]
                if reduction_type is None:
                    raise ValueError(
                        f"No reduction type set for session {current_session}"
                    )

                # Initialize aggregated tensors for each key
                aggregated_tensors = {}

                if reduction_type == AggregationOp.MAX.value:
                    # MAX reduction: element-wise maximum
                    for key, tensor in first_data.items():
                        all_tensors = [
                            client_data[key]
                            for client_data in session_state["data"].values()
                        ]
                        aggregated_tensors[key] = torch.max(
                            torch.stack(all_tensors), dim=0
                        )[0]

                elif reduction_type in (
                    AggregationOp.SUM.value,
                    AggregationOp.MEAN.value,
                ):
                    # SUM/MEAN reduction: sum all contributions
                    for key, tensor in first_data.items():
                        aggregated_tensors[key] = torch.zeros_like(tensor).cpu()

                    # Sum all client contributions
                    for client_data in session_state["data"].values():
                        for key, tensor in client_data.items():
                            if key in aggregated_tensors:
                                # Ensure tensor is on CPU
                                # tensor = tensor.to("cpu")

                                # # Ensure same dtype as accumulator
                                # tensor = tensor.to(aggregated_tensors[key].dtype)

                                aggregated_tensors[key] += tensor.cpu()

                    # Convert sum to mean if needed
                    if reduction_type == AggregationOp.MEAN.value:
                        for tensor in aggregated_tensors.values():
                            tensor /= self.world_size
                    elif (
                        reduction_type == AggregationOp.SUM.value
                        and self.normalize_by_total_samples
                    ):
                        total_samples = int(session_state.get("total_samples", 0))
                        # Grad round-end submits num_samples; epoch-heartbeat scalars do not.
                        if total_samples >= 1:
                            for tensor in aggregated_tensors.values():
                                tensor /= total_samples

                else:
                    raise ValueError(f"Unknown reduction type: {reduction_type}")
            if is_model_communicated:
                self.model = aggregated_tensors
            session_state["result"] = aggregated_tensors
            self._release_aggregation_inputs(session_state)
            session_state["event"].set()
            print(
                f"Aggregated {len(aggregated_tensors)} tensors using {reduction_type}"
            )
            self.current_aggregation_session += 1
            return True
        return False

    def GetBroadcastState(self, request, context):
        """
        gRPC endpoint: Send broadcast state to requesting client.

        Args:
            request: ClientInfo with client identifier
            context: gRPC context (unused)

        Returns:
            OperationResponse with tensor data or not-ready status
        """
        print(f"request.client_id={request.client_id}")

        with self.lock:
            if self._broadcast_state:
                proto_tensordict = tensordict_to_proto(self._broadcast_state)
                return grpc_pb2.OperationResponse(
                    tensor_dict=proto_tensordict, is_ready=True
                )
            else:
                return grpc_pb2.OperationResponse(is_ready=False)

    def _create_aggregation_result_response(self, session_id: int):
        """
        Create gRPC response containing aggregated tensor results.

        Args:
            session_id: Aggregation session identifier

        Returns:
            OperationResponse with aggregated tensors or not-ready status
        """
        if session_id in self.aggregation_state:
            session_state = self.aggregation_state[session_id]
            if session_state["result"] is not None:
                aggregated_tensors = session_state["result"]
                # Grad sessions carry num_samples>0; sample/BN sessions stay dense.
                active_compressor = (
                    self.compressor
                    if int(session_state.get("total_samples", 0)) > 0
                    else None
                )
                compressed_tensordict = compress_message_tensors(
                    aggregated_tensors,
                    active_compressor,
                    self.aggregation_metric,
                )
                compressor_name = compressor_proto_name(active_compressor)
                if isinstance(aggregated_tensors, torch.Tensor):
                    compressor_name = None
                    compressed_tensordict = aggregated_tensors
                proto_tensordict = tensordict_to_proto(compressed_tensordict, compressor_name)
                # proto_tensordict = tensordict_to_proto(aggregated_tensors)
                return grpc_pb2.OperationResponse(
                    tensor_dict=proto_tensordict, is_ready=True
                )
        return grpc_pb2.OperationResponse(is_ready=False)

    def SubmitForAggregation(self, request, context):
        """
        gRPC endpoint: Receive client tensors for distributed aggregation.

        Stores client contributions and triggers aggregation when all
        clients have submitted their data.

        Args:
            request: AggregationRequest with client data and reduction type
            context: gRPC context (unused)

        Returns:
            StatusResponse indicating success or failure
        """
        with self.lock:
            client_id = request.client_id
            current_session = self.current_aggregation_session
            # print(
            #     f"Client {client_id} submitting {len(request.tensor_dict.entries)} tensors"
            # )

            try:
                # Deserialize tensors; keep on CPU on the server to avoid GPU retention.
                data, is_model_communicated = proto_to_tensordict_extended(
                    request.tensor_dict,
                    overlay_base=None,
                )
                data = {key: tensor.cpu() for key, tensor in data.items()}
                # print(f"Now the data is ready: data = {data}")
                session_state = self.aggregation_state[current_session]

                if (
                    session_state["reduction_type"] is not None
                    and session_state["reduction_type"] != request.reduction_type
                ):
                    if len(session_state["data"]) < self.world_size:
                        print(
                            f"Partial session {current_session} reduction mismatch "
                            f"(expected={session_state['reduction_type']} "
                            f"got={request.reduction_type}) — resetting session"
                        )
                        self._reset_aggregation_session(current_session)
                    else:
                        raise ValueError(
                            f"Reduction mismatch | expected={session_state['reduction_type']} "
                            f"got={request.reduction_type}"
                        )

                if session_state["reduction_type"] is None:
                    session_state["reduction_type"] = request.reduction_type

                session_state["data"][client_id] = data
                session_state["total_samples"] = int(
                    session_state.get("total_samples", 0)
                ) + int(request.num_samples)
                data_count = len(session_state["data"])
                print(
                    f"Received from client {client_id} ({data_count}/{self.world_size} ready)"
                )

                self.perform_aggregation_if_ready(session_state, current_session, is_model_communicated)
                return grpc_pb2.StatusResponse(success=True)

            except Exception as e:
                print(f"Error is: {e}")
                traceback.print_exc()
                warnings.warn(
                    f"Failed to process aggregation submission from client {client_id} | {e}",
                    RuntimeWarning,
                )
                try:
                    if len(session_state.get("data", {})) < self.world_size:
                        self._abandon_current_session()
                except Exception:
                    pass
                return grpc_pb2.StatusResponse(success=False)

    def GetAggregationResult(self, request, context):
        """
        gRPC endpoint: Send aggregation result to requesting client.

        Waits for aggregation to complete if necessary, then returns
        the aggregated tensors to the requesting client.

        Args:
            request: ClientInfo with client identifier
            context: gRPC context (unused)

        Returns:
            OperationResponse with aggregated tensors or error status
        """
        client_id = request.client_id

        with self.lock:
            target_session = None
            for session_id in sorted(self.aggregation_state.keys(), reverse=True):
                session_state = self.aggregation_state[session_id]
                if not self._session_has_participant(session_state, client_id):
                    continue
                if client_id in session_state.get("results_delivered", set()):
                    continue
                target_session = session_id
                break
            if target_session is None:
                warnings.warn(
                    f"Client {client_id} has no data submitted for aggregation",
                    RuntimeWarning,
                )
                return grpc_pb2.OperationResponse(is_ready=False)

        print(f"Client {client_id} requesting aggregation result")

        try:
            session_state = self.aggregation_state[target_session]
            with self.lock:
                if session_state["result"] is not None:
                    print(f"Sending aggregated model to client {client_id}")
                    response = self._create_aggregation_result_response(target_session)
                    self._mark_aggregation_result_delivered_locked(
                        target_session, client_id
                    )
                    return response

            print(f"Client {client_id} waiting for aggregation to complete")
            session_state["event"].wait()
            print(f"Aggregation complete for client {client_id}")
            with self.lock:
                if session_state["result"] is not None:
                    print(f"Sending aggregated model to client {client_id}")
                    response = self._create_aggregation_result_response(target_session)
                    self._mark_aggregation_result_delivered_locked(
                        target_session, client_id
                    )
                    return response
            return grpc_pb2.OperationResponse(is_ready=False)

        except Exception as e:
            warnings.warn(
                f"Failed to get aggregation result for client {client_id} | {e}",
                RuntimeWarning,
            )
            return grpc_pb2.OperationResponse(is_ready=False)

    def RegisterClient(self, request, context):
        """
        gRPC endpoint: Register client connection and track participant count.

        Args:
            request: ClientInfo with unique client identifier
            context: gRPC context (unused)

        Returns:
            StatusResponse confirming successful registration
        """
        with self.lock:
            self.registered_clients.add(request.client_id)
            total_clients = len(self.registered_clients)
            print(f"{total_clients}/{self.world_size} total")
            return grpc_pb2.StatusResponse(success=True)
