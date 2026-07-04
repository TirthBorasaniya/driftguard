"""Pydantic schema for incoming Kafka network flow events."""

from pydantic import BaseModel

# ============= Network Flow Event =============


class NetworkFlowEvent(BaseModel):
    """
    Validated Kafka network flow event.

    Mirrors the NetworkFlowEvent Avro contract in
    src/schemas/network_flow_event.avsc. Any message that fails validation
    against this model is routed to the dead letter queue topic with the
    validation error attached as a header.
    """

    event_id: str
    flow_id: str
    timestamp_utc: int
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    flow_duration: float
    flow_bytes_per_sec: float
    flow_packets_per_sec: float
    total_fwd_packets: float
    total_bwd_packets: float
    total_length_fwd_packets: float
    total_length_bwd_packets: float
    packet_length_mean: float
    packet_length_std: float
    flow_iat_mean: float
    syn_flag_count: float
    label: str
    label_binary: int
