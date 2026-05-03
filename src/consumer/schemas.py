"""Pydantic schema for incoming Kafka transaction events."""

from pydantic import BaseModel, Field


class TransactionEvent(BaseModel):
    """
    Validated Kafka transaction event.

    Any message that fails validation against this model is routed to
    the dead letter queue topic with the validation error attached as a header.
    """

    cc_num: str
    merchant: str
    category: str
    amt: float = Field(gt=0)
    gender: str
    city: str
    state: str
    zip: str
    lat: float
    long: float
    city_pop: int
    job: str
    dob: str
    merch_lat: float
    merch_long: float
    trans_date_trans_time: str
