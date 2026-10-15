"""Redis idempotency keys with claim leases: at-least-once, effectively-once (SPEC 4.1)."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

# ============= Constants =============

DONE = "1"
PENDING_PREFIX = "pending:"
NEW, DUPLICATE, HELD = "new", "duplicate", "held"


class HeldClaimsError(RuntimeError):
    """Event ids are still claimed by another consumer after the wait; retry after redelivery."""


# ============= Store =============


@dataclass
class ClaimResult:
    """Outcome of claiming a batch of event ids."""

    new: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    held: list[str] = field(default_factory=list)


class IdempotencyStore:
    """
    Claims event ids before scoring and marks them done after the prediction is produced.

    A claim is `SET dg:seen:{id} pending:<owner> NX EX <lease>`. After the prediction is
    flushed to Kafka the key becomes `1` with `EX <ttl>` (172,800 s). A key equal to `1` means
    the event was already scored (duplicate). A pending key owned by another consumer is held:
    that owner is still working or died, in which case its lease expires and the event is
    reclaimed. This avoids losing events that a crashed consumer had claimed but not scored.

    Parameters
    ----------
    redis_client : redis.Redis
        Client with `decode_responses=True`.
    owner : str
        Unique id of this consumer instance.
    key_prefix : str
        Key prefix, `dg:seen:`.
    ttl_s : int
        Lifetime of a done key.
    lease_s : int
        Lifetime of a pending claim.
    """

    def __init__(
        self,
        redis_client,
        owner: str,
        key_prefix: str = "dg:seen:",
        ttl_s: int = 172_800,
        lease_s: int = 30,
    ):
        self.redis = redis_client
        self.owner = owner
        self.prefix = key_prefix
        self.ttl_s = int(ttl_s)
        self.lease_s = int(lease_s)

    def key(self, event_id: str) -> str:
        """Return the Redis key for an event id."""
        return f"{self.prefix}{event_id}"

    def claim(self, event_ids: list[str]) -> ClaimResult:
        """
        Try to claim each event id in one pipeline round trip.

        Parameters
        ----------
        event_ids : list[str]
            Event ids to claim.

        Returns
        -------
        result : ClaimResult
            Ids claimed by this owner, ids already done, and ids held by another owner.
        """
        result = ClaimResult()
        if not event_ids:
            return result
        token = f"{PENDING_PREFIX}{self.owner}"
        pipe = self.redis.pipeline(transaction=False)
        for event_id in event_ids:
            pipe.set(self.key(event_id), token, nx=True, ex=self.lease_s)
        acquired = pipe.execute()
        contested = [eid for eid, ok in zip(event_ids, acquired, strict=True) if not ok]
        if contested:
            values = self.redis.mget([self.key(eid) for eid in contested])
            for event_id, value in zip(contested, values, strict=True):
                if value == DONE:
                    result.duplicates.append(event_id)
                elif value == token:
                    # this owner claimed it earlier (redelivery within the lease)
                    result.new.append(event_id)
                elif value is None:
                    # expired between SET and GET; retry on the next claim
                    result.held.append(event_id)
                else:
                    result.held.append(event_id)
        result.new = [eid for eid, ok in zip(event_ids, acquired, strict=True) if ok] + result.new
        return result

    def claim_with_wait(
        self,
        event_ids: list[str],
        sleep: Callable[[float], None] = time.sleep,
        poll_s: float = 0.5,
        max_wait_s: float | None = None,
    ) -> ClaimResult:
        """
        Claim ids, waiting for held ids until they become done or their lease expires.

        Parameters
        ----------
        event_ids : list of str
            Ids to claim.
        sleep : callable
            Sleep function, injectable for tests.
        poll_s : float
            Seconds between retries for held ids.
        max_wait_s : float or None
            Upper bound on waiting; defaults to twice the lease.

        Returns
        -------
        result : ClaimResult
            `held` is empty unless `max_wait_s` elapsed.
        """
        max_wait_s = 2 * self.lease_s if max_wait_s is None else max_wait_s
        total = self.claim(event_ids)
        waited = 0.0
        while total.held and waited < max_wait_s:
            sleep(poll_s)
            waited += poll_s
            retry = self.claim(total.held)
            total.new += retry.new
            total.duplicates += retry.duplicates
            total.held = retry.held
        return total

    def renew(self, event_ids: list[str]) -> None:
        """Extend the lease of this owner's pending claims (called between scoring chunks)."""
        if not event_ids:
            return
        token = f"{PENDING_PREFIX}{self.owner}"
        keys = [self.key(event_id) for event_id in event_ids]
        values = self.redis.mget(keys)
        pipe = self.redis.pipeline(transaction=False)
        for key, value in zip(keys, values, strict=True):
            if value == token:
                pipe.expire(key, self.lease_s)
        pipe.execute()

    def mark_done(self, event_ids: list[str]) -> None:
        """Mark ids as scored with the long TTL; called after predictions are flushed."""
        if not event_ids:
            return
        pipe = self.redis.pipeline(transaction=False)
        for event_id in event_ids:
            pipe.set(self.key(event_id), DONE, ex=self.ttl_s)
        pipe.execute()

    def release(self, event_ids: list[str]) -> None:
        """Delete this owner's pending claims (events sent to the DLQ stay replayable)."""
        token = f"{PENDING_PREFIX}{self.owner}"
        for event_id in event_ids:
            key = self.key(event_id)
            if self.redis.get(key) == token:
                self.redis.delete(key)
