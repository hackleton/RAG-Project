"""Cache keys.

**The rule this module exists to enforce: a cached retrieval result is keyed on
the caller's access scope.** A key that omits it is not a performance bug, it is
an access-control bypass - the first caller to ask a question populates an entry
the next caller reads, and the pre-search filter that Sprint 3 built to stop a
chunk ever being retrieved is bypassed by a cache hit.

So `scope_digest` is derived from `dataclasses.fields(AccessFilter)` rather than
from a hand-written list of the fields that exist today. A field added to
`AccessFilter` later is included automatically; a hand-written list would keep
working, silently, while no longer describing the scope it claims to.

Two encoding details carry weight:

* `None` and the empty set are distinct. On `patient_ids`, `None` means an
  administrator with no restriction and `frozenset()` means a caller entitled
  to no patient at all. Collapsing them would key the widest scope and the
  narrowest to the same entry.
* Keys are digests, so no question text reaches Redis. The value stored against
  an embedding key is a vector and against a ranking key a list of chunk ids -
  no document text, and nothing that reconstructs what was asked.

`PREFIX` carries a version. Any change to what a value means - the embedding
model's output shape, the fields in a cached ranking - is a bump, because
poisoning a shared cache with entries of the previous shape is a worse failure
than a cold cache.
"""

from __future__ import annotations

import hashlib
from dataclasses import fields
from typing import Any

from app.providers.vectorstore import AccessFilter

VERSION = "v1"

# 128 bits of a sha256. Collisions are not a security boundary here - the scope
# is inside the digest, so a collision returns another scope's ranking - but at
# 2^-128 it is not a risk anyone can act on either.
DIGEST_CHARS = 32


def digest(*parts: str) -> str:
    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:DIGEST_CHARS]


def _canonical(value: Any) -> str:
    if value is None:
        # Unrestricted. Deliberately not "", which is the empty scope.
        return "*"
    if isinstance(value, frozenset | set):
        return ",".join(sorted(str(item) for item in value))
    return str(value)


def scope_digest(access: AccessFilter) -> str:
    """A stable digest of every field of the caller's access filter."""
    parts = [f"{field.name}={_canonical(getattr(access, field.name))}" for field in fields(access)]
    return digest(*parts)


def key(namespace: str, *parts: str, prefix: str) -> str:
    return f"{prefix}:{VERSION}:{namespace}:{digest(*parts)}"


def embedding_key(*, prefix: str, model: str, dimensions: int, text: str) -> str:
    """Keyed on the model as well as the text.

    Two models produce different vectors for the same string, and a deployment
    that changes `EMBEDDING_MODEL` without this would search one model's index
    with another model's vectors - which does not error, it just quietly
    returns nonsense.
    """
    return key("embed", model, str(dimensions), text, prefix=prefix)


def ranking_key(
    *, prefix: str, query: str, config_digest: str, access: AccessFilter, generation: int
) -> str:
    """Keyed on query, retrieval configuration, access scope and index generation."""
    return key(
        "rank",
        query.strip().lower(),
        config_digest,
        scope_digest(access),
        str(generation),
        prefix=prefix,
    )


def generation_key(*, prefix: str) -> str:
    """The index generation counter. Not a digest - it is read and incremented."""
    return f"{prefix}:{VERSION}:index:generation"


def throttle_key(*, prefix: str, email: str, ip: str | None) -> str:
    """Failed sign-ins for one email and IP pair (Sprint 6, decision 38)."""
    return key("login", email.strip().lower(), ip or "-", prefix=prefix)
