"""Community Attack Registry — curated adversarial test cases.

Provides a bundled registry of 25 high-signal attack cases across all six
toki categories, with OWASP LLM Top 10 2025 tags, severity ratings, and
provenance metadata. The registry ships inside the package (no network
required). An optional remote pull supports pinned-SHA-256 integrity
verification.
"""
from __future__ import annotations

import hashlib
import json
import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_BUNDLED_PATH = Path(__file__).parent / "data" / "community_registry.json"

VALID_SEVERITIES = frozenset({"critical", "high", "medium", "low"})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommunityAttack:
    """One curated community attack case.

    Attributes
    ----------
    id:
        Stable community ID (e.g. "com-jb-001").
    text:
        The adversarial prompt text.
    category:
        Attack category: jailbreak / injection / edge_case / boundary /
        indirect / agentic.
    tags:
        Technique tags (e.g. ["DAN", "persona-swap"]).
    severity:
        "critical" | "high" | "medium" | "low"
    owasp_tag:
        OWASP LLM Top 10 2025 reference (e.g. "OWASP-LLM01:2025").
    author:
        Contributor identifier.
    source:
        Where this case originated (e.g. "toki-curated").
    expected_refusal:
        True when a safe model should refuse this prompt.
    """

    id: str
    text: str
    category: str
    tags: List[str]
    severity: str
    owasp_tag: str
    author: str
    source: str
    expected_refusal: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "category": self.category,
            "tags": list(self.tags),
            "severity": self.severity,
            "owasp_tag": self.owasp_tag,
            "author": self.author,
            "source": self.source,
            "expected_refusal": self.expected_refusal,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class CommunityRegistry:
    """Loaded community attack registry.

    Attributes
    ----------
    version:
        Registry schema version string.
    updated:
        ISO-8601 date the registry was last updated.
    description:
        Human-readable description.
    sha256:
        SHA-256 of the compact-JSON attacks array; verified on load.
    attacks:
        All CommunityAttack entries.
    """

    version: str
    updated: str
    description: str
    sha256: str
    attacks: List[CommunityAttack] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.attacks)

    def filter(
        self,
        *,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> List[CommunityAttack]:
        """Return attacks matching all provided filters (AND logic).

        Parameters
        ----------
        category:
            Filter by attack category.
        tag:
            Filter to attacks whose ``tags`` list contains this tag.
        severity:
            Filter by severity level.
        """
        result = self.attacks
        if category is not None:
            result = [a for a in result if a.category == category]
        if tag is not None:
            result = [a for a in result if tag in a.tags]
        if severity is not None:
            result = [a for a in result if a.severity == severity]
        return result

    def stats(self) -> Dict[str, object]:
        """Return summary stats: total, by_category, by_severity."""
        by_cat: Dict[str, int] = {}
        by_sev: Dict[str, int] = {}
        for a in self.attacks:
            by_cat[a.category] = by_cat.get(a.category, 0) + 1
            by_sev[a.severity] = by_sev.get(a.severity, 0) + 1
        return {
            "total": self.total,
            "version": self.version,
            "updated": self.updated,
            "by_category": by_cat,
            "by_severity": by_sev,
        }


# ---------------------------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------------------------


def _verify_sha256(attacks_data: list, declared: str) -> bool:
    """Verify SHA-256 of compact attacks JSON matches the declared value."""
    computed = hashlib.sha256(
        json.dumps(attacks_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return computed == declared


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _parse_registry(data: dict, *, verify: bool = True) -> CommunityRegistry:
    """Parse a manifest dict into a CommunityRegistry.

    Parameters
    ----------
    data:
        Parsed JSON manifest dict.
    verify:
        When True, verify the SHA-256 of the attacks array. Logs a warning
        (but does not raise) if the digest mismatches.
    """
    if verify and "sha256" in data:
        if not _verify_sha256(data.get("attacks", []), data["sha256"]):
            logger.warning(
                "CommunityRegistry: SHA-256 mismatch — registry may be tampered"
            )

    attacks = [
        CommunityAttack(
            id=item["id"],
            text=item["text"],
            category=item["category"],
            tags=list(item.get("tags", [])),
            severity=item.get("severity", "medium"),
            owasp_tag=item.get("owasp_tag", ""),
            author=item.get("author", "unknown"),
            source=item.get("source", "unknown"),
            expected_refusal=bool(item.get("expected_refusal", True)),
        )
        for item in data.get("attacks", [])
    ]

    return CommunityRegistry(
        version=data.get("version", "0.0.0"),
        updated=data.get("updated", ""),
        description=data.get("description", ""),
        sha256=data.get("sha256", ""),
        attacks=attacks,
    )


def load_bundled() -> CommunityRegistry:
    """Load the registry bundled with this package. Never makes network calls."""
    raw = _BUNDLED_PATH.read_text(encoding="utf-8")
    return _parse_registry(json.loads(raw))


def load_remote(
    url: str,
    *,
    expected_sha256: Optional[str] = None,
    timeout: float = 10.0,
) -> CommunityRegistry:
    """Fetch and parse a remote community registry manifest.

    Parameters
    ----------
    url:
        HTTPS URL of the remote registry JSON.
    expected_sha256:
        When provided, the attacks array SHA-256 must match this value.
        Raises ``ValueError`` on mismatch (stronger than the bundled
        warning-only behaviour).
    timeout:
        HTTP request timeout in seconds (default: 10).

    Raises
    ------
    ValueError:
        SHA-256 mismatch when ``expected_sha256`` is provided.
    urllib.error.URLError:
        Network or HTTP error.
    """
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        raw = resp.read().decode("utf-8")

    data = json.loads(raw)

    if expected_sha256 is not None:
        attacks_data = data.get("attacks", [])
        computed = hashlib.sha256(
            json.dumps(attacks_data, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        if computed != expected_sha256:
            raise ValueError(
                f"Remote registry SHA-256 mismatch: "
                f"expected {expected_sha256}, got {computed}"
            )

    return _parse_registry(data, verify=expected_sha256 is None)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_cached_registry: Optional[CommunityRegistry] = None


def get_registry(*, reload: bool = False) -> CommunityRegistry:
    """Return the bundled community registry (cached after first load).

    Parameters
    ----------
    reload:
        Force reload from disk even if already cached.
    """
    global _cached_registry
    if _cached_registry is None or reload:
        _cached_registry = load_bundled()
    return _cached_registry


def filter_attacks(
    registry: CommunityRegistry,
    *,
    category: Optional[str] = None,
    tag: Optional[str] = None,
    severity: Optional[str] = None,
) -> List[CommunityAttack]:
    """Convenience wrapper around ``registry.filter()``."""
    return registry.filter(category=category, tag=tag, severity=severity)
