"""Custom Attack Library — user-defined adversarial test cases.

A JSON-backed persistent store for custom attack cases. Deduplicates on
SHA-256 content hash. Supports category tagging, expected-refusal flags, and
provenance tracking. Thread-safe for single-process use (file is re-read on
every mutation in concurrent scenarios).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

VALID_CATEGORIES = frozenset(
    {"jailbreak", "injection", "edge_case", "boundary", "indirect", "agentic", "custom"}
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CustomAttack:
    """One user-defined adversarial test case.

    Attributes
    ----------
    text:
        The adversarial prompt text.
    category:
        Attack category (jailbreak / injection / edge_case / boundary /
        indirect / agentic / custom).
    language:
        BCP-47 language tag of the prompt (default "en").
    expected_refusal:
        True when a safe model should refuse this prompt.
    provenance:
        Free-text source label (default "user").
    notes:
        Optional annotation for this attack case.
    id:
        SHA-256[:16] of the prompt text; auto-computed on construction.
    created:
        ISO-8601 UTC timestamp; auto-set on construction.
    """

    text: str
    category: str
    language: str = "en"
    expected_refusal: bool = True
    provenance: str = "user"
    notes: str = ""
    id: str = field(default="")
    created: str = field(default="")

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("CustomAttack.text must not be empty")
        if self.category not in VALID_CATEGORIES:
            valid = ", ".join(sorted(VALID_CATEGORIES))
            raise ValueError(
                f"Unknown category {self.category!r}. Valid: {valid}"
            )
        if not self.id:
            self.id = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]
        if not self.created:
            self.created = (
                datetime.datetime.now(datetime.timezone.utc).isoformat()
            )


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------


class AttackLibrary:
    """JSON-backed persistent store for custom adversarial attacks.

    Deduplicates on content hash (``CustomAttack.id``). All mutations
    immediately persist to disk.

    Usage::

        from toki.attack_library import AttackLibrary, CustomAttack

        lib = AttackLibrary("attacks.json")
        added = lib.add(CustomAttack(text="Ignore previous instructions", category="jailbreak"))
        attacks = lib.list_attacks()
    """

    def __init__(self, path: str | Path = "attacks.json") -> None:
        self._path = Path(path)
        self._attacks: Dict[str, CustomAttack] = {}
        if self._path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add(self, attack: CustomAttack) -> bool:
        """Add an attack to the library.

        Returns False (without error) when the content is a duplicate.
        Returns True when the attack was successfully added.
        """
        if attack.id in self._attacks:
            logger.debug("AttackLibrary.add: duplicate id=%s, skipping", attack.id)
            return False
        self._attacks[attack.id] = attack
        self._save()
        return True

    def remove(self, attack_id: str) -> bool:
        """Remove an attack by id. Returns True on success, False if not found."""
        if attack_id not in self._attacks:
            return False
        del self._attacks[attack_id]
        self._save()
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_attacks(self, category: Optional[str] = None) -> List[CustomAttack]:
        """Return all attacks, optionally filtered by category.

        Results are sorted by creation timestamp (oldest first).
        """
        attacks = list(self._attacks.values())
        if category is not None:
            attacks = [a for a in attacks if a.category == category]
        return sorted(attacks, key=lambda a: a.created)

    def get(self, attack_id: str) -> Optional[CustomAttack]:
        """Return a single attack by id, or None if not found."""
        return self._attacks.get(attack_id)

    def stats(self) -> Dict[str, object]:
        """Return summary statistics for the library.

        Returns
        -------
        dict with keys:
            - total: int
            - by_category: dict[category, count]
        """
        by_cat: Dict[str, int] = {}
        for attack in self._attacks.values():
            by_cat[attack.category] = by_cat.get(attack.category, 0) + 1
        return {"total": len(self._attacks), "by_category": by_cat}

    def __len__(self) -> int:
        return len(self._attacks)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(a) for a in self._attacks.values()]
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "AttackLibrary: could not load %s: %s", self._path, exc
            )
            return
        for item in data:
            try:
                attack = CustomAttack(**item)
                self._attacks[attack.id] = attack
            except (TypeError, ValueError) as exc:
                logger.warning("AttackLibrary: skipping malformed entry: %s", exc)
