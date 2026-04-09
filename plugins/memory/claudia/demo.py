"""Claudia Hybrid Memory — runnable demo.

A scripted walkthrough of the plugin's end-to-end flow. Runs
offline (no Ollama required) using scripted fakes for the
extractor and commitment detector, so you can see every
feature of the provider without any setup.

Usage:
    python -m plugins.memory.claudia.demo

What it demonstrates (in order):
    1. Spin up a provider in a temporary directory
    2. Store a manual fact via memory.remember
    3. Simulate a sync_turn — scripted extractor picks out
       entities (Sarah Chen, Acme Labs), scripted detector
       picks out a commitment ("send the proposal by Friday")
    4. Query: memory.recall, memory.about, memory.commitments
    5. User correction: memory.correct_memory replaces a fact
       and marks the old version as contradicts
    6. memory.trace walks the correction chain (audit trail)
    7. Maintenance: provider.consolidate() + provider.verify()
    8. Print internal metrics counters

Everything is printed as structured output so you can see
exactly what the plugin is doing.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List

from plugins.memory.claudia import ClaudiaMemoryProvider
from plugins.memory.claudia.commitment_detector import (
    CommitmentDetector,
    DetectedCommitment,
)
from plugins.memory.claudia.embeddings import OllamaEmbedder
from plugins.memory.claudia.extractor import (
    ExtractedEntity,
    LLMExtractor,
)


# ─── Console formatting ────────────────────────────────────────────────


RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM = "\033[2m"


def h1(text: str) -> None:
    print(f"\n{BOLD}{CYAN}━━━ {text} ━━━{RESET}")


def h2(text: str) -> None:
    print(f"\n{BOLD}{YELLOW}→ {text}{RESET}")


def say(text: str) -> None:
    print(f"  {DIM}User says:{RESET} {text}")


def show(label: str, value: str) -> None:
    print(f"  {GREEN}{label}{RESET} {value}")


def block(text: str) -> None:
    for line in text.splitlines():
        print(f"  {DIM}│{RESET} {line}")


def pretty(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


# ─── Fake cognitive components (scripted, offline) ─────────────────────


class _DemoEmbedder(OllamaEmbedder):
    """Stable fake — every embedding is a fixed 3-dim vector."""

    def __init__(self):
        super().__init__()

    def _call_embed(self, text):  # type: ignore[override]
        # Return a unit vector that varies slightly based on first char
        # to make recall results feel distinct
        if not text:
            return [0.0, 0.0, 0.0]
        seed = ord(text[0]) / 128.0
        return [seed, 1.0 - seed, 0.5]


class _DemoExtractor(LLMExtractor):
    """Scripted extractor. Each call pops the next result from a queue."""

    def __init__(self, script: List[List[ExtractedEntity]]):
        self._script = list(script)
        self.call_count = 0

    def extract(self, text, *, source_ref=""):  # type: ignore[override]
        self.call_count += 1
        if not self._script:
            return []
        return self._script.pop(0)


class _DemoDetector(CommitmentDetector):
    """Scripted commitment detector."""

    def __init__(self, script: List[List[DetectedCommitment]]):
        self._script = list(script)
        self.call_count = 0

    def detect(self, text, *, source_ref=""):  # type: ignore[override]
        self.call_count += 1
        if not self._script:
            return []
        return self._script.pop(0)


class _DemoProvider(ClaudiaMemoryProvider):
    """Provider that uses the demo fakes. Everything else is real —
    SQLite writes, hybrid search, consolidation, verification, all
    the trust-factor ranking logic, etc."""

    def __init__(self, extractor, detector):
        super().__init__()
        self._demo_ext = extractor
        self._demo_det = detector

    def _make_embedder(self) -> OllamaEmbedder:  # type: ignore[override]
        return _DemoEmbedder()

    def _make_extractor(self) -> LLMExtractor:  # type: ignore[override]
        return self._demo_ext

    def _make_commitment_detector(self) -> CommitmentDetector:  # type: ignore[override]
        return self._demo_det


# ─── The scripted scenario ─────────────────────────────────────────────


def run_demo() -> None:
    print(f"{BOLD}Claudia Hybrid Memory — runnable demo{RESET}")
    print(f"{DIM}Everything runs offline with scripted fakes.{RESET}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        h1("1. Spinning up the provider")

        extractor = _DemoExtractor(script=[
            # Turn 1: extract Sarah and Acme Labs
            [
                ExtractedEntity(
                    name="Sarah Chen", kind="person",
                    canonical_name="sarah chen", confidence=0.92,
                    attributes={"role": "CTO"},
                ),
                ExtractedEntity(
                    name="Acme Labs", kind="organization",
                    canonical_name="acme labs", confidence=0.9,
                ),
            ],
            # Turn 2: no new entities (commitment only)
            [],
        ])

        detector = _DemoDetector(script=[
            # Turn 1: no commitment
            [],
            # Turn 2: "I'll send the proposal by Friday"
            [
                DetectedCommitment(
                    content="send the proposal to Sarah",
                    deadline_raw="by Friday",
                    deadline_iso="2026-04-10T00:00:00+00:00",
                    confidence=0.9,
                    commitment_type="explicit",
                ),
            ],
        ])

        p = _DemoProvider(extractor=extractor, detector=detector)
        p.initialize(
            session_id="demo-session-001",
            claudia_home=str(tmp_path),
            platform="cli",
        )
        show("Database:", f"{tmp_path}/memory/claudia/claudia.db")
        show("Session:", "demo-session-001")
        show("Tool surface:", f"{len(p.get_tool_schemas())} tools")

        # ── Step 2: store a fact manually ─────────────────────────
        h1("2. Manual memory.remember")
        say("Remember that my favorite color is blue.")
        result = json.loads(p.handle_tool_call(
            "memory.remember",
            {"content": "The user's favorite color is blue",
             "importance": 0.7},
        ))
        block(pretty(result))

        # ── Step 3: sync_turn with extraction + detection ─────────
        h1("3. sync_turn triggers extraction + detection")
        say("I met Sarah Chen at Acme Labs yesterday. "
            "She's their CTO.")
        p.sync_turn(
            "I met Sarah Chen at Acme Labs yesterday. She's their CTO.",
            "Got it — nice to know about Sarah and Acme.",
            session_id="demo-session-001",
        )
        say("I'll send her the proposal by Friday.")
        p.sync_turn(
            "I'll send her the proposal by Friday.",
            "Good luck with the proposal.",
            session_id="demo-session-001",
        )
        assert p.flush(timeout=5.0)
        show("Extractor calls:", str(extractor.call_count))
        show("Detector calls:", str(detector.call_count))

        # ── Step 4: read back what was stored ─────────────────────
        h1("4. memory.about — look up Sarah")
        result = json.loads(p.handle_tool_call(
            "memory.about", {"name": "Sarah Chen"},
        ))
        block(pretty(result))

        h1("5. memory.commitments — list open commitments")
        result = json.loads(p.handle_tool_call(
            "memory.commitments", {},
        ))
        block(pretty(result))

        h1("6. memory.recall — semantic search")
        result = json.loads(p.handle_tool_call(
            "memory.recall", {"query": "favorite color"},
        ))
        block(pretty(result))

        # ── Step 7: user correction ────────────────────────────────
        h1("7. memory.correct_memory — user correction")
        say("Actually, Sarah is VP Engineering, not CTO.")
        # Grab the id of the sync_turn-stored memory about Sarah
        remember_result = json.loads(p.handle_tool_call(
            "memory.remember",
            {"content": "Sarah Chen is the CTO at Acme Labs"},
        ))
        old_id = remember_result["memory_id"]
        correction = json.loads(p.handle_tool_call(
            "memory.correct_memory",
            {"id": old_id,
             "new_content": "Sarah Chen is VP Engineering at Acme Labs"},
        ))
        show("Correction:", "stored with origin='corrected', "
             "confidence=1.0, verification='verified'")
        block(pretty(correction))

        # ── Step 8: trace the correction chain ─────────────────────
        h1("8. memory.trace — provenance audit")
        trace = json.loads(p.handle_tool_call(
            "memory.trace",
            {"id": correction["memory"]["id"]},
        ))
        print(f"  {DIM}Full chain:{RESET}")
        for entry in trace["chain"]:
            ver = entry["verification"]
            color_code = {
                "verified": GREEN,
                "contradicts": YELLOW,
                "pending": DIM,
            }.get(ver, RESET)
            arrow = "←" if entry.get("corrected_from") else " "
            print(f"  {arrow} [{color_code}{ver:11s}{RESET}] "
                  f"id={entry['id']}  "
                  f"{entry['content'][:60]}")

        # ── Step 9: maintenance ────────────────────────────────────
        h1("9. provider.consolidate — fuzzy entity dedup pass")
        cons = p.consolidate(timeout=5.0)
        show("Entities merged:", str(cons.entities_merged))
        show("Commitments linked:", str(cons.commitments_linked))
        show("Duration:", f"{cons.duration_seconds*1000:.1f}ms")

        h1("10. provider.verify — confidence decay + stale flag")
        ver = p.verify(timeout=5.0)
        show("Memories decayed:", str(ver.decayed_count))
        show("Stale flagged:", str(ver.flagged_stale_count))
        show("Duration:", f"{ver.duration_seconds*1000:.1f}ms")

        # ── Step 11: observability ─────────────────────────────────
        h1("11. memory.metrics — internal counters")
        metrics = json.loads(p.handle_tool_call("memory.metrics", {}))
        snapshot = metrics["metrics"]
        # Group counters by prefix for readable output
        groups = {
            "sync_turn": [],
            "memories": [],
            "entities": [],
            "commitments": [],
            "cognitive": [],
            "consolidate": [],
            "verify": [],
            "purge": [],
            "tool": [],
        }
        for key, val in sorted(snapshot.items()):
            prefix = key.split(".", 1)[0]
            if prefix in groups:
                groups[prefix].append((key, val))
            else:
                groups.setdefault("other", []).append((key, val))

        for group_name, items in groups.items():
            if not items:
                continue
            print(f"  {DIM}{group_name}{RESET}")
            for key, val in items:
                print(f"    {key:40s} {val}")

        # ── Step 12: status summary ────────────────────────────────
        h1("12. memory.status — final snapshot")
        status = json.loads(p.handle_tool_call("memory.status", {}))
        block(pretty(status))

        p.shutdown()

    print(f"\n{BOLD}{GREEN}✓ Demo complete.{RESET}")
    print(f"{DIM}Everything above ran locally, in a temp directory "
          f"that's now cleaned up.{RESET}")
    print(f"{DIM}To run against real Ollama, swap the demo "
          f"extractor/detector for the real OllamaLLMExtractor "
          f"and HybridCommitmentDetector.{RESET}")


if __name__ == "__main__":
    run_demo()
