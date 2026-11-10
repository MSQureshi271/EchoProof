"""R3-S2: render the report template by filling `{{fact.key}}` placeholders from the computed facts.

The template is the human-edited text. The rendered draft is generated and is never edited by hand. Rendering fails on an unknown placeholder, so a typo or a
renamed fact cannot silently leave a gap or a stale number in the draft.
"""
from __future__ import annotations

import re

from echoproof.reporting.facts import FactError

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_.]+)\s*\}\}")


def placeholders(template: str) -> list[str]:
    return PLACEHOLDER.findall(template)


def render(template: str, facts: dict) -> tuple[str, set]:
    """Return the rendered text and the set of fact keys used."""
    unknown = sorted({key for key in placeholders(template) if key not in facts})
    if unknown:
        raise FactError(f"template uses unknown facts: {unknown}")
    used: set = set()

    def fill(match):
        key = match.group(1)
        used.add(key)
        return facts[key]["text"]

    return PLACEHOLDER.sub(fill, template), used


def unused_facts(facts: dict, used: set) -> list[str]:
    return sorted(set(facts) - set(used))
