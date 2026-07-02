"""
Deterministic, pre-LLM guardrails.

Why this exists: the system prompt already instructs the model to refuse
legal/compliance questions and resist prompt injection, but relying on the
LLM alone for this means a single bad sample can leak a wrong answer or
follow an injected instruction. For the two refusal categories that are
reliably pattern-detectable -- legal/compliance questions and prompt
injection attempts -- we check BEFORE calling the LLM at all. If a pattern
fires, we return a fixed, safe refusal immediately: faster (skips the LLM
call entirely), and provably consistent (a regex either matches or it
doesn't; it can't be argued into ignoring itself).

This is intentionally narrow and conservative. It does NOT try to catch
general off-topic chat (weather, coding help, etc.) -- that's open-ended
enough that false positives would hurt legitimate use, so it's left to the
LLM's judgment per the system prompt, and covered by adversarial probes
instead.
"""
import re
from dataclasses import dataclass
from typing import Optional

_LEGAL_PATTERNS = [
    r"\blegally requir",
    r"\blegal(ly)? (obligat|requirement|complian)",
    r"\bis (it|this) legal\b",
    r"\bam i (legally )?required by law\b",
    r"\bcompliance requirement",
    r"\bregulatory (obligation|requirement)",
    r"\bunder (hipaa|gdpr|eeoc|ada|ofccp|gina)\b.*(requir|complian|legal|obligat)",
    r"\b(sue|lawsuit|litigation) risk\b",
    r"\bdoes (this|it) satisfy.*(legal|regulatory|compliance)",
    r"\bcan (we|i) be sued\b",
    r"\blegal advice\b",
]

_INJECTION_PATTERNS = [
    r"\bignore (all |any )?(your |the )?(previous|prior|above|earlier) instructions\b",
    r"\bignore (all |any )?(your )?(system )?(prompt|instructions)\b",
    r"\bdisregard (the )?(above|previous|prior)\b",
    r"\breveal (your |the )?(system )?prompt\b",
    r"\bshow (me )?(your |the )?(system )?prompt\b",
    r"\bwhat (are|is) your (system )?(prompt|instructions)\b",
    r"\byou are now\b",
    r"\bdeveloper mode\b",
    r"\bact as (if |a |an )?\b.{0,40}\b(no restrictions|unfiltered|dan|unrestricted)\b",
    r"\bno restrictions\b.{0,30}\b(ai|model|assistant)\b",
    r"\b(ai|model|assistant)\b.{0,30}\bno restrictions\b",
    r"\bpretend (you have|to have) no (restrictions|rules|guidelines)\b",
    r"\bnew instructions?:\s",
    r"\boverride (your |the )?(rules|instructions|guidelines)\b",
    r"\bjailbreak\b",
    r"\bprint (your |the )?(instructions|system prompt)\b",
    r"\btell me your (system )?prompt\b",
]

_legal_re = re.compile("|".join(_LEGAL_PATTERNS), re.IGNORECASE)
_injection_re = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


@dataclass
class GuardrailResult:
    triggered: bool
    reason: Optional[str] = None
    reply: Optional[str] = None

    @property
    def recommendations_allowed(self) -> bool:
        """Convenience: True only when guardrail did NOT fire."""
        return not self.triggered


def check(latest_user_text: str) -> GuardrailResult:
    """Check only the LATEST user message, not the full history -- we don't
    want an injection attempt three turns ago to permanently lock the
    conversation into refusal mode once the user has moved on to a
    legitimate follow-up."""
    if not latest_user_text:
        return GuardrailResult(triggered=False)

    if _injection_re.search(latest_user_text):
        return GuardrailResult(
            triggered=True,
            reason="prompt_injection",
            reply=(
                "I can't follow instructions embedded in a message like that. "
                "I'm happy to help you find or compare SHL assessments -- what "
                "role or need are you assessing for?"
            ),
        )

    if _legal_re.search(latest_user_text):
        return GuardrailResult(
            triggered=True,
            reason="legal_compliance",
            reply=(
                "That's a legal or regulatory compliance question, which is outside "
                "what I can advise on -- your legal or compliance team is the right "
                "resource for that. I can tell you what a given SHL assessment "
                "measures, but not whether using it satisfies a specific legal "
                "requirement."
            ),
        )

    return GuardrailResult(triggered=False)
