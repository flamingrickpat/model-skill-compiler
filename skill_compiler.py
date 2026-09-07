#!/usr/bin/env python3
"""
model-skill-compiler

Progressively compress a Markdown coding-agent skill *for the exact model
currently loaded in llama.cpp*.

Core idea
---------
The skill is compiled left-to-right. For each prose block:

    generic coding-agent framing
    + already-COMPILED prefix
    -> teacher-force ORIGINAL current block one token at a time
    -> measure what this model finds surprising
    -> ask the SAME model to rewrite only the predictable redundancy
    -> append the accepted rewrite
    -> score the next block against that new compressed prefix

This matters: later probabilities are conditioned on the text the production
agent will actually see, not on an original prefix that was already deleted.

Protected source
----------------
Version 1 NEVER rewrites:
- YAML frontmatter
- headings
- fenced code
- indented Markdown code blocks
- raw HTML blocks
- complete raw HTML documents/templates (<html> ... </html>)
- Markdown tables
- thematic/structural Markdown outside prose paragraphs

Inline literals inside prose (backticks, paths, flags, numbers, etc.) are
protected during rewrite.

The output Markdown is written after every accepted block. The unprocessed
remainder stays original, so even an interrupted run leaves a complete usable
SKILL.compiled.md.

Requires llama.cpp native server endpoints:
  /props
  /apply-template
  /tokenize
  /completion

Scoring deliberately uses n_predict=1. Some llama.cpp builds return empty
top_logprobs for multi-token generations with n_probs enabled.

Install:
    pip install -r requirements.txt

Example:
    python skill_compiler.py agents/skills/foo/SKILL.md \
        --url http://127.0.0.1:8080

Useful:
    --output foo.compiled.md
    --top-n 128
    --keep-p90 5
    --keep-max 8
    --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import requests
from rich.console import Console
from rich.table import Table

try:
    import frontmatter
    from blingfire import text_to_sentences_and_offsets
    from json_repair import loads as json_repair_loads
    from markdown_it import MarkdownIt
    from mdit_py_plugins.front_matter import front_matter_plugin
    import yake
except ImportError as e:
    missing = getattr(e, "name", str(e))
    raise SystemExit(
        f"Missing dependency: {missing}\n"
        "Run: pip install -r requirements.txt"
    )

console = Console()
LN2 = math.log(2)

GENERIC_AGENT_SYSTEM = """You are a capable autonomous coding agent working in a software repository.

You may be assigned specialized roles, workflows, or skills. Depending on the task, you may inspect and modify source code, configuration, tests, documentation, scripts, build files, and other project artifacts; use development tools; run commands and tests; diagnose failures; and continue iterating until the assigned work is complete.

Follow the instructions and constraints supplied for your current role. Respect project-specific conventions, scope boundaries, required workflows, tool semantics, validation requirements, and completion criteria.

Use your existing software-engineering knowledge and normal coding-agent competence unless the supplied instructions specify otherwise."""

SKILL_USER_PREFIX = """You are assigned the specialized role `{skill_name}`.

The following is the skill definition for this role. Treat it as the authoritative instructions for how this role operates and follow it when performing the assigned work.

<skill>
"""

REWRITE_SYSTEM = """You compile one prose block from an operational coding-agent skill into a shorter representation for the SAME language model that will later read it.

This is model-specific instruction compression, not ordinary summarization.

Rules:
- Preserve every project-specific behavior, invariant, prohibition, condition, ordering relation, scope boundary, failure rule, exact quantity, status, identifier, path, command, and tool semantic.
- Preserve negation and modality. "must", "must not", "only", "before", "after", "unless", "exactly", etc. can carry more information than nouns.
- EXACT spans supplied by the caller must appear byte-for-byte in the result.
- HIGH-SURPRISE spans are strong evidence of model-novel information. Preserve their operational meaning; prefer their original wording when already terse.
- Low-surprise generic coding advice may be shortened to a tiny semantic cue or removed when the surrounding role already implies it.
- Do not turn relational rules into noun-only keyword soup. "commit evidence artifact" is not equivalent to "never commit before the evidence artifact exists".
- Remove rationale/examples only when they are not necessary to disambiguate or generalize a rule.
- Do not add requirements, exceptions, defaults, commands, tools, or facts.
- Preserve the Markdown role of the block (paragraph/list item/blockquote). Do not emit code fences.
- If the entire block is safely implicit in ordinary coding-agent competence and carries no contextual cue needed by later instructions, it may become empty.
- Prefer terse operational language.

Return JSON only:
{"text":"<rewritten markdown block, or empty string>"}
"""

# Strong semantic operators. We do not force these exact words to survive, but
# their presence makes a sentence less safe to drop.
OPERATOR_RE = re.compile(
    r"\b(?:"
    r"must(?:\s+not)?|never|always|only|exactly|forbid(?:den)?|"
    r"before|after|until|unless|if|when|otherwise|except|without|"
    r"first|last|required|shall|may\s+not|cannot|can't|do\s+not|no"
    r")\b",
    re.IGNORECASE,
)

# Things whose exact spelling commonly *is* the contract.
INLINE_CODE_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
FLAG_RE = re.compile(r"(?<!\w)--?[A-Za-z][A-Za-z0-9_-]*(?:=[^\s,;]+)?")
URL_RE = re.compile(r"https?://[^\s)>]+")
PATH_RE = re.compile(
    r"(?<!\w)(?:"
    r"[A-Za-z]:[\\/][^\s`\"']+|"
    r"(?:\.{0,2}[\\/])(?:[\w.\-]+[\\/])+[\w.\-]+|"
    r"(?:[\w.\-]+[\\/]){2,}[\w.\-]+|"
    r"[\w.\-]+\.(?:md|json|ya?ml|toml|ini|cfg|py|js|ts|tsx|jsx|cs|cpp|c|h|"
    r"html|css|xml|sql|sh|ps1|bat|cmd)"
    r")"
)
NUMBER_RE = re.compile(
    r"(?<![\w.])"
    r"(?:\d+(?:\.\d+)?(?:%|ms|s|min|h|k|m|gb|mb|kb|tokens?)?"
    r"|0x[0-9A-Fa-f]+)"
    r"(?![\w.])",
    re.IGNORECASE,
)
COMPARATOR_RE = re.compile(r"(?:<=|>=|==|!=|<|>)")


@dataclass
class Segment:
    index: int
    kind: str                 # prose | protected
    reason: str
    start_line: int           # 0-based inclusive
    end_line: int             # 0-based exclusive
    text: str


@dataclass
class TokenScore:
    token_id: int
    text: str
    start_byte: int
    end_byte: int
    rank: int | None
    logprob: float | None
    surprisal_bits: float
    regret_bits: float
    censored: bool


@dataclass
class SentenceSignal:
    text: str
    start: int
    end: int
    label: str
    mean_regret: float
    p90_regret: float
    max_regret: float
    top1_fraction: float
    high_fraction: float
    has_operator: bool
    exact_spans: list[str]
    high_surprise_spans: list[str]


class Llama:
    def __init__(self, url: str, slot: int = 0, timeout: int = 3600):
        url = url.strip()
        if "://" not in url:
            url = "http://" + url
        url = url.rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        self.url = url
        self.slot = slot
        self.timeout = timeout
        self.http = requests.Session()

    def get(self, path: str) -> dict[str, Any]:
        r = self.http.get(self.url + path, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        r = self.http.post(self.url + path, json=data, timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(
                f"{path}: HTTP {r.status_code}\n{r.text[:4000]}"
            )
        return r.json()

    def apply_template(self, messages: list[dict[str, str]]) -> str:
        r = self.post("/apply-template", {
            "messages": messages,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_format": "none",
        })
        return r["prompt"]

    def tokenize(
        self,
        text: str,
        *,
        add_special: bool = True,
        pieces: bool = False,
    ) -> list[Any]:
        r = self.post("/tokenize", {
            "content": text,
            "add_special": add_special,
            "parse_special": True,
            "with_pieces": pieces,
        })
        return r["tokens"]

    def completion_one(
        self,
        prompt_ids: list[int],
        *,
        top_n: int,
        cache_prompt: bool = True,
    ) -> dict[str, Any]:
        # n_predict=1 is intentional. See module docstring.
        return self.post("/completion", {
            "prompt": prompt_ids,
            "n_predict": 1,
            "n_probs": top_n,
            "temperature": 0.0,
            "post_sampling_probs": False,
            "return_tokens": True,
            "stream": False,
            "cache_prompt": cache_prompt,
            "id_slot": self.slot,
            "ignore_eos": True,
            "seed": 1,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "dry_multiplier": 0.0,
        })

    def generate_chat(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        slot: int | None = None,
    ) -> str:
        prompt = self.apply_template([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0,
            "stream": False,
            "cache_prompt": False,
            "seed": 1,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "dry_multiplier": 0.0,
        }
        payload["id_slot"] = self.slot if slot is None else slot
        r = self.post("/completion", payload)
        return r.get("content", "")


def piece_bytes(token: Any) -> bytes:
    p = token["piece"]
    return p.encode("utf-8") if isinstance(p, str) else bytes(p)


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    pos = (len(ys) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ys[lo]
    return ys[lo] * (hi - pos) + ys[hi] * (pos - lo)


def merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ranges = sorted((a, b) for a, b in ranges if b > a)
    out: list[list[int]] = []
    for a, b in ranges:
        if not out or a > out[-1][1]:
            out.append([a, b])
        else:
            out[-1][1] = max(out[-1][1], b)
    return [(a, b) for a, b in out]


def is_in_ranges(line: int, ranges: list[tuple[int, int]]) -> bool:
    return any(a <= line < b for a, b in ranges)


def standalone_html_ranges(lines: list[str]) -> list[tuple[int, int]]:
    """
    CommonMark intentionally splits a large HTML document at some blank lines.
    For skill templates, preserve a complete <html>...</html> source region.
    """
    out = []
    i = 0
    while i < len(lines):
        s = lines[i].lstrip().lower()
        if s.startswith("<!doctype html") or re.match(r"<html(?:\s|>)", s):
            start = i
            j = i
            while j < len(lines):
                if "</html>" in lines[j].lower():
                    out.append((start, j + 1))
                    i = j + 1
                    break
                j += 1
            else:
                # Unclosed raw HTML: preserve to EOF rather than touching it.
                out.append((start, len(lines)))
                return out
        else:
            i += 1
    return out


def markdown_segments(text: str) -> list[Segment]:
    """
    Use markdown-it-py for actual Markdown structure. Custom code is limited to
    mapping exact source line ranges and protecting complete raw HTML documents.
    """
    lines = text.splitlines(keepends=True)

    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    md.use(front_matter_plugin)
    tokens = md.parse(text)

    protected: list[tuple[int, int]] = []
    prose: list[tuple[int, int]] = []

    for tok in tokens:
        if not tok.map:
            continue
        a, b = tok.map

        if tok.type in {
            "front_matter",
            "fence",
            "code_block",
            "html_block",
            "table_open",
            "heading_open",      # structural/navigation cue: keep v1 exact
            "hr",
        }:
            protected.append((a, b))

        if tok.type == "paragraph_open":
            prose.append((a, b))

    protected.extend(standalone_html_ranges(lines))
    protected = merge_ranges(protected)

    # Remove prose ranges that overlap anything protected.
    prose = [
        (a, b) for a, b in prose
        if not any(max(a, p0) < min(b, p1) for p0, p1 in protected)
    ]

    # Paragraph maps can theoretically overlap under plugins. Keep only unique,
    # non-overlapping leaf ranges.
    prose = merge_ranges(prose)

    # Every source line belongs to either a selected prose paragraph or an exact
    # passthrough region/gap. This guarantees source order and lossless handling
    # of syntax we did not explicitly classify.
    marks: list[str | None] = [None] * len(lines)

    for a, b in protected:
        for i in range(a, min(b, len(lines))):
            marks[i] = "protected"

    for a, b in prose:
        for i in range(a, min(b, len(lines))):
            if marks[i] is None:
                marks[i] = "prose"

    # Gaps, blank lines, list/container syntax, and anything unknown are exact
    # passthrough. This is deliberately conservative.
    for i in range(len(marks)):
        if marks[i] is None:
            marks[i] = "protected"

    segments: list[Segment] = []
    i = 0
    while i < len(lines):
        kind = marks[i]
        j = i + 1
        while j < len(lines) and marks[j] == kind:
            # Do not merge separate prose paragraphs across blank structural
            # passthrough: paragraph maps already stop before those lines.
            if kind == "prose":
                # A continuous prose map is okay, but stop when line j belongs
                # to a different original paragraph range.
                owner_i = next(((a, b) for a, b in prose if a <= i < b), None)
                if owner_i and not (owner_i[0] <= j < owner_i[1]):
                    break
            j += 1

        reason = "markdown prose paragraph" if kind == "prose" else "protected/structural markdown"
        segments.append(Segment(
            index=len(segments),
            kind=kind or "protected",
            reason=reason,
            start_line=i,
            end_line=j,
            text="".join(lines[i:j]),
        ))
        i = j

    return segments


def find_exact_spans(text: str) -> list[str]:
    spans: list[str] = []
    for rx in (INLINE_CODE_RE, FLAG_RE, URL_RE, PATH_RE, NUMBER_RE, COMPARATOR_RE):
        for m in rx.finditer(text):
            s = m.group(0)
            if s and s not in spans:
                spans.append(s)
    return spans


def build_skill_prompt_prefix(
    llm: Llama,
    *,
    system_prompt: str,
    skill_name: str,
) -> str:
    """
    Render a chat template whose user message is intentionally left open at the
    skill body. This scores the skill as *input instructions*, not as assistant
    prose the model was asked to author.

    We place a sentinel inside the user content, apply the model's exact chat
    template, then cut the rendered prompt immediately before the sentinel.
    """
    sentinel = "__MODEL_SKILL_COMPILER_BODY_9f6c1c4a__"
    user_prefix = SKILL_USER_PREFIX.format(skill_name=skill_name)

    for n_newlines in range(0, 5):
        user = user_prefix + ("\n" * n_newlines) + sentinel
        rendered = llm.apply_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ])
        if sentinel in rendered:
            before, after = rendered.split(sentinel, 1)
            if sentinel not in after:
                return before

    raise RuntimeError(
        "The chat template transformed/removed the skill sentinel. "
        "Cannot safely build an open user-message prefix."
    )


def tokenize_for_continuation(
    llm: Llama,
    context: str,
    continuation: str,
) -> tuple[list[int], list[int], list[bytes], int]:
    """
    Tokenize exact context+continuation once.

    If a BPE token straddles the text boundary, include that token in the prefix
    and skip scoring only that boundary token. Subsequent token probabilities
    remain exact for the true final byte stream.
    """
    # /apply-template already rendered the model's special tokens. Re-adding
    # tokenizer BOS/EOS here would corrupt source-byte offset accounting.
    full = llm.tokenize(context + continuation, add_special=False, pieces=True)
    ids = [int(x["id"]) for x in full]
    pieces = [piece_bytes(x) for x in full]

    boundary = len(context.encode("utf-8"))
    offset = 0
    start_idx = None
    skipped_boundary_bytes = 0

    for i, b in enumerate(pieces):
        token_start = offset
        token_end = offset + len(b)
        if token_start >= boundary:
            start_idx = i
            break
        if token_start < boundary < token_end:
            # This token contains bytes from both sides. Treat it as context.
            skipped_boundary_bytes = token_end - boundary
        offset = token_end

    if start_idx is None:
        start_idx = len(ids)

    prefix_ids = ids[:start_idx]
    target_ids = ids[start_idx:]
    target_pieces = pieces[start_idx:]
    return prefix_ids, target_ids, target_pieces, skipped_boundary_bytes


def score_distribution(target_id: int, item: dict[str, Any]) -> dict[str, Any]:
    top = item.get("top_logprobs") or []
    generated_id = int(item["id"])
    generated_lp = item.get("logprob")

    if not top:
        if generated_id == target_id and generated_lp is not None:
            lp = float(generated_lp)
            return {
                "rank": 1,
                "logprob": lp,
                "surprisal_bits": max(0.0, -lp / LN2),
                "regret_bits": 0.0,
                "censored": False,
            }
        raise RuntimeError(
            "llama.cpp returned empty top_logprobs on a mismatch. "
            "Use n_predict=1 (this script does) and verify your build's n_probs support."
        )

    top = sorted(top, key=lambda x: x["logprob"], reverse=True)
    top1 = float(top[0]["logprob"])

    for rank, x in enumerate(top, 1):
        if int(x["id"]) == target_id:
            lp = float(x["logprob"])
            return {
                "rank": rank,
                "logprob": lp,
                "surprisal_bits": max(0.0, -lp / LN2),
                "regret_bits": max(0.0, (top1 - lp) / LN2),
                "censored": False,
            }

    cutoff = float(top[-1]["logprob"])
    return {
        "rank": None,
        "logprob": None,
        "surprisal_bits": max(0.0, -cutoff / LN2),
        "regret_bits": max(0.0, (top1 - cutoff) / LN2),
        "censored": True,
    }


def score_block(
    llm: Llama,
    *,
    context: str,
    block: str,
    top_n: int,
    cache_prompt: bool,
) -> tuple[list[TokenScore], int]:
    prefix_ids, ids, pieces, skipped_boundary_bytes = tokenize_for_continuation(
        llm, context, block
    )

    # Map target token bytes back to byte offsets inside this block.
    # skipped_boundary_bytes means the first few block bytes were swallowed by
    # a BPE token that straddled the context boundary and are intentionally
    # unscored.
    block_offset = skipped_boundary_bytes
    scores: list[TokenScore] = []

    for i, (target, piece) in enumerate(zip(ids, pieces)):
        r = llm.completion_one(
            prefix_ids + ids[:i],
            top_n=top_n,
            cache_prompt=cache_prompt,
        )
        probs = r.get("completion_probabilities") or []
        if not probs:
            raise RuntimeError(
                "llama.cpp returned no completion_probabilities.\n"
                f"generation_settings={json.dumps(r.get('generation_settings', {}), indent=2)}"
            )

        item = probs[0]
        d = score_distribution(target, item)

        start = block_offset
        end = start + len(piece)
        scores.append(TokenScore(
            token_id=target,
            text=piece.decode("utf-8", "replace"),
            start_byte=start,
            end_byte=end,
            rank=d["rank"],
            logprob=d["logprob"],
            surprisal_bits=d["surprisal_bits"],
            regret_bits=d["regret_bits"],
            censored=d["censored"],
        ))
        block_offset = end

    return scores, skipped_boundary_bytes


def char_to_byte_offsets(text: str) -> list[int]:
    out = [0]
    n = 0
    for ch in text:
        n += len(ch.encode("utf-8"))
        out.append(n)
    return out


def sentence_ranges(text: str) -> list[tuple[str, int, int]]:
    """
    BlingFire returns:
        (newline-delimited sentence string, [(start, end), ...])

    We use BlingFire for segmentation, then locate each returned sentence
    sequentially in the original Python string. That avoids depending on whether
    a particular BlingFire build exposes byte or character offsets for Unicode,
    while still using its sentence boundary decisions.
    """
    try:
        sentence_blob, _offsets = text_to_sentences_and_offsets(text)
        sentences = [s for s in sentence_blob.split("\\n") if s]
    except Exception:
        sentences = []

    out: list[tuple[str, int, int]] = []
    cursor = 0

    for sent in sentences:
        # BlingFire removes separator newlines from its output but otherwise
        # normally preserves sentence text. Search forward so repeated sentences
        # resolve to the correct occurrence.
        a = text.find(sent, cursor)
        if a < 0:
            # Whitespace-normalizing fallback for unusual Markdown wrapping:
            # don't invent fragile offset math; keep the whole block instead.
            return [(text, 0, len(text))] if text else []
        b = a + len(sent)
        out.append((text[a:b], a, b))
        cursor = b

    if not out and text:
        out = [(text, 0, len(text))]
    return out


def expand_byte_span_to_words(text: str, start_b: int, end_b: int) -> str:
    raw = text.encode("utf-8")
    start_b = max(0, min(start_b, len(raw)))
    end_b = max(start_b, min(end_b, len(raw)))

    # Move outward over ASCII-ish word/token punctuation. Decode only after
    # locating UTF-8-safe boundaries.
    while start_b > 0 and raw[start_b - 1:start_b] not in b" \t\r\n,;()[]{}":
        start_b -= 1
    while end_b < len(raw) and raw[end_b:end_b + 1] not in b" \t\r\n,;()[]{}":
        end_b += 1

    # UTF-8 boundary repair.
    while start_b < len(raw):
        try:
            raw[start_b:end_b].decode("utf-8")
            break
        except UnicodeDecodeError:
            start_b += 1
    while end_b > start_b:
        try:
            return raw[start_b:end_b].decode("utf-8").strip()
        except UnicodeDecodeError:
            end_b -= 1
    return ""


def high_surprise_spans(
    text: str,
    scores: list[TokenScore],
    *,
    threshold: float,
    bridge_bytes: int = 3,
) -> list[str]:
    hits = [
        (s.start_byte, s.end_byte)
        for s in scores
        if s.regret_bits >= threshold or s.censored
    ]
    if not hits:
        return []

    merged: list[list[int]] = []
    for a, b in hits:
        if not merged or a - merged[-1][1] > bridge_bytes:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)

    out = []
    for a, b in merged:
        s = expand_byte_span_to_words(text, a, b)
        if s and s not in out:
            out.append(s)
    return out


def classify_sentences(
    text: str,
    scores: list[TokenScore],
    *,
    keep_p90: float,
    keep_max: float,
    keep_fraction_threshold: float,
    high_token_threshold: float,
    drop_p90: float,
) -> list[SentenceSignal]:
    c2b = char_to_byte_offsets(text)
    exact_all = find_exact_spans(text)
    surprising_all = high_surprise_spans(
        text, scores, threshold=high_token_threshold
    )

    signals: list[SentenceSignal] = []
    for sent, ca, cb in sentence_ranges(text):
        ba, bb = c2b[ca], c2b[cb]
        ts = [
            s for s in scores
            if s.end_byte > ba and s.start_byte < bb
        ]
        if not ts:
            # Boundary-token skip or odd markup. Be conservative.
            signals.append(SentenceSignal(
                text=sent,
                start=ca,
                end=cb,
                label="KEEP",
                mean_regret=0.0,
                p90_regret=0.0,
                max_regret=0.0,
                top1_fraction=0.0,
                high_fraction=0.0,
                has_operator=bool(OPERATOR_RE.search(sent)),
                exact_spans=[x for x in exact_all if x in sent],
                high_surprise_spans=[],
            ))
            continue

        regrets = [x.regret_bits for x in ts]
        p90 = percentile(regrets, 0.90)
        mx = max(regrets)
        high_fraction = sum(x >= high_token_threshold for x in regrets) / len(regrets)
        top1_fraction = sum(x.rank == 1 for x in ts) / len(ts)
        exact = [x for x in exact_all if x in sent]
        surprising = [x for x in surprising_all if x and x in sent]
        has_op = bool(OPERATOR_RE.search(sent))

        keep = (
            bool(exact)
            or mx >= keep_max
            or p90 >= keep_p90
            or high_fraction >= keep_fraction_threshold
        )

        # Semantic operators alone should stop outright deletion, but ordinary
        # sentences containing "before/if" can still be condensed.
        if keep:
            label = "KEEP"
        elif p90 < drop_p90 and not has_op:
            label = "DROP_CANDIDATE"
        else:
            label = "CONDENSE"

        signals.append(SentenceSignal(
            text=sent,
            start=ca,
            end=cb,
            label=label,
            mean_regret=statistics.fmean(regrets),
            p90_regret=p90,
            max_regret=mx,
            top1_fraction=top1_fraction,
            high_fraction=high_fraction,
            has_operator=has_op,
            exact_spans=exact,
            high_surprise_spans=surprising,
        ))

    return signals


def yake_keywords(text: str, language: str, top: int = 10) -> list[str]:
    try:
        extractor = yake.KeywordExtractor(
            lan=language,
            n=3,
            dedupLim=0.82,
            top=top,
            features=None,
        )
        return [kw for kw, _score in extractor.extract_keywords(text)]
    except Exception:
        return []


def markdown_shape_hint(block: str) -> str:
    first = block.splitlines()[0] if block.splitlines() else ""
    if re.match(r"^\s*[-+*]\s+", first):
        return "bullet-list item; preserve its bullet marker"
    if re.match(r"^\s*\d+[.)]\s+", first):
        return "numbered-list item; preserve its numbering marker"
    if re.match(r"^\s*>\s?", first):
        return "blockquote paragraph; preserve blockquote marker"
    return "plain Markdown paragraph"


def collect_exact_spans(signals: list[SentenceSignal]) -> list[str]:
    out = []
    for s in signals:
        for x in s.exact_spans:
            if x not in out:
                out.append(x)
    return out


def collect_high_spans(signals: list[SentenceSignal]) -> list[str]:
    out = []
    for s in signals:
        for x in s.high_surprise_spans:
            if x not in out:
                out.append(x)
    return out


def rewrite_prompt(
    *,
    block: str,
    signals: list[SentenceSignal],
    prior_context: str,
    keywords: list[str],
) -> str:
    compact_signals = []
    for s in signals:
        compact_signals.append({
            "label": s.label,
            "text": s.text,
            "p90_regret_bits": round(s.p90_regret, 2),
            "max_regret_bits": round(s.max_regret, 2),
            "top1_fraction": round(s.top1_fraction, 2),
            "operator": s.has_operator,
        })

    exact = collect_exact_spans(signals)
    high = collect_high_spans(signals)

    return f"""Markdown shape: {markdown_shape_hint(block)}

Immediately preceding COMPILED context (may be truncated):
--- context ---
{prior_context}
--- end context ---

Original block:
--- original ---
{block}
--- end original ---

Sentence signals:
{json.dumps(compact_signals, ensure_ascii=False, indent=2)}

EXACT spans (must survive byte-for-byte):
{json.dumps(exact, ensure_ascii=False)}

HIGH-SURPRISE spans (preserve operational meaning):
{json.dumps(high, ensure_ascii=False)}

YAKE cue candidates (hints only; relations/negation outrank keywords):
{json.dumps(keywords, ensure_ascii=False)}

Rewrite only this block. Output JSON only.
"""


def parse_rewrite(text: str) -> str:
    try:
        obj = json_repair_loads(text)
    except Exception as e:
        raise ValueError(f"Could not parse rewrite JSON: {e}\n{text[:2000]}")
    if not isinstance(obj, dict) or "text" not in obj:
        raise ValueError(f"Rewrite JSON has no text field:\n{text[:2000]}")
    value = obj["text"]
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value


def trailing_newlines(text: str) -> str:
    m = re.search(r"(\r?\n)+$", text)
    return m.group(0) if m else ""


def normalize_candidate(original: str, candidate: str) -> str:
    """
    Keep surrounding source separation stable. The model owns prose, not the
    number of line breaks separating source blocks.
    """
    tail = trailing_newlines(original)
    core = candidate.strip()
    if not core:
        return tail if tail and original.strip() == "" else ""
    return core + tail


def markdown_leader(text: str) -> str | None:
    """
    Return a structural prefix that a non-empty rewrite must preserve.
    This catches an LLM accidentally turning a list item or blockquote into a
    plain paragraph.
    """
    first = text.splitlines()[0] if text.splitlines() else ""
    m = re.match(r"^(\s*(?:[-+*]|\d+[.)]|>)\s+)", first)
    return m.group(1) if m else None


def validate_candidate(
    original: str,
    candidate: str,
    signals: list[SentenceSignal],
) -> list[str]:
    errors = []
    for exact in collect_exact_spans(signals):
        if exact not in candidate:
            errors.append(f"missing EXACT span: {exact!r}")

    if any(s.label == "KEEP" for s in signals) and not candidate.strip():
        errors.append("candidate dropped a block containing KEEP material")

    leader = markdown_leader(original)
    if candidate.strip() and leader and not candidate.startswith(leader):
        errors.append(
            f"candidate changed Markdown structural prefix {leader!r}"
        )

    # Do not let a prose rewrite suddenly introduce a fenced code block.
    if "```" in candidate or "~~~" in candidate:
        errors.append("candidate introduced a fenced code block")

    return errors


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def infer_skill_name(path: Path, text: str) -> str:
    try:
        post = frontmatter.loads(text)
        name = post.metadata.get("name")
        if name:
            return str(name)
    except Exception:
        pass
    return path.parent.name if path.name.upper().startswith("SKILL") else path.stem


def token_count(llm: Llama, text: str) -> int:
    return len(llm.tokenize(text, add_special=False, pieces=False))


def write_progress_output(
    output_path: Path,
    *,
    compiled_prefix: str,
    segments: list[Segment],
    next_index: int,
) -> None:
    remainder = "".join(s.text for s in segments[next_index:])
    atomic_write(output_path, compiled_prefix + remainder)


def save_checkpoint(
    path: Path,
    *,
    input_hash: str,
    next_index: int,
    compiled_prefix: str,
    stats: list[dict[str, Any]],
    profile: dict[str, Any],
) -> None:
    atomic_write(path, json.dumps({
        "input_hash": input_hash,
        "next_index": next_index,
        "compiled_prefix": compiled_prefix,
        "stats": stats,
        "profile": profile,
    }, ensure_ascii=False, indent=2))


def load_checkpoint(path: Path, input_hash: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("input_hash") != input_hash:
        raise RuntimeError(
            f"Checkpoint {path} belongs to a different input file."
        )
    return data


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compile a long Markdown coding-agent skill against a llama.cpp model prior."
    )
    ap.add_argument("skill", type=Path)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--stats", type=Path)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--skill-name")
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument(
        "--rewrite-slot",
        type=int,
        default=None,
        help="Optional second llama.cpp slot for rewrite generations.",
    )
    ap.add_argument("--top-n", type=int, default=128)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--resume", action="store_true")

    # Conservative initial thresholds. Calibrate against your model/skills.
    ap.add_argument("--keep-p90", type=float, default=5.0)
    ap.add_argument("--keep-max", type=float, default=8.0)
    ap.add_argument("--keep-fraction", type=float, default=0.20)
    ap.add_argument("--high-token", type=float, default=6.0)
    ap.add_argument("--drop-p90", type=float, default=2.0)

    ap.add_argument("--rewrite-context-chars", type=int, default=2500)
    ap.add_argument("--rewrite-max-tokens", type=int, default=1024)
    ap.add_argument("--language", default="en", help="YAKE language code.")
    ap.add_argument(
        "--system-prompt-file",
        type=Path,
        help="Override the canonical generic coding-agent system prompt.",
    )
    ap.add_argument(
        "--max-prose-blocks",
        type=int,
        default=0,
        help="Debug: stop after N prose blocks; 0 = all.",
    )
    args = ap.parse_args()

    if args.top_n < 1:
        ap.error("--top-n must be >= 1")

    skill_path = args.skill.resolve()
    original = skill_path.read_text(encoding="utf-8")
    input_hash = sha256_text(original)

    output = args.output or skill_path.with_name(skill_path.stem + ".compiled.md")
    stats_path = args.stats or output.with_suffix(output.suffix + ".stats.json")
    checkpoint_path = args.checkpoint or output.with_suffix(output.suffix + ".checkpoint.json")

    system_prompt = (
        args.system_prompt_file.read_text(encoding="utf-8")
        if args.system_prompt_file
        else GENERIC_AGENT_SYSTEM
    )
    skill_name = args.skill_name or infer_skill_name(skill_path, original)

    llm = Llama(args.url, slot=args.slot)
    props = llm.get("/props")

    console.print(f"[bold]model[/bold]: {props.get('model_path', '?')}")
    console.print(f"[bold]build[/bold]: {props.get('build_info', '?')}")
    console.print(f"[bold]skill[/bold]: {skill_name}")

    segments = markdown_segments(original)
    prose_count = sum(s.kind == "prose" for s in segments)
    protected_chars = sum(len(s.text) for s in segments if s.kind == "protected")
    console.print(
        f"segments={len(segments)} prose={prose_count} "
        f"protected_chars={protected_chars:,}/{len(original):,}"
    )

    rendered_prefix = build_skill_prompt_prefix(
        llm,
        system_prompt=system_prompt,
        skill_name=skill_name,
    )

    profile = {
        "model_path": props.get("model_path"),
        "build_info": props.get("build_info"),
        "server": llm.url,
        "skill_name": skill_name,
        "system_prompt": system_prompt,
        "skill_user_prefix": SKILL_USER_PREFIX,
        "top_n": args.top_n,
        "cache_prompt": not args.no_cache,
        "thresholds": {
            "keep_p90": args.keep_p90,
            "keep_max": args.keep_max,
            "keep_fraction": args.keep_fraction,
            "high_token": args.high_token,
            "drop_p90": args.drop_p90,
        },
    }

    compiled_prefix = ""
    stats: list[dict[str, Any]] = []
    start_index = 0

    if args.resume:
        cp = load_checkpoint(checkpoint_path, input_hash)
        if cp:
            start_index = int(cp["next_index"])
            compiled_prefix = str(cp["compiled_prefix"])
            stats = list(cp.get("stats", []))
            console.print(
                f"[yellow]resuming[/yellow] at segment {start_index}/{len(segments)}"
            )

    # Always create a complete output immediately.
    write_progress_output(
        output,
        compiled_prefix=compiled_prefix,
        segments=segments,
        next_index=start_index,
    )

    processed_prose = 0

    for idx in range(start_index, len(segments)):
        seg = segments[idx]

        if seg.kind != "prose" or not seg.text.strip():
            compiled_prefix += seg.text
            stats.append({
                "segment": idx,
                "kind": seg.kind,
                "lines": [seg.start_line + 1, seg.end_line],
                "action": "verbatim",
                "reason": seg.reason,
                "original_chars": len(seg.text),
                "compiled_chars": len(seg.text),
            })
        else:
            processed_prose += 1
            if args.max_prose_blocks and processed_prose > args.max_prose_blocks:
                console.print("[yellow]debug prose-block limit reached[/yellow]")
                write_progress_output(
                    output,
                    compiled_prefix=compiled_prefix,
                    segments=segments,
                    next_index=idx,
                )
                save_checkpoint(
                    checkpoint_path,
                    input_hash=input_hash,
                    next_index=idx,
                    compiled_prefix=compiled_prefix,
                    stats=stats,
                    profile=profile,
                )
                break

            # IMPORTANT: context contains the already-compiled skill prefix.
            context = rendered_prefix + compiled_prefix

            token_scores, skipped = score_block(
                llm,
                context=context,
                block=seg.text,
                top_n=args.top_n,
                cache_prompt=not args.no_cache,
            )

            signals = classify_sentences(
                seg.text,
                token_scores,
                keep_p90=args.keep_p90,
                keep_max=args.keep_max,
                keep_fraction_threshold=args.keep_fraction,
                high_token_threshold=args.high_token,
                drop_p90=args.drop_p90,
            )

            low_text = " ".join(
                s.text for s in signals if s.label != "KEEP"
            )
            keywords = yake_keywords(low_text, args.language)

            prior = compiled_prefix[-args.rewrite_context_chars:]
            rp = rewrite_prompt(
                block=seg.text,
                signals=signals,
                prior_context=prior,
                keywords=keywords,
            )

            raw = llm.generate_chat(
                REWRITE_SYSTEM,
                rp,
                max_tokens=args.rewrite_max_tokens,
                slot=args.rewrite_slot,
            )

            action = "rewrite"
            errors: list[str] = []
            try:
                candidate = parse_rewrite(raw)
                candidate = normalize_candidate(seg.text, candidate)
                errors = validate_candidate(seg.text, candidate, signals)
            except Exception as e:
                candidate = seg.text
                errors = [f"rewrite parse failure: {e}"]

            original_tokens = token_count(llm, seg.text)
            candidate_tokens = token_count(llm, candidate) if candidate else 0

            # Compression must actually compress. An equal/larger rewrite adds
            # model-authored risk for no context benefit.
            if candidate_tokens >= original_tokens and candidate != seg.text:
                errors.append(
                    f"rewrite did not shrink block ({candidate_tokens} >= {original_tokens} tokens)"
                )

            if errors:
                candidate = seg.text
                candidate_tokens = original_tokens
                action = "kept_original_after_validation"
            elif not candidate.strip():
                action = "dropped"
            elif candidate == seg.text:
                action = "unchanged"

            compiled_prefix += candidate

            mean_regret = (
                statistics.fmean(s.regret_bits for s in token_scores)
                if token_scores else 0.0
            )
            p90_regret = percentile(
                [s.regret_bits for s in token_scores], 0.90
            ) if token_scores else 0.0
            top1_fraction = (
                sum(s.rank == 1 for s in token_scores) / len(token_scores)
                if token_scores else 0.0
            )

            stats.append({
                "segment": idx,
                "kind": "prose",
                "lines": [seg.start_line + 1, seg.end_line],
                "action": action,
                "original_chars": len(seg.text),
                "compiled_chars": len(candidate),
                "original_tokens": original_tokens,
                "compiled_tokens": candidate_tokens,
                "mean_regret_bits": mean_regret,
                "p90_regret_bits": p90_regret,
                "top1_fraction": top1_fraction,
                "boundary_bytes_unscored": skipped,
                "keywords": keywords,
                "validation_errors": errors,
                "sentences": [asdict(s) for s in signals],
                "original": seg.text,
                "compiled": candidate,
            })

            saved = original_tokens - candidate_tokens
            label_counts = {
                k: sum(s.label == k for s in signals)
                for k in ("KEEP", "CONDENSE", "DROP_CANDIDATE")
            }
            console.print(
                f"[cyan]{idx+1}/{len(segments)}[/cyan] "
                f"lines {seg.start_line+1}-{seg.end_line} "
                f"{original_tokens}->{candidate_tokens} tok "
                f"saved={saved:+d} "
                f"p90={p90_regret:.2f} "
                f"K/C/D={label_counts['KEEP']}/{label_counts['CONDENSE']}/{label_counts['DROP_CANDIDATE']} "
                f"[bold]{action}[/bold]"
            )
            if errors:
                for e in errors:
                    console.print(f"  [red]reject:[/red] {e}")

        # Proper output after EVERY segment:
        # compiled prefix + untouched original remainder.
        write_progress_output(
            output,
            compiled_prefix=compiled_prefix,
            segments=segments,
            next_index=idx + 1,
        )
        save_checkpoint(
            checkpoint_path,
            input_hash=input_hash,
            next_index=idx + 1,
            compiled_prefix=compiled_prefix,
            stats=stats,
            profile=profile,
        )

    else:
        # Finished.
        final_text = compiled_prefix
        atomic_write(output, final_text)

        orig_tokens = token_count(llm, original)
        final_tokens = token_count(llm, final_text)
        report = {
            "input": str(skill_path),
            "output": str(output),
            "input_hash": input_hash,
            "profile": profile,
            "original_tokens": orig_tokens,
            "compiled_tokens": final_tokens,
            "saved_tokens": orig_tokens - final_tokens,
            "compression_ratio": (final_tokens / orig_tokens) if orig_tokens else 1.0,
            "segments": stats,
        }
        atomic_write(
            stats_path,
            json.dumps(report, ensure_ascii=False, indent=2),
        )

        # Keep the checkpoint as a reproducibility artifact, but mark complete.
        save_checkpoint(
            checkpoint_path,
            input_hash=input_hash,
            next_index=len(segments),
            compiled_prefix=compiled_prefix,
            stats=stats,
            profile=profile,
        )

        table = Table(title="Skill compilation complete")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Original tokens", f"{orig_tokens:,}")
        table.add_row("Compiled tokens", f"{final_tokens:,}")
        table.add_row("Saved", f"{orig_tokens-final_tokens:,}")
        table.add_row(
            "Remaining",
            f"{(100*final_tokens/orig_tokens):.1f}%" if orig_tokens else "n/a",
        )
        console.print(table)
        console.print(f"[bold green]Markdown:[/bold green] {output}")
        console.print(f"[bold]Stats:[/bold] {stats_path}")
        console.print(f"[bold]Checkpoint:[/bold] {checkpoint_path}")


if __name__ == "__main__":
    main()
