#!/usr/bin/env python3
"""
model-skill-compiler v0.3: section-budget compiler

Compress Markdown coding-agent skills for the exact model loaded in llama.cpp.

Key differences from the earlier paragraph compiler:
  * process whole H1/H2 sections, not individual paragraphs
  * user supplies an explicit reduction target per section
  * surprisal is a RELATIVE priority heatmap inside the section, not a KEEP gate
  * fenced/indented code, raw HTML, tables, frontmatter, and headings are immutable
  * sparse teacher-forced scoring (default: every 4th mutable token)
  * compile left-to-right against the already-compiled prefix
  * retry rewrites that exceed the section token budget
  * compiler-owned Markdown block separators between sections

Example:
  python skill_compiler_v3.py SKILL.md --url http://127.0.0.1:8080 --reduce-percent 50

"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import frontmatter
import requests
from blingfire import text_to_sentences
from json_repair import loads as json_repair_loads
from markdown_it import MarkdownIt
from mdit_py_plugins.front_matter import front_matter_plugin
from rich.console import Console
from rich.table import Table

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

REWRITE_SYSTEM = """You are compiling one SECTION of an operational coding-agent skill into a much shorter representation for the SAME language model that will later read it.

This is lossy model-specific instruction compression, not prose editing.

You receive a HARD mutable-prose token budget. Spend those tokens on information the target model is least likely to infer by itself.

Priority order:
1. project-specific invariants, prohibitions, exact workflow semantics, ordering, scope/ownership, failure behavior
2. exact identifiers, paths, commands, status values, quantities, thresholds, tool semantics
3. information marked high relative surprisal for this model
4. only then generic explanation/rationale

Rules:
- Achieve the requested budget. Do not preserve prose style for its own sake.
- Merge duplicate rules inside the section.
- Generic competent-coder advice should usually disappear.
- Preserve negation, conditions, ordering, exceptions, and modality when they change behavior.
- Protected placeholders like [[PROTECTED_003]] must appear EXACTLY ONCE, unchanged, and in the same order. They stand for code/HTML/tables/headings copied byte-for-byte later.
- EXACT literals supplied by the caller must survive byte-for-byte if they occur in mutable prose.
- Do not invent facts, commands, defaults, tools, or requirements.
- Dense Markdown is desirable: terse clauses, semicolons, compact bullets.
- Do not wrap output in a code fence.

Return JSON only:
{"text":"<compiled masked Markdown section>"}
"""

TIGHTEN_SYSTEM = """Shorten the supplied compiled Markdown section to fit its hard mutable-prose token budget without losing its operational contract.

Never alter/remove/reorder [[PROTECTED_NNN]] placeholders. Preserve all supplied EXACT literals byte-for-byte. Delete explanation and redundancy before deleting rules. Merge repeated rules. Use terse clauses and compact bullets. Do not invent anything.

Return JSON only:
{"text":"<shorter masked Markdown section>"}
"""

INLINE_CODE_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
FLAG_RE = re.compile(r"(?<!\w)--?[A-Za-z][A-Za-z0-9_-]*(?:=[^\s,;]+)?")
URL_RE = re.compile(r"https?://[^\s)>]+")
PATH_RE = re.compile(
    r"(?<!\w)(?:[A-Za-z]:[\\/][^\s`\"']+|(?:\.{0,2}[\\/])(?:[\w.\-]+[\\/])+[\w.\-]+|"
    r"(?:[\w.\-]+[\\/]){2,}[\w.\-]+|[\w.\-]+\.(?:md|json|ya?ml|toml|ini|cfg|py|js|ts|tsx|jsx|cs|cpp|c|h|html|css|xml|sql|sh|ps1|bat|cmd))"
)
NUMBER_RE = re.compile(r"(?<![\w.])(?:\d+(?:\.\d+)?(?:%|ms|s|min|h|k|m|gb|mb|kb|tokens?)?|0x[0-9A-Fa-f]+)(?![\w.])", re.I)
PLACEHOLDER_RE = re.compile(r"\[\[PROTECTED_\d{3}\]\]")


@dataclass
class ProtectedChunk:
    placeholder: str
    start_line: int
    end_line: int
    text: str


@dataclass
class Section:
    index: int
    start_line: int
    end_line: int
    title: str
    raw: str
    masked: str
    protected: list[ProtectedChunk]
    protected_line_ranges: list[tuple[int, int]]


@dataclass
class Sample:
    token_index: int
    start_byte: int
    end_byte: int
    text: str
    regret_bits: float
    surprisal_bits: float
    rank: int | None
    censored: bool
    percentile: float = 0.0


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
            raise RuntimeError(f"{path}: HTTP {r.status_code}\n{r.text[:5000]}")
        return r.json()

    def apply_template(self, messages: list[dict[str, str]]) -> str:
        return self.post("/apply-template", {
            "messages": messages,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_format": "none",
        })["prompt"]

    def tokenize(self, text: str, *, add_special: bool = False, pieces: bool = False):
        return self.post("/tokenize", {
            "content": text,
            "add_special": add_special,
            "parse_special": True,
            "with_pieces": pieces,
        })["tokens"]

    def score_one(self, prompt_ids: list[int], *, top_n: int, cache_prompt: bool) -> dict[str, Any]:
        r = self.post("/completion", {
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
        probs = r.get("completion_probabilities") or []
        if not probs:
            raise RuntimeError("llama.cpp returned no completion_probabilities")
        return probs[0]

    def generate(self, system: str, user: str, *, max_tokens: int, slot: int | None = None) -> str:
        prompt = self.apply_template([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        r = self.post("/completion", {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0,
            "stream": False,
            "cache_prompt": False,
            "id_slot": self.slot if slot is None else slot,
            "seed": 1,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "dry_multiplier": 0.0,
        })
        return r.get("content", "")


def piece_bytes(tok: Any) -> bytes:
    p = tok["piece"]
    return p.encode("utf-8") if isinstance(p, str) else bytes(p)


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted(ranges):
        if b <= a:
            continue
        if not out or a > out[-1][1]:
            out.append([a, b])
        else:
            out[-1][1] = max(out[-1][1], b)
    return [(a, b) for a, b in out]


def standalone_html_ranges(lines: list[str]) -> list[tuple[int, int]]:
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
                out.append((start, len(lines)))
                return out
        else:
            i += 1
    return out


def parse_sections(text: str, section_level: int = 2) -> list[Section]:
    lines = text.splitlines(keepends=True)
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    md.use(front_matter_plugin)
    tokens = md.parse(text)

    protected_ranges: list[tuple[int, int]] = []
    heading_starts: list[tuple[int, int, str]] = []

    for i, tok in enumerate(tokens):
        if not tok.map:
            continue
        a, b = tok.map
        if tok.type in {"front_matter", "fence", "code_block", "html_block", "table_open", "hr"}:
            protected_ranges.append((a, b))
        if tok.type == "heading_open":
            protected_ranges.append((a, b))
            level = int(tok.tag[1:]) if tok.tag.startswith("h") else 9
            # inline token follows heading_open in markdown-it
            title = ""
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                title = tokens[i + 1].content
            if level <= section_level:
                heading_starts.append((a, level, title))

    protected_ranges.extend(standalone_html_ranges(lines))
    protected_ranges = merge_ranges(protected_ranges)

    boundaries = sorted({0, len(lines), *[a for a, _level, _title in heading_starts]})
    title_by_start = {a: title for a, _level, title in heading_starts}

    sections: list[Section] = []
    for si, (a, b) in enumerate(zip(boundaries, boundaries[1:])):
        if b <= a:
            continue
        raw = "".join(lines[a:b])
        local_protected = []
        for p0, p1 in protected_ranges:
            x0, x1 = max(a, p0), min(b, p1)
            if x1 > x0:
                local_protected.append((x0, x1))
        local_protected = merge_ranges(local_protected)

        chunks: list[ProtectedChunk] = []
        cursor = a
        masked_parts: list[str] = []
        for n, (p0, p1) in enumerate(local_protected):
            masked_parts.append("".join(lines[cursor:p0]))
            ph = f"[[PROTECTED_{n:03d}]]"
            exact = "".join(lines[p0:p1])
            chunks.append(ProtectedChunk(ph, p0, p1, exact))
            # Put placeholder on its own line; restored bytes are exact later.
            masked_parts.append("\n" + ph + "\n")
            cursor = p1
        masked_parts.append("".join(lines[cursor:b]))
        masked = "".join(masked_parts)

        sections.append(Section(
            index=len(sections),
            start_line=a,
            end_line=b,
            title=title_by_start.get(a, "(preamble)"),
            raw=raw,
            masked=masked,
            protected=chunks,
            protected_line_ranges=local_protected,
        ))
    return sections


def build_skill_prompt_prefix(llm: Llama, system_prompt: str, skill_name: str) -> str:
    sentinel = "__MODEL_SKILL_COMPILER_BODY_9f6c1c4a__"
    user = SKILL_USER_PREFIX.format(skill_name=skill_name) + sentinel
    rendered = llm.apply_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ])
    if sentinel not in rendered:
        raise RuntimeError("chat template transformed/removed skill sentinel")
    return rendered.split(sentinel, 1)[0]


def tokenize_continuation(llm: Llama, context: str, continuation: str):
    full = llm.tokenize(context + continuation, add_special=False, pieces=True)
    ids = [int(x["id"]) for x in full]
    pieces = [piece_bytes(x) for x in full]
    boundary = len(context.encode("utf-8"))
    off = 0
    start = len(ids)
    skipped = 0
    for i, pb in enumerate(pieces):
        a, b = off, off + len(pb)
        if a >= boundary:
            start = i
            break
        if a < boundary < b:
            skipped = b - boundary
        off = b
    return ids[:start], ids[start:], pieces[start:], skipped


def line_ranges_to_byte_ranges(section: Section) -> list[tuple[int, int]]:
    lines = section.raw.splitlines(keepends=True)
    prefix = [0]
    for line in lines:
        prefix.append(prefix[-1] + len(line.encode("utf-8")))
    out = []
    for a_abs, b_abs in section.protected_line_ranges:
        a = max(0, a_abs - section.start_line)
        b = max(0, b_abs - section.start_line)
        a = min(a, len(lines)); b = min(b, len(lines))
        out.append((prefix[a], prefix[b]))
    return merge_ranges(out)


def inside_ranges(a: int, b: int, ranges: list[tuple[int, int]]) -> bool:
    return any(max(a, x0) < min(b, x1) for x0, x1 in ranges)


def distribution_score(target: int, item: dict[str, Any]):
    top = item.get("top_logprobs") or []
    generated = int(item["id"])
    if not top:
        lp = item.get("logprob")
        if generated == target and lp is not None:
            lp = float(lp)
            return 1, lp, max(0.0, -lp / LN2), 0.0, False
        raise RuntimeError("empty top_logprobs on mismatch")
    top = sorted(top, key=lambda x: x["logprob"], reverse=True)
    top1 = float(top[0]["logprob"])
    for rank, x in enumerate(top, 1):
        if int(x["id"]) == target:
            lp = float(x["logprob"])
            return rank, lp, max(0.0, -lp / LN2), max(0.0, (top1 - lp) / LN2), False
    cutoff = float(top[-1]["logprob"])
    return None, None, max(0.0, -cutoff / LN2), max(0.0, (top1 - cutoff) / LN2), True


def sparse_score_section(
    llm: Llama,
    context: str,
    section: Section,
    *,
    top_n: int,
    stride: int,
    cache_prompt: bool,
) -> list[Sample]:
    prefix_ids, ids, pieces, skipped = tokenize_continuation(llm, context, section.raw)
    protected_bytes = line_ranges_to_byte_ranges(section)

    offsets = []
    off = skipped
    mutable_ord = 0
    sample_indices = []
    for i, pb in enumerate(pieces):
        a, b = off, off + len(pb)
        offsets.append((a, b))
        if not inside_ranges(a, b, protected_bytes):
            if mutable_ord % stride == 0:
                sample_indices.append(i)
            mutable_ord += 1
        off = b

    samples = []
    for n, i in enumerate(sample_indices, 1):
        item = llm.score_one(prefix_ids + ids[:i], top_n=top_n, cache_prompt=cache_prompt)
        rank, lp, surprise, regret, censored = distribution_score(ids[i], item)
        a, b = offsets[i]
        samples.append(Sample(
            token_index=i,
            start_byte=a,
            end_byte=b,
            text=pieces[i].decode("utf-8", "replace"),
            regret_bits=regret,
            surprisal_bits=surprise,
            rank=rank,
            censored=censored,
        ))

    # Relative percentile is the useful signal; absolute regret is model/framing dependent.
    vals = sorted(s.regret_bits for s in samples)
    for s in samples:
        if len(vals) <= 1:
            s.percentile = 1.0
        else:
            # rightmost rank among equal values
            rank = max(i for i, v in enumerate(vals) if v <= s.regret_bits)
            s.percentile = rank / (len(vals) - 1)
    return samples


def exact_spans(text: str) -> list[str]:
    out = []
    # Inline code is the most important exact class. Other exacts are retained too,
    # but avoid protecting bare common integers inside placeholders.
    for rx in (INLINE_CODE_RE, FLAG_RE, URL_RE, PATH_RE, NUMBER_RE):
        for m in rx.finditer(text):
            s = m.group(0)
            if PLACEHOLDER_RE.fullmatch(s):
                continue
            if s not in out:
                out.append(s)
    return out


def mutable_text(masked: str) -> str:
    return PLACEHOLDER_RE.sub("", masked)


def token_count(llm: Llama, text: str) -> int:
    return len(llm.tokenize(text, add_special=False, pieces=False))


def snippet_around(raw: str, sample: Sample, radius: int = 55) -> str:
    b = raw.encode("utf-8")
    a = max(0, sample.start_byte - radius)
    z = min(len(b), sample.end_byte + radius)
    # repair UTF-8 boundaries
    while a < sample.start_byte:
        try:
            part = b[a:z].decode("utf-8")
            break
        except UnicodeDecodeError:
            a += 1
    else:
        part = sample.text
    part = re.sub(r"\s+", " ", part).strip()
    return part[:220]


def priority_excerpts(section: Section, samples: list[Sample], limit: int = 18):
    ranked = sorted(samples, key=lambda s: (s.percentile, s.regret_bits), reverse=True)
    out = []
    seen = set()
    for s in ranked:
        if s.percentile < 0.70:
            break
        snip = snippet_around(section.raw, s)
        key = snip.lower()
        if not snip or key in seen:
            continue
        seen.add(key)
        out.append({
            "priority_percentile": round(s.percentile * 100),
            "regret_bits": round(s.regret_bits, 2),
            "excerpt": snip,
        })
        if len(out) >= limit:
            break
    return out


def restore_protected(masked: str, chunks: list[ProtectedChunk]) -> str:
    out = masked
    for c in chunks:
        out = out.replace(c.placeholder, c.text)
    return out


def validate_masked(candidate: str, section: Section, exact: list[str]) -> list[str]:
    errors = []
    positions = []
    for c in section.protected:
        count = candidate.count(c.placeholder)
        if count != 1:
            errors.append(f"{c.placeholder} occurs {count} times")
        else:
            positions.append(candidate.index(c.placeholder))
    if positions != sorted(positions):
        errors.append("protected placeholders reordered")
    for x in exact:
        if x not in candidate:
            errors.append(f"missing exact literal {x!r}")
    # No new placeholders.
    allowed = {c.placeholder for c in section.protected}
    for ph in PLACEHOLDER_RE.findall(candidate):
        if ph not in allowed:
            errors.append(f"invented placeholder {ph}")
    return errors


def parse_json_text(raw: str) -> str:
    obj = json_repair_loads(raw)
    if not isinstance(obj, dict) or "text" not in obj:
        raise ValueError("model response missing JSON text field")
    return str(obj["text"] or "")


def rewrite_section(
    llm: Llama,
    section: Section,
    samples: list[Sample],
    compiled_prefix: str,
    *,
    reduce_percent: float,
    tolerance: float,
    retries: int,
    context_chars: int,
    rewrite_slot: int | None,
) -> tuple[str, dict[str, Any]]:
    exact = exact_spans(mutable_text(section.masked))
    orig_mutable = token_count(llm, mutable_text(section.masked))
    target = max(1, round(orig_mutable * (1.0 - reduce_percent / 100.0))) if orig_mutable else 0

    # Nothing mutable -> verbatim.
    if orig_mutable == 0:
        return section.raw, {
            "original_mutable_tokens": 0,
            "target_mutable_tokens": 0,
            "compiled_mutable_tokens": 0,
            "attempts": 0,
            "validation_errors": [],
        }

    priorities = priority_excerpts(section, samples)
    prior = compiled_prefix[-context_chars:]
    user = f"""Section title: {section.title}

Hard budget:
- original mutable prose: {orig_mutable} tokens
- reduce by: {reduce_percent:.1f}%
- OUTPUT mutable prose MUST be <= {target} tokens (protected placeholders do not count)

Already-compiled context immediately before this section:
--- context ---
{prior}
--- end context ---

Masked original section. Protected material is represented by immutable placeholders:
--- section ---
{section.masked}
--- end section ---

Highest RELATIVE surprisal excerpts for this model (ranking hints, not mandatory verbatim text):
{json.dumps(priorities, ensure_ascii=False, indent=2)}

EXACT mutable literals that must survive byte-for-byte:
{json.dumps(exact, ensure_ascii=False)}

Compile the WHOLE SECTION to the hard budget. Merge duplicate rules and remove generic explanation aggressively. Return JSON only.
"""

    # First generation limit includes placeholder overhead + JSON overhead.
    placeholder_tok = token_count(llm, "\n".join(c.placeholder for c in section.protected))
    max_gen = max(128, target + placeholder_tok + 160)

    candidates: list[tuple[int, str, list[str], int]] = []
    raw = llm.generate(REWRITE_SYSTEM, user, max_tokens=max_gen, slot=rewrite_slot)

    for attempt in range(retries + 1):
        try:
            cand = parse_json_text(raw).strip()
        except Exception as e:
            cand = ""
            errors = [f"parse: {e}"]
        else:
            errors = validate_masked(cand, section, exact)

        cand_mutable = token_count(llm, mutable_text(cand)) if cand else 0
        if cand and not errors:
            candidates.append((cand_mutable, cand, [], attempt + 1))
            if cand_mutable <= math.ceil(target * (1.0 + tolerance)):
                return restore_protected(cand, section.protected), {
                    "original_mutable_tokens": orig_mutable,
                    "target_mutable_tokens": target,
                    "compiled_mutable_tokens": cand_mutable,
                    "attempts": attempt + 1,
                    "validation_errors": [],
                    "priority_excerpts": priorities,
                }

        if attempt >= retries:
            break

        # Tighten the best valid candidate if available; otherwise retry original.
        basis = min(candidates, key=lambda x: x[0])[1] if candidates else section.masked
        tighten = f"""Hard mutable-prose budget: <= {target} tokens.
Current mutable prose: {token_count(llm, mutable_text(basis))} tokens.

Current masked section:
--- section ---
{basis}
--- end section ---

Immutable placeholders in order:
{json.dumps([c.placeholder for c in section.protected])}

EXACT literals:
{json.dumps(exact, ensure_ascii=False)}

Shorten further. Do not reconstruct deleted explanation. Return JSON only.
"""
        raw = llm.generate(TIGHTEN_SYSTEM, tighten, max_tokens=max_gen, slot=rewrite_slot)

    if candidates:
        best_n, best, _e, attempts = min(candidates, key=lambda x: x[0])
        return restore_protected(best, section.protected), {
            "original_mutable_tokens": orig_mutable,
            "target_mutable_tokens": target,
            "compiled_mutable_tokens": best_n,
            "attempts": attempts,
            "validation_errors": ["budget missed after retries"],
            "priority_excerpts": priorities,
        }

    return section.raw, {
        "original_mutable_tokens": orig_mutable,
        "target_mutable_tokens": target,
        "compiled_mutable_tokens": orig_mutable,
        "attempts": retries + 1,
        "validation_errors": ["no valid rewrite; kept original"],
        "priority_excerpts": priorities,
    }



def ensure_markdown_section_separator(text: str) -> str:
    """
    Ensure the next Markdown section starts as a real block.

    Rewriters may shorten prose, but they do not own inter-section whitespace.
    This prevents output such as:

        ... stop.## Next Section

    Only NEWLINES are appended. Nothing is stripped, because the section may
    end in immutable code/HTML/table content whose bytes must remain exact.
    """
    if not text:
        return text
    if text.endswith("\n\n"):
        return text
    if text.endswith("\n"):
        return text + "\n"
    return text + "\n\n"

def atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    except:
        print(text)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def infer_skill_name(path: Path, text: str) -> str:
    try:
        name = frontmatter.loads(text).metadata.get("name")
        if name:
            return str(name)
    except Exception:
        pass
    return path.parent.name if path.name.upper().startswith("SKILL") else path.stem


def main():
    ap = argparse.ArgumentParser(description="Section-budget, model-specific coding-agent skill compiler")
    ap.add_argument("skill", type=Path)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--stats", type=Path)
    ap.add_argument("--skill-name")
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--rewrite-slot", type=int, default=None,
                    help="Use another llama.cpp slot for rewriting if --parallel >= 2.")
    ap.add_argument("--reduce-percent", type=float, default=50.0,
                    help="Percent of MUTABLE PROSE to remove from every section. Default: 50.")
    ap.add_argument("--section-level", type=int, default=2,
                    help="Start a new compilation section at headings <= this level. Default: 2.")
    ap.add_argument("--sample-stride", type=int, default=4,
                    help="Score every Nth mutable token. Default: 4. 1 = exact/slow.")
    ap.add_argument("--top-n", type=int, default=128)
    ap.add_argument("--budget-tolerance", type=float, default=0.08,
                    help="Allow this fractional budget overflow before retrying. Default: .08")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--rewrite-context-chars", type=int, default=4000)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--system-prompt-file", type=Path)
    args = ap.parse_args()

    if not 0 <= args.reduce_percent < 100:
        ap.error("--reduce-percent must be >=0 and <100")
    if args.sample_stride < 1:
        ap.error("--sample-stride must be >=1")
    if args.section_level < 1 or args.section_level > 6:
        ap.error("--section-level must be 1..6")

    src_path = args.skill.resolve()
    original = src_path.read_text(encoding="utf-8")
    output = args.output or src_path.with_name(src_path.stem + ".compiled.md")
    stats_path = args.stats or output.with_suffix(output.suffix + ".stats.json")
    system = args.system_prompt_file.read_text(encoding="utf-8") if args.system_prompt_file else GENERIC_AGENT_SYSTEM
    skill_name = args.skill_name or infer_skill_name(src_path, original)

    llm = Llama(args.url, args.slot)
    props = llm.get("/props")
    sections = parse_sections(original, args.section_level)
    rendered_prefix = build_skill_prompt_prefix(llm, system, skill_name)

    console.print(f"[bold]model[/bold]: {props.get('model_path', '?')}")
    console.print(f"[bold]skill[/bold]: {skill_name}")
    console.print(f"sections={len(sections)} reduce={args.reduce_percent:.1f}% sample_stride={args.sample_stride}")
    if args.rewrite_slot is None:
        console.print("[yellow]note:[/yellow] scorer and rewriter share one slot; --parallel 2 + --rewrite-slot 1 preserves more scorer cache.")

    compiled = ""
    report_sections = []

    for sec in sections:
        # Sections with no mutable text are exact passthrough and need no scoring/generation.
        orig_mutable = token_count(llm, mutable_text(sec.masked))
        if orig_mutable == 0:
            compiled_sec = sec.raw
            info = {
                "original_mutable_tokens": 0,
                "target_mutable_tokens": 0,
                "compiled_mutable_tokens": 0,
                "attempts": 0,
                "validation_errors": [],
                "priority_excerpts": [],
            }
            samples = []
        else:
            context = rendered_prefix + compiled
            samples = sparse_score_section(
                llm, context, sec,
                top_n=args.top_n,
                stride=args.sample_stride,
                cache_prompt=not args.no_cache,
            )
            compiled_sec, info = rewrite_section(
                llm, sec, samples, compiled,
                reduce_percent=args.reduce_percent,
                tolerance=args.budget_tolerance,
                retries=args.retries,
                context_chars=args.rewrite_context_chars,
                rewrite_slot=args.rewrite_slot,
            )

        # Inter-section whitespace is compiler-owned, not model-owned.
        compiled_sec = ensure_markdown_section_separator(compiled_sec)
        compiled += compiled_sec

        # Proper outfile after every section: compiled prefix + untouched remainder.
        remainder = "".join(s.raw for s in sections[sec.index + 1:])
        atomic_write(output, compiled + remainder)

        total_orig = token_count(llm, sec.raw)
        total_new = token_count(llm, compiled_sec)
        report_sections.append({
            "section": sec.index,
            "title": sec.title,
            "lines": [sec.start_line + 1, sec.end_line],
            "original_total_tokens": total_orig,
            "compiled_total_tokens": total_new,
            "saved_total_tokens": total_orig - total_new,
            "sample_count": len(samples),
            "sample_stride": args.sample_stride,
            **info,
        })

        target = info["target_mutable_tokens"]
        got = info["compiled_mutable_tokens"]
        status = "ok" if target == 0 or got <= math.ceil(target * (1 + args.budget_tolerance)) else "over"
        console.print(
            f"[{sec.index+1:>2}/{len(sections)}] {sec.title[:42]:42} "
            f"mutable {info['original_mutable_tokens']}->{got} target={target} "
            f"total {total_orig}->{total_new} samples={len(samples)} [{status}]"
        )

    atomic_write(output, compiled)
    orig_tokens = token_count(llm, original)
    new_tokens = token_count(llm, compiled)
    report = {
        "input": str(src_path),
        "output": str(output),
        "input_hash": sha256_text(original),
        "model_path": props.get("model_path"),
        "build_info": props.get("build_info"),
        "skill_name": skill_name,
        "reduce_percent_mutable_prose": args.reduce_percent,
        "section_level": args.section_level,
        "sample_stride": args.sample_stride,
        "original_tokens": orig_tokens,
        "compiled_tokens": new_tokens,
        "saved_tokens": orig_tokens - new_tokens,
        "remaining_percent": 100.0 * new_tokens / orig_tokens if orig_tokens else 100.0,
        "sections": report_sections,
    }
    atomic_write(stats_path, json.dumps(report, ensure_ascii=False, indent=2))

    t = Table(title="Skill compilation complete")
    t.add_column("Metric"); t.add_column("Value", justify="right")
    t.add_row("Original tokens", f"{orig_tokens:,}")
    t.add_row("Compiled tokens", f"{new_tokens:,}")
    t.add_row("Saved", f"{orig_tokens-new_tokens:,}")
    t.add_row("Remaining", f"{100*new_tokens/orig_tokens:.1f}%" if orig_tokens else "n/a")
    console.print(t)
    console.print(f"[bold green]Markdown:[/bold green] {output}")
    console.print(f"[bold]Stats:[/bold] {stats_path}")


if __name__ == "__main__":
    main()
