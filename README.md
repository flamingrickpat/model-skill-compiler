# model-skill-compiler

Are you sad because your big-boy skills with a million tokens aren't working as well in your local models
because you're a dirty, unwashed **VRAMLET**?

Compile your skills into a compressed version, tailored to your exact model. Llamacpp offers a bunch of cool API stuff.
We can find out how "surprised" the model is by your skill instructions, and distill a version where the surprising parts 
are written out fully, and the "implicit" parts are reduced to keywords.

#### General idea
1. Make generic system prompt and user prompt 
2. Split skill into batches (like new sections)
3. Start generation with SYS + USER + (first_letter_of_skill_content)
4. Compare the sampled logits with the actual skills text
5. Do this for every letter/word
6. When this is done for the whole sentence/section, you can see what parts of the text "surprised" the LLM the most
7. Reduce those parts to keywords and compress "implicit" knowledge
8. Use that part as new skill beginning and do it for the next batch (next optimization step takes first changed part into account)

### Does this affect skill performance?

Probably. The LLM being able to autocomplete a skill doesn't mean that it will actually follow its rules.

### Does this work with (any_paid_api)?

Dunno, you need access to "top_logprobs".

## Safety policy in v0.1

Copied byte-for-byte:

- YAML frontmatter
- headings
- fenced and indented code examples
- raw HTML
- complete `<html>...</html>` templates
- Markdown tables
- unknown/structural Markdown

Only Markdown prose paragraphs/list-item paragraphs are rewrite candidates.

## Install

```powershell
pip install -r requirements.txt
```

or:

```powershell
pip install -e .
```

## Run

```powershell
python .\skill_compiler.py .\agents\skills\foo\SKILL.md `
  --url http://127.0.0.1:8080
```

Default artifacts beside the input:

- `SKILL.compiled.md` — the actual compiled skill
- `SKILL.compiled.md.stats.json` — detailed score/rewrite diagnostics
- `SKILL.compiled.md.checkpoint.json` — resume state

The Markdown output is rewritten after every block as:

`compiled prefix + untouched original remainder`

so an interrupted run still leaves a complete usable skill.

Resume:

```powershell
python .\skill_compiler.py .\agents\skills\foo\SKILL.md `
  --url http://127.0.0.1:8080 `
  --resume
```

## Important llama.cpp detail

Scoring uses `n_predict=1` intentionally. Some builds return empty
`top_logprobs` when `n_probs` is combined with multi-token generation.

## Initial thresholds

Defaults are deliberately conservative:

- sentence `KEEP` if p90 regret >= 5 bits
- or max regret >= 8 bits
- or >=20% of tokens are >=6-bit regret
- or it contains exact literals such as inline code/paths/numbers
- sentence `DROP_CANDIDATE` only when p90 < 2 bits and it has no strong
  semantic operator such as `never`, `only`, `before`, `unless`, etc.

These are starting points, not universal constants. The statistics sidecar is
intended for calibrating them against your own model and agent suite.

## Libraries deliberately used

- `markdown-it-py` + `mdit-py-plugins`: Markdown/source block structure
- `BlingFire`: sentence segmentation
- `YAKE`: low-information cue/keyphrase hints
- `json-repair`: robust parsing of local-model rewrite JSON
- `python-frontmatter`: role-name metadata
- `requests`: llama.cpp native HTTP API
- `rich`: progress/reporting

The custom code is limited to the model-specific scoring/compiler logic,
source-range preservation, and the special case of preserving a complete raw
HTML document across CommonMark HTML-block splits.
