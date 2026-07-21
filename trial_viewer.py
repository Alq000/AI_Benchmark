"""
Agent Trial Viewer
==================

A Streamlit viewer for the JSON logs produced by ``run_benchmark.py`` (e.g.
``trial_1.json``).  Each log is an agent "run" in which an LLM plays a research
physicist trying to reverse-engineer a hidden non-linear ODE.  The agent speaks
through three XML actions embedded in its messages:

    <run_experiment>{...}</run_experiment>   -> sample 100 trajectories
    <run_python>...</run_python>             -> run analysis code in a sandbox
    <submission>[{term, coeff}]</submission> -> final answer for k_0 (coeff of x)

and the environment answers with ``<experiment_output>`` blocks or
``[Python Execution Results]`` text.

This module is split into two halves:

  * A pure-Python parsing layer (stdlib only) that turns a raw ``chat_history``
    into typed, ordered "blocks".  It is import-safe and unit-testable without
    Streamlit installed.
  * A Streamlit UI (guarded under ``__main__``) that renders an Overview across
    all discovered trials and a rich per-trial conversation view.

Run with:  streamlit run trial_viewer.py
"""

from __future__ import annotations

import glob
import html
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Parsing layer  (stdlib only -- safe to import for tests)
# ---------------------------------------------------------------------------

# The four XML tags that can appear anywhere inside a message body.  We use a
# back-reference so opening/closing tags must match, and DOTALL so bodies may
# span many lines.
_TAG_RE = re.compile(
    r"<(run_experiment|run_python|submission|experiment_output)>(.*?)</\1>",
    re.DOTALL,
)

# Plain-text marker the host prepends to sandbox stdout/stderr (see
# run_benchmark.py).  It is NOT an XML tag, so we detect it by prefix.
_PY_RESULT_MARKER = "[Python Execution Results]"

# Substrings that identify a host error/retry message (see prompts.ERROR_PROMPT
# and the exception branches in run_benchmark.run_trial).
_ERROR_MARKERS = (
    "Submission failed",
    "Experiment failed",
    "formatted incorrectly or caused an error",
)


@dataclass
class Block:
    """One renderable unit inside a message."""

    kind: str            # experiment | python | submission | experiment_output |
                         # python_output | error | prose | system
    body: str            # raw text of the block
    data: Any = None     # parsed JSON payload where applicable
    hallucinated: bool = False   # a tool-output the *assistant* fabricated
    error: bool = False          # output/text that carries a failure signal


def _try_json(text: str) -> Any:
    """Best-effort JSON parse; returns None on failure."""
    try:
        return json.loads(text)
    except Exception:
        return None


def _split_tags(content: str) -> list[tuple[str, str]]:
    """Split ``content`` into an ordered list of (tag, body) pairs.

    Text that sits *between* tags is returned with the pseudo-tag ``"text"``.
    Whitespace-only gaps are dropped.
    """
    segments: list[tuple[str, str]] = []
    pos = 0
    for m in _TAG_RE.finditer(content):
        if m.start() > pos:
            gap = content[pos:m.start()]
            if gap.strip():
                segments.append(("text", gap.strip()))
        segments.append((m.group(1), m.group(2).strip()))
        pos = m.end()
    if pos < len(content):
        tail = content[pos:]
        if tail.strip():
            segments.append(("text", tail.strip()))
    # A message with no tags at all still yields its whole text.
    if not segments and content.strip():
        segments.append(("text", content.strip()))
    return segments


def _looks_like_py_output(text: str) -> bool:
    """The sandbox marker usually leads the message but may trail a stray
    newline / prefix, so we look near the start."""
    return _PY_RESULT_MARKER in text[:80]


def _output_has_error(text: str) -> bool:
    return ("Traceback (most recent call last)" in text
            or "\nError" in text
            or re.search(r"^\w*Error\b", text) is not None
            or "Exception" in text)


def classify_message(role: str, content: str) -> list[Block]:
    """Turn a single ``{role, content}`` message into ordered :class:`Block`s.

    The same routine handles assistant turns (which own the action tags) and
    environment/user turns (which own tool outputs and error prompts).  A tool
    output that appears inside an *assistant* message is flagged
    ``hallucinated`` -- the model fabricated a response the host never sent.

    The system prompt is treated as one opaque block: it embeds *illustrative*
    ``<run_experiment>`` / ``<run_python>`` / ``<submission>`` examples that must
    not be mistaken for real actions.
    """
    if role == "system":
        return [Block("system", content)]

    blocks: list[Block] = []
    for tag, body in _split_tags(content):
        if tag == "run_experiment":
            blocks.append(Block("experiment", body, data=_try_json(body)))
        elif tag == "run_python":
            blocks.append(Block("python", body))
        elif tag == "submission":
            blocks.append(Block("submission", body, data=_try_json(body)))
        elif tag == "experiment_output":
            blocks.append(Block(
                "experiment_output", body, data=_try_json(body),
                hallucinated=(role == "assistant"),
            ))
        else:  # "text"
            if _looks_like_py_output(body):
                blocks.append(Block(
                    "python_output", body,
                    hallucinated=(role == "assistant"),
                    error=_output_has_error(body),
                ))
            elif role == "system":
                blocks.append(Block("system", body))
            elif role != "assistant" and any(mk in body for mk in _ERROR_MARKERS):
                blocks.append(Block("error", body, error=True))
            else:
                blocks.append(Block("prose", body))
    return blocks


# --- submission / scoring helpers -----------------------------------------

def submission_terms(data: Any) -> list[tuple[str, Any]]:
    """Normalise a submission payload into a list of (term, coeff)."""
    out: list[tuple[str, Any]] = []
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and "term" in entry:
                out.append((str(entry.get("term")), entry.get("coeff")))
    return out


def _term_is_x(term: str) -> bool:
    """Does this term denote the bare linear displacement ``x``?"""
    t = str(term).replace(" ", "").replace("*", "").lower()
    return t in ("x", "1.0x", "1x")


def extract_k0(data: Any) -> float | None:
    """Pull the coefficient of the linear ``x`` term from a submission."""
    for term, coeff in submission_terms(data):
        if _term_is_x(term):
            try:
                return float(coeff)
            except (TypeError, ValueError):
                return None
    return None


def equation_string(data: Any) -> str:
    """Render a submission as a human-readable ``x_ddot = ...`` string."""
    parts = submission_terms(data)
    if not parts:
        return "ẍ = ?"
    segs: list[str] = []
    for term, coeff in parts:
        try:
            c = float(coeff)
            sign = "−" if c < 0 else "+"
            segs.append(f"{sign} {abs(c):.4g}·{term}")
        except (TypeError, ValueError):
            segs.append(f"+ {coeff}·{term}")
    body = " ".join(segs).lstrip("+ ").strip()
    # A leading minus keeps its sign but drops the stray space.
    body = re.sub(r"^−\s+", "−", body)
    return "ẍ = " + (body if body else "0")


def experiment_summary(data: Any) -> dict[str, Any] | None:
    """Derive convenient fields from a ``<run_experiment>`` payload."""
    if not isinstance(data, dict):
        return None
    try:
        t_start = float(data.get("t_start", 0.0))
        delta_t = float(data.get("delta_t", 0.0))
        steps = int(data.get("steps", 0))
    except (TypeError, ValueError):
        return None
    return {
        "t_start": t_start,
        "delta_t": delta_t,
        "steps": steps,
        "t_end": t_start + delta_t * steps,
        "points": steps + 1,
    }


# --- trial-level aggregation ----------------------------------------------

@dataclass
class TrialStats:
    trial_id: Any
    status: str
    n_messages: int
    n_turns: int                 # assistant turns
    n_experiments: int
    n_python: int
    n_submissions: int
    n_errors: int
    n_hallucinations: int
    total_chars: int
    submission: Any = None
    k0: float | None = None
    experiments: list[dict] = field(default_factory=list)


def summarize_trial(trial: dict) -> TrialStats:
    chat = trial.get("chat_history", []) or []
    n_exp = n_py = n_sub = n_err = n_hall = 0
    submission = None
    experiments: list[dict] = []
    total_chars = 0
    n_turns = 0
    for msg in chat:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        total_chars += len(content)
        if role == "assistant":
            n_turns += 1
        for b in classify_message(role, content):
            if b.kind == "experiment":
                n_exp += 1
                s = experiment_summary(b.data)
                if s:
                    experiments.append(s)
            elif b.kind == "python":
                n_py += 1
            elif b.kind == "submission":
                n_sub += 1
                if b.data is not None:
                    submission = b.data
            elif b.kind == "error":
                n_err += 1
            if b.hallucinated:
                n_hall += 1
    return TrialStats(
        trial_id=trial.get("trial_id", "?"),
        status=str(trial.get("status", "unknown")),
        n_messages=len(chat),
        n_turns=n_turns,
        n_experiments=n_exp,
        n_python=n_py,
        n_submissions=n_sub,
        n_errors=n_err,
        n_hallucinations=n_hall,
        total_chars=total_chars,
        submission=submission,
        k0=extract_k0(submission),
        experiments=experiments,
    )


def relative_error(k_pred: float | None, true_k0: float | None) -> float | None:
    if k_pred is None or true_k0 in (None, 0):
        return None
    return (k_pred - true_k0) / true_k0


# --- file discovery / loading ---------------------------------------------

def is_trial_obj(obj: Any) -> bool:
    return isinstance(obj, dict) and "chat_history" in obj


def normalize_to_trials(obj: Any) -> list[dict]:
    """Accept a single trial, a list of trials, or a dict-of-trials."""
    if is_trial_obj(obj):
        return [obj]
    if isinstance(obj, list):
        return [o for o in obj if is_trial_obj(o)]
    if isinstance(obj, dict):
        return [v for v in obj.values() if is_trial_obj(v)]
    return []


def load_trial_file(path: str) -> list[dict]:
    with open(path, "r") as f:
        obj = json.load(f)
    return normalize_to_trials(obj)


def discover_trial_files(directory: str) -> list[str]:
    """Find candidate JSON logs in ``directory``.

    Prefers ``trial_*.json`` (natural-sorted), then falls back to scanning any
    other ``*.json`` that actually contains a ``chat_history``.
    """
    directory = directory or "."
    named = glob.glob(os.path.join(directory, "trial_*.json"))

    def _key(p: str):
        m = re.search(r"(\d+)", os.path.basename(p))
        return (int(m.group(1)) if m else 1 << 30, os.path.basename(p))

    named.sort(key=_key)
    found = list(named)
    named_set = set(found)
    for p in sorted(glob.glob(os.path.join(directory, "*.json"))):
        if p in named_set:
            continue
        try:
            with open(p, "r") as f:
                if is_trial_obj(json.load(f)):
                    found.append(p)
        except Exception:
            continue
    return found


def autodetect_true_k0(directory: str) -> tuple[float | None, str | None]:
    """Regex-scrape ``TRUE_COEFFS['k_0']`` from a diffeq config (no exec)."""
    for path in sorted(glob.glob(os.path.join(directory or ".", "diffeq_config*.py"))):
        try:
            with open(path, "r") as f:
                txt = f.read()
        except Exception:
            continue
        block = re.search(r"TRUE_COEFFS\s*=\s*\{(.*?)\}", txt, re.DOTALL)
        scope = block.group(1) if block else txt
        m = re.search(r"['\"]k_0['\"]\s*:\s*(-?\d+(?:\.\d+)?)", scope)
        if m:
            return float(m.group(1)), os.path.basename(path)
    return None, None


# ===========================================================================
# Streamlit UI  (only runs under `streamlit run` / __main__)
# ===========================================================================

def _run_app() -> None:  # pragma: no cover - exercised via the app, not tests
    import pandas as pd
    import streamlit as st

    st.set_page_config(
        page_title="Agent Trial Viewer",
        page_icon="\U0001F52C",
        layout="wide",
    )

    _inject_css(st)

    # -- Sidebar: data source -------------------------------------------
    st.sidebar.title("\U0001F52C Agent Trial Viewer")
    st.sidebar.caption("Inspect LLM diff-eq benchmark runs")

    default_dir = os.path.dirname(os.path.abspath(__file__))
    directory = st.sidebar.text_input("Logs folder", value=default_dir)

    uploads = st.sidebar.file_uploader(
        "…or upload trial JSON files",
        type=["json"],
        accept_multiple_files=True,
    )

    trials, sources = _gather_trials(directory, uploads, st)

    if not trials:
        st.info(
            "No trial logs found. Point the **Logs folder** at a directory "
            "containing `trial_*.json` files, or upload some JSON logs from the "
            "sidebar."
        )
        st.stop()

    # -- Sidebar: scoring reference -------------------------------------
    auto_k0, cfg_name = autodetect_true_k0(directory)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Scoring")
    if cfg_name:
        st.sidebar.caption(f"true k₀ auto-detected from `{cfg_name}`")
    true_k0 = st.sidebar.number_input(
        "True k₀ (coeff of x)",
        value=float(auto_k0) if auto_k0 is not None else -4.761,
        step=0.001,
        format="%.4f",
        help="Ground-truth linear coefficient used to score submissions.",
    )

    # -- Sidebar: view options ------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("View options")
    opts = {
        "collapse_python": st.sidebar.checkbox("Collapse Python code", value=False),
        "collapse_output": st.sidebar.checkbox("Collapse long outputs", value=True),
        "expand_system": st.sidebar.checkbox("Expand system prompt", value=False),
        "show_raw": st.sidebar.checkbox("Show raw message JSON", value=False),
    }

    stats = [summarize_trial(t) for t in trials]

    page = st.sidebar.radio("Page", ["\U0001F4CA Overview", "\U0001F4AC Trial detail"])

    if page.endswith("Overview"):
        _render_overview(st, pd, trials, stats, sources, true_k0)
    else:
        _render_detail(st, trials, stats, sources, true_k0, opts)


def _gather_trials(directory, uploads, st):
    """Return (trials, sources) from the folder plus any uploads."""
    trials: list[dict] = []
    sources: list[str] = []

    for path in discover_trial_files(directory):
        try:
            for t in load_trial_file(path):
                trials.append(t)
                sources.append(os.path.basename(path))
        except Exception as e:  # noqa: BLE001
            st.sidebar.warning(f"Could not read {os.path.basename(path)}: {e}")

    for up in uploads or []:
        try:
            obj = json.loads(up.getvalue().decode("utf-8"))
            for t in normalize_to_trials(obj):
                trials.append(t)
                sources.append(up.name)
        except Exception as e:  # noqa: BLE001
            st.sidebar.warning(f"Could not parse upload {up.name}: {e}")

    return trials, sources


# --- Overview page ---------------------------------------------------------

def _render_overview(st, pd, trials, stats, sources, true_k0):
    st.markdown("## \U0001F4CA Benchmark overview")
    st.caption(
        "Reverse-engineering a hidden non-linear oscillator — each trial's "
        "job is to recover **k₀**, the coefficient of the linear `x` term."
    )

    n = len(stats)
    n_success = sum(1 for s in stats if s.status.lower() == "success")
    rel_errs = [relative_error(s.k0, true_k0) for s in stats]
    scored = [abs(e) for e in rel_errs if e is not None]
    mean_abs = sum(scored) / len(scored) if scored else None
    n_hall = sum(s.n_hallucinations for s in stats)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trials", n)
    c2.metric("Succeeded", f"{n_success}/{n}")
    c3.metric("Mean |rel. error|", f"{mean_abs*100:.2f}%" if mean_abs is not None else "—")
    c4.metric("Hallucinated outputs", n_hall,
              help="Tool responses the model fabricated in its own turn.")

    # Build the table.
    rows = []
    for s, src in zip(stats, sources):
        rel = relative_error(s.k0, true_k0)
        rows.append({
            "trial": s.trial_id,
            "source": src,
            "status": s.status,
            "turns": s.n_turns,
            "\U0001F9EA exp": s.n_experiments,
            "\U0001F40D py": s.n_python,
            "k₀ pred": None if s.k0 is None else round(s.k0, 4),
            "rel. err": None if rel is None else rel,
            "⚠ halluc.": s.n_hallucinations,
            "errors": s.n_errors,
            "chars": s.total_chars,
        })
    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "rel. err": st.column_config.NumberColumn("rel. err", format="%.4f"),
            "k₀ pred": st.column_config.NumberColumn("k₀ pred", format="%.4f"),
            "chars": st.column_config.NumberColumn("chars", format="%d"),
            "status": st.column_config.TextColumn("status"),
        },
    )

    # Signed-error chart.
    chart_rows = [
        {"trial": f"#{s.trial_id}", "rel_err": relative_error(s.k0, true_k0)}
        for s in stats
    ]
    chart_rows = [r for r in chart_rows if r["rel_err"] is not None]
    if chart_rows:
        try:
            import altair as alt

            cdf = pd.DataFrame(chart_rows)
            bars = (
                alt.Chart(cdf)
                .mark_bar()
                .encode(
                    x=alt.X("trial:N", title="Trial"),
                    y=alt.Y("rel_err:Q", title="Signed relative error"),
                    color=alt.condition(
                        alt.datum.rel_err > 0,
                        alt.value("#f59e0b"),
                        alt.value("#6366f1"),
                    ),
                    tooltip=["trial", alt.Tooltip("rel_err:Q", format=".4f")],
                )
                .properties(height=260)
            )
            zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
                color="#10b981", strokeDash=[4, 4]).encode(y="y:Q")
            st.altair_chart(bars + zero, width="stretch")
        except Exception:
            pass

    st.caption(
        "Signed relative error = (predicted k₀ − true k₀) / true k₀. "
        "The green dashed line is a perfect estimate."
    )


# --- Trial detail page -----------------------------------------------------

def _render_detail(st, trials, stats, sources, true_k0, opts):
    labels = [
        f"#{s.trial_id} — {src} ({s.status})"
        for s, src in zip(stats, sources)
    ]
    idx = st.selectbox(
        "Trial", range(len(trials)),
        format_func=lambda i: labels[i],
    )
    trial = trials[idx]
    s = stats[idx]

    _render_trial_header(st, s, true_k0)
    _render_flow_strip(st, trial)
    st.markdown("---")
    _render_conversation(st, trial, opts)


def _render_trial_header(st, s, true_k0):
    ok = s.status.lower() == "success"
    badge = (
        f'<span class="badge {"ok" if ok else "bad"}">'
        f'{"✔ success" if ok else "✖ " + html.escape(s.status)}</span>'
    )
    st.markdown(
        f"## Trial #{html.escape(str(s.trial_id))} {badge}",
        unsafe_allow_html=True,
    )

    rel = relative_error(s.k0, true_k0)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Agent turns", s.n_turns)
    c2.metric("Experiments", s.n_experiments)
    c3.metric("Python runs", s.n_python)
    c4.metric("Predicted k₀", f"{s.k0:.4f}" if s.k0 is not None else "—")
    c5.metric("Rel. error", f"{rel*100:.2f}%" if rel is not None else "—",
              help=f"vs true k₀ = {true_k0:.4f}")

    chips = []
    if s.n_hallucinations:
        chips.append(f'<span class="chip warn">⚠ {s.n_hallucinations} '
                     f'hallucinated tool output(s)</span>')
    if s.n_errors:
        chips.append(f'<span class="chip warn">⛔ {s.n_errors} host error(s)</span>')
    if s.submission is not None:
        chips.append(f'<span class="chip">\U0001F9EE {html.escape(equation_string(s.submission))}</span>')
    if chips:
        st.markdown('<div class="chiprow">' + " ".join(chips) + "</div>",
                    unsafe_allow_html=True)


def _render_flow_strip(st, trial):
    """A compact left-to-right glyph strip of the actions taken."""
    icons = {
        "experiment": "\U0001F9EA",
        "python": "\U0001F40D",
        "submission": "✅",
        "experiment_output": "\U0001F4E1",
        "python_output": "\U0001F5A5",
        "error": "⛔",
    }
    steps = []
    for msg in trial.get("chat_history", []):
        role = msg.get("role", "")
        if role == "system":
            continue
        for b in classify_message(role, msg.get("content", "") or ""):
            ic = icons.get(b.kind)
            if not ic:
                continue
            cls = "flow-item"
            if b.hallucinated:
                cls += " halluc"
            elif b.error:
                cls += " err"
            steps.append(f'<span class="{cls}" title="{b.kind}">{ic}</span>')
    if steps:
        strip = '<span class="flow-arrow">›</span>'.join(steps)
        st.markdown(f'<div class="flowstrip">{strip}</div>', unsafe_allow_html=True)


_AVATARS = {"system": "\U0001F9ED", "assistant": "\U0001F916", "user": "\U0001F5A5️"}
_ROLE_NAMES = {"system": "system", "assistant": "agent", "user": "environment"}


def _render_conversation(st, trial, opts):
    chat = trial.get("chat_history", []) or []
    for i, msg in enumerate(chat):
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""
        blocks = classify_message(role, content)

        # The huge, identical system prompt gets its own collapsible slot.
        if role == "system":
            with st.expander("\U0001F9ED System prompt (task instructions)",
                             expanded=opts["expand_system"]):
                st.markdown(_muted(content), unsafe_allow_html=True)
            continue

        avatar = _AVATARS.get(role, "❓")
        with st.chat_message(_ROLE_NAMES.get(role, role), avatar=avatar):
            for b in blocks:
                _render_block(st, b, opts)
            if opts["show_raw"]:
                with st.expander("raw message JSON"):
                    st.code(json.dumps(msg, indent=2), language="json")


def _render_block(st, b: Block, opts):
    if b.kind == "prose":
        st.markdown(b.body)

    elif b.kind == "experiment":
        _render_experiment(st, b)

    elif b.kind == "python":
        label = "\U0001F40D  Python — sandbox code"
        if opts["collapse_python"]:
            with st.expander(label):
                st.code(b.body, language="python")
        else:
            st.caption(label)
            st.code(b.body, language="python")

    elif b.kind == "python_output":
        _render_output(st, b, opts)

    elif b.kind == "submission":
        _render_submission(st, b)

    elif b.kind == "experiment_output":
        _render_experiment_output(st, b)

    elif b.kind == "error":
        st.markdown(
            f'<div class="card err"><b>⛔ Host error / retry</b><br>'
            f'{html.escape(b.body)}</div>',
            unsafe_allow_html=True,
        )

    elif b.kind == "system":
        st.markdown(_muted(b.body), unsafe_allow_html=True)

    else:
        st.write(b.body)


def _render_experiment(st, b: Block):
    s = experiment_summary(b.data)
    st.markdown('<div class="card exp"><b>\U0001F9EA Run experiment</b> '
                '<span class="muted">— sample 100 trajectories</span></div>',
                unsafe_allow_html=True)
    if s:
        cols = st.columns(5)
        cols[0].metric("t_start", f"{s['t_start']:.4g}")
        cols[1].metric("delta_t", f"{s['delta_t']:.4g}")
        cols[2].metric("steps", s["steps"])
        cols[3].metric("t_end", f"{s['t_end']:.4g}")
        cols[4].metric("points/traj", s["points"])
    else:
        st.code(b.body, language="json")


def _render_output(st, b: Block, opts):
    label = "\U0001F5A5️  Sandbox output"
    if b.hallucinated:
        st.markdown('<div class="card halluc"><b>⚠ Fabricated sandbox output</b>'
                    '<br><span class="muted">The model wrote this itself; the host '
                    'never ran it.</span></div>', unsafe_allow_html=True)
    elif b.error:
        label = "⛔  Sandbox output — error"
    body = b.body
    long = len(body) > 2000
    if opts["collapse_output"] and long:
        with st.expander(f"{label}  ({len(body):,} chars)"):
            st.code(body, language="text")
    else:
        st.caption(label)
        st.code(body, language="text")


def _render_submission(st, b: Block):
    eq = equation_string(b.data)
    rows = submission_terms(b.data)
    terms_html = ""
    if rows:
        items = "".join(
            f'<tr><td class="mono">{html.escape(str(t))}</td>'
            f'<td class="mono num">{html.escape(str(c))}</td></tr>'
            for t, c in rows
        )
        terms_html = f'<table class="terms">{items}</table>'
    st.markdown(
        f'<div class="card sub">'
        f'<div class="sub-h">✅ Submission</div>'
        f'<div class="eq">{html.escape(eq)}</div>'
        f'{terms_html}</div>',
        unsafe_allow_html=True,
    )


def _render_experiment_output(st, b: Block):
    if b.hallucinated:
        st.markdown(
            '<div class="card halluc"><b>⚠ Fabricated experiment output</b><br>'
            '<span class="muted">The model invented this environment response.</span>'
            '</div>', unsafe_allow_html=True)
    meta = None
    if isinstance(b.data, dict):
        meta = b.data.get("metadata", b.data)
    if isinstance(meta, dict):
        kv = "".join(
            f'<tr><td class="mono">{html.escape(str(k))}</td>'
            f'<td class="mono">{html.escape(str(v))}</td></tr>'
            for k, v in meta.items()
        )
        msg = ""
        if isinstance(b.data, dict) and b.data.get("message"):
            msg = f'<div class="muted">{html.escape(str(b.data["message"]))}</div>'
        st.markdown(
            f'<div class="card {"halluc" if b.hallucinated else "out"}">'
            f'<b>\U0001F4E1 Experiment output</b>{msg}'
            f'<table class="terms">{kv}</table></div>',
            unsafe_allow_html=True,
        )
    else:
        st.code(b.body, language="json")


def _muted(text: str) -> str:
    return f'<div class="muted small">{html.escape(text)}</div>'


def _inject_css(st) -> None:
    st.markdown(
        """
        <style>
        .badge { font-size:0.55em; padding:2px 8px; border-radius:10px;
                 vertical-align:middle; font-weight:600; }
        .badge.ok  { background:rgba(16,185,129,.18); color:#059669; }
        .badge.bad { background:rgba(239,68,68,.18);  color:#dc2626; }

        .card { border-radius:10px; padding:10px 14px; margin:4px 0 8px 0;
                border:1px solid rgba(128,128,128,.25); }
        .card.exp    { border-left:4px solid #6366f1; background:rgba(99,102,241,.07); }
        .card.out    { border-left:4px solid #0ea5e9; background:rgba(14,165,233,.07); }
        .card.sub    { border-left:4px solid #10b981; background:rgba(16,185,129,.09); }
        .card.err    { border-left:4px solid #ef4444; background:rgba(239,68,68,.08); }
        .card.halluc { border-left:4px solid #f59e0b; background:rgba(245,158,11,.10); }

        .sub-h { font-weight:700; color:#059669; margin-bottom:4px; }
        .eq { font-family:ui-monospace,Menlo,monospace; font-size:1.25rem;
              font-weight:700; margin:2px 0 8px 0; }

        table.terms { border-collapse:collapse; margin-top:4px; font-size:.85rem; }
        table.terms td { padding:2px 14px 2px 0; }
        .mono { font-family:ui-monospace,Menlo,monospace; }
        .num  { text-align:right; }

        .muted { color:rgba(130,130,140,.95); }
        .small { font-size:.8rem; white-space:pre-wrap;
                 max-height:340px; overflow:auto; }

        .chiprow { margin:6px 0 2px 0; }
        .chip { display:inline-block; font-size:.8rem; padding:3px 10px; margin:2px 6px 2px 0;
                border-radius:12px; background:rgba(99,102,241,.12);
                border:1px solid rgba(99,102,241,.3); font-family:ui-monospace,Menlo,monospace; }
        .chip.warn { background:rgba(245,158,11,.14); border-color:rgba(245,158,11,.4); }

        .flowstrip { font-size:1.35rem; line-height:2.1rem; margin:2px 0 6px 0; }
        .flow-item { padding:1px 3px; border-radius:6px; }
        .flow-item.halluc { background:rgba(245,158,11,.22); }
        .flow-item.err    { background:rgba(239,68,68,.20); }
        .flow-arrow { color:rgba(140,140,150,.6); margin:0 3px; font-size:1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    _run_app()
