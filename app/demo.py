"""Phase 6: Gradio demo showing the live investigation trace.

Runs the ReAct agent and the single-shot baseline head-to-head on the same
claim, since the whole point of this project is the comparison: does the
agent's autonomy show up as genuinely different (better) evidence-gathering,
or does it just spend more steps and tokens to land on the same place the
baseline reached in one shot? The demo streams the agent's trace turn by
turn as it runs rather than only showing a final verdict box, so a viewer
can watch that evidence-seeking happen (or not) live.
"""

import os

import gradio as gr

from misinfo_agent import agent, baseline, trace

MAX_CLAIM_CHARS = 500
DEFAULT_MAX_STEPS = 10

EXAMPLE_CLAIMS = [
    "The measles vaccine causes autism.",
    "NASA's Voyager 1 probe has left the solar system.",
    "Drinking coffee stunts a child's growth.",
]

VERDICT_EMOJI = {"true": "✅", "false": "❌", "mixed": "🟡"}

MISSING_KEYS = [
    name
    for name in ("ANTHROPIC_API_KEY", "TAVILY_API_KEY")
    if not os.environ.get(name)
]


def _verdict_badge(investigation: trace.Investigation) -> str:
    if investigation.verdict is None:
        return "⏳ **No verdict** — stopped without submitting one."
    emoji = VERDICT_EMOJI.get(investigation.verdict, "")
    return f"{emoji} **{investigation.verdict.upper()}** — confidence {investigation.confidence:.2f}"


def _step_markdown(step: trace.TraceStep) -> str:
    observation = step.observation
    if len(observation) > 600:
        observation = observation[:600] + "…"
    args = ", ".join(f"{k}={v!r}" for k, v in step.action_input.items())
    return (
        f"**Step {step.step_number} — `{step.action}`**\n\n"
        f"{step.thought}\n\n"
        f"> `{step.action}({args})`\n\n"
        f"```\n{observation}\n```"
    )


def _trace_markdown(investigation: trace.Investigation | None) -> str:
    if investigation is None or not investigation.steps:
        return "_Waiting for the first step…_"
    return "\n\n---\n\n".join(_step_markdown(s) for s in investigation.steps)


def _summary_markdown(investigation: trace.Investigation | None) -> str:
    if investigation is None:
        return "_Not run yet._"
    parts = [_verdict_badge(investigation)]
    if investigation.reasoning:
        parts.append(investigation.reasoning)
    parts.append(
        f"*{len(investigation.steps)} step(s) · "
        f"{investigation.total_input_tokens} in / {investigation.total_output_tokens} out tokens · "
        f"stopped: {investigation.stop_reason}*"
    )
    return "\n\n".join(parts)


def _evidence_rows(investigation: trace.Investigation | None) -> list[list]:
    if investigation is None:
        return []
    return [
        [ev.step_number, ev.stance, ev.source_credibility, ev.source_bias, ev.url, ev.quote or ""]
        for ev in investigation.evidence
    ]


def investigate(claim: str, max_steps: int):
    claim = (claim or "").strip()
    if not claim:
        yield (
            "_Enter a claim above and press Investigate._",
            "_Not run yet._",
            [],
            "_Not run yet._",
            "_Not run yet._",
        )
        return
    claim = claim[:MAX_CLAIM_CHARS]

    if MISSING_KEYS:
        msg = f"⚠️ Missing environment variable(s): {', '.join(MISSING_KEYS)}. Set them and restart."
        yield msg, msg, [], msg, msg
        return

    # Baseline is one search + one forced tool call — cheap enough to just
    # run to completion before the agent's first yield.
    try:
        baseline_investigation = baseline.run_baseline(claim)
    except Exception as exc:
        baseline_investigation = None
        baseline_trace_md = "_Not run yet._"
        baseline_summary_md = f"⚠️ Baseline failed: {exc}"
    else:
        baseline_trace_md = _trace_markdown(baseline_investigation)
        baseline_summary_md = _summary_markdown(baseline_investigation)

    yield "_Starting investigation…_", "_Not run yet._", [], baseline_trace_md, baseline_summary_md

    investigation = None
    try:
        for investigation in agent.run_investigation_stream(claim, max_steps=int(max_steps)):
            yield (
                _trace_markdown(investigation),
                _summary_markdown(investigation),
                _evidence_rows(investigation),
                baseline_trace_md,
                baseline_summary_md,
            )
    except Exception as exc:
        yield (
            _trace_markdown(investigation),
            f"⚠️ Investigation failed: {exc}",
            _evidence_rows(investigation),
            baseline_trace_md,
            baseline_summary_md,
        )


with gr.Blocks(title="Misinformation Investigation Agent") as demo:
    gr.Markdown(
        "# Misinformation Investigation Agent\n"
        "A ReAct agent investigates a claim step by step — deciding what to "
        "search, which sources to check, and when it has enough evidence — "
        "against a single-shot RAG baseline given the same tools. Watch the "
        "agent's trace build live below; a passing agent is expected to seek "
        "out at least one piece of *disconfirming* evidence before it's "
        "allowed to submit a verdict."
    )
    if MISSING_KEYS:
        gr.Markdown(
            f"⚠️ **Missing environment variable(s): {', '.join(MISSING_KEYS)}.** "
            "Copy `.env.example` to `.env`, add your keys, and restart."
        )

    claim_box = gr.Textbox(
        label="Claim to investigate",
        placeholder="e.g. The measles vaccine causes autism.",
        max_lines=3,
    )
    with gr.Accordion("Advanced settings", open=False):
        max_steps_slider = gr.Slider(
            minimum=5,
            maximum=15,
            value=DEFAULT_MAX_STEPS,
            step=1,
            label="Max agent steps (safety cap)",
        )
    run_button = gr.Button("Investigate", variant="primary")
    gr.Examples(examples=EXAMPLE_CLAIMS, inputs=claim_box)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Agent (ReAct loop)")
            agent_verdict_md = gr.Markdown("_Not run yet._")
            agent_trace_md = gr.Markdown("_Waiting for the first step…_")
            agent_evidence_df = gr.Dataframe(
                headers=["step", "stance", "credibility", "bias", "url", "quote"],
                label="Evidence gathered",
                value=[],
            )
        with gr.Column():
            gr.Markdown("### Baseline (single-shot RAG)")
            baseline_verdict_md = gr.Markdown("_Not run yet._")
            baseline_trace_md = gr.Markdown("_Not run yet._")

    run_button.click(
        fn=investigate,
        inputs=[claim_box, max_steps_slider],
        outputs=[agent_trace_md, agent_verdict_md, agent_evidence_df, baseline_trace_md, baseline_verdict_md],
    )
    claim_box.submit(
        fn=investigate,
        inputs=[claim_box, max_steps_slider],
        outputs=[agent_trace_md, agent_verdict_md, agent_evidence_df, baseline_trace_md, baseline_verdict_md],
    )

demo.queue()


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
