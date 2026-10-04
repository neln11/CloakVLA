"""Terminal-only styling for LIBERO evaluation output."""

import os


RESET = "\033[0m"
BOLD = "1"
RED = "31"
GREEN = "32"
YELLOW = "33"
BLUE = "34"
MAGENTA = "35"
CYAN = "36"


def terminal_color_enabled() -> bool:
    """Keep colors under tee; an explicit VLA_EVAL_COLOR overrides NO_COLOR."""
    explicit_value = os.environ.get("VLA_EVAL_COLOR")
    if explicit_value is None and "NO_COLOR" in os.environ:
        return False
    value = (explicit_value if explicit_value is not None else "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def style_terminal(text: str, color: str, *, bold: bool = False) -> str:
    if not terminal_color_enabled():
        return text
    codes = [BOLD, color] if bold else [color]
    return f"\033[{';'.join(codes)}m{text}{RESET}"


def colorize_eval_message(message: str) -> str:
    """Highlight important dual-evaluation messages while leaving routine output plain."""
    leading_newlines = message[: len(message) - len(message.lstrip("\n"))]
    text = message[len(leading_newlines) :]
    lowered = text.lower()

    if "episode error" in lowered or "[error]" in lowered:
        styled = style_terminal(text, RED, bold=True)
    elif "success: false" in lowered:
        styled = style_terminal(text, RED, bold=True)
    elif "success: true" in lowered:
        styled = style_terminal(text, GREEN, bold=True)
    elif text.startswith("====="):
        if "metric 3" in lowered or "trigger" in lowered:
            styled = style_terminal(text, MAGENTA, bold=True)
        elif "metric 2" in lowered:
            styled = style_terminal(text, BLUE, bold=True)
        else:
            styled = style_terminal(text, CYAN, bold=True)
    elif "[asr_t]" in lowered or "asr_u" in lowered:
        styled = style_terminal(text, MAGENTA, bold=True)
    elif "[trigger]" in lowered:
        styled = style_terminal(text, YELLOW, bold=True)
    elif text.startswith("[POISON-SOURCE-W]"):
        styled = style_terminal(text, MAGENTA)
    elif text.startswith("[POISON-SOURCE-WO]"):
        styled = style_terminal(text, CYAN)
    elif text.startswith("[POISON-MODEL]"):
        styled = style_terminal(text, BLUE)
    elif text.startswith("[CLEAN-MODEL]"):
        styled = style_terminal(text, CYAN)
    elif "success rate:" in lowered or "# successes:" in lowered:
        styled = style_terminal(text, GREEN, bold=True)
    elif (
        "saved to:" in lowered
        or "logging to local log file" in lowered
        or "loading poison model checkpoint" in lowered
    ):
        styled = style_terminal(text, CYAN)
    elif "skipping" in lowered or "no manual labels" in lowered or "[warn]" in lowered:
        styled = style_terminal(text, YELLOW)
    else:
        styled = text

    return leading_newlines + styled


def colorize_rollout_message(message: str) -> str:
    return style_terminal(message, CYAN, bold=True)
