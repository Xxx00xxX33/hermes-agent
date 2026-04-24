#!/usr/bin/env bash
set -euo pipefail

TMUX_CONF="${TMUX_CONF:-$HOME/.tmux.conf}"

mkdir -p "$(dirname "$TMUX_CONF")"
touch "$TMUX_CONF"

python3 - "$TMUX_CONF" <<'PY'
import pathlib, sys

path = pathlib.Path(sys.argv[1])
text = path.read_text() if path.exists() else ""

legacy = """# Hermes local tmux usability tweaks
# Allow direct PageUp/PageDown scrollback in tmux panes without pressing the prefix first.
# - PageUp: enter copy mode and page up immediately.
# - PageDown: if already in copy mode, page down; otherwise pass the key through.
bind-key -n PPage if-shell -F '#{pane_in_mode}' 'send-keys -X page-up' 'copy-mode -eu'
bind-key -n NPage if-shell -F '#{pane_in_mode}' 'send-keys -X page-down' 'send-keys NPage'
"""

old_block = """# >>> Hermes tmux direct PageUp/PageDown >>>
# Allow direct PageUp/PageDown scrollback in tmux panes without pressing the prefix first.
# - PageUp: enter copy mode and page up immediately.
# - PageDown: if already in copy mode, page down; otherwise pass the key through.
bind-key -n PPage if-shell -F '#{pane_in_mode}' 'send-keys -X page-up' 'copy-mode -eu'
bind-key -n NPage if-shell -F '#{pane_in_mode}' 'send-keys -X page-down' 'send-keys NPage'
# <<< Hermes tmux direct PageUp/PageDown <<<
"""

scroll_block = r"""# >>> Hermes tmux direct scroll/history keys >>>
# Allow direct history navigation in tmux panes without pressing the prefix first.
# - PageUp: enter copy mode and page up immediately.
# - PageDown: if already in copy mode, page down; otherwise pass the key through.
# - Home: jump to the top of scrollback while browsing history; otherwise pass through.
# - End: jump to the bottom of scrollback and return to the live pane; otherwise pass through.
bind-key -n PPage if-shell -F '#{pane_in_mode}' 'send-keys -X page-up' 'copy-mode -eu'
bind-key -n NPage if-shell -F '#{pane_in_mode}' 'send-keys -X page-down' 'send-keys NPage'
bind-key -n Home if-shell -F '#{pane_in_mode}' 'send-keys -X history-top' 'send-keys Home'
bind-key -n End if-shell -F '#{pane_in_mode}' 'send-keys -X history-bottom \; send-keys -X cancel' 'send-keys End'
bind-key -T copy-mode Home send-keys -X history-top
bind-key -T copy-mode End send-keys -X history-bottom \; send-keys -X cancel
bind-key -T copy-mode-vi Home send-keys -X history-top
bind-key -T copy-mode-vi End send-keys -X history-bottom \; send-keys -X cancel
# <<< Hermes tmux direct scroll/history keys <<<
"""

title_block = r"""# >>> Hermes tmux task title status-right >>>
# Show only the current pane title in tmux's bottom-right status area.
# Hermes updates pane titles with the current prompt summary, so omit time/date.
set-option -g status-right '#{?pane_title,#{=80:pane_title},Hermes}'
set-option -g status-right-length 80
# <<< Hermes tmux task title status-right <<<
"""

blocks = [
    ("# >>> Hermes tmux direct scroll/history keys >>>", "# <<< Hermes tmux direct scroll/history keys <<<", scroll_block),
    ("# >>> Hermes tmux task title status-right >>>", "# <<< Hermes tmux task title status-right <<<", title_block),
]

if legacy in text:
    text = text.replace(legacy, "")

if old_block in text:
    text = text.replace(old_block, "")

for start, end, _block in blocks:
    if start in text and end in text:
        si = text.index(start)
        ei = text.index(end, si) + len(end)
        if ei < len(text) and text[ei:ei + 1] == "\n":
            ei += 1
        text = text[:si] + text[ei:]

text = text.rstrip("\n")
if text:
    text += "\n\n"
text += scroll_block + "\n\n" + title_block + "\n"
path.write_text(text)
PY

if command -v tmux >/dev/null 2>&1; then
  if tmux list-sessions >/dev/null 2>&1; then
    tmux source-file "$TMUX_CONF" >/dev/null 2>&1 || true
  fi
fi

echo "Ensured tmux scroll/history keys and task-title status-right in $TMUX_CONF"
