# Local Hermes customization maintenance

This directory contains the canonical local customization bundle for the local Hermes maintenance branch.

The active bundle is defined only by `patches/manifest.yaml`.
Do not infer active coverage from stray `patches/*.patch` files, archived patch variants, or memory.

## Update-readiness rule

Before any AI or human updates Hermes against newer upstream code, run the governance gates and then run the clean-upstream rehearsal:

```bash
cd ~/.hermes/hermes-agent
python3 -m patches.local_customization_manifest --repo-root . verify --strict
python3 -m patches.local_customization_manifest --repo-root . verify-ahead
patches/rehearse-upstream-update.sh
```

This is mandatory.

`verify --strict` and `verify-ahead` prove that the current local customization surface is still covered by the manifest-listed canonical patch bundle.
They do not prove that the bundle can still be replayed onto the latest upstream tree.

The rehearsal helper is the upgrade-readiness gate.
If rehearsal reports any manifest-listed patch as `conflict`, the upstream update must stop until that patch is aligned/refreshed or deliberately retired.
Do not continue the rebase/merge/update while a required local customization still conflicts.

## Meaning of patch states

The rehearsal/apply helpers classify each manifest-listed patch as one of these states:

- `already-applied`: the target tree already contains the effect of the patch, so replay is not needed there.
- `applicable`: the patch still applies cleanly to the target tree.
- `conflict`: the patch no longer applies cleanly to the target tree and cannot be trusted to survive the upstream update automatically.

## What `conflict` means here

A conflict is an upgrade-carry-forward problem, not necessarily a current runtime failure.
It usually means upstream changed the same code region, moved the logic, renamed symbols, split functions, or otherwise changed the surrounding context enough that `git apply --check` can no longer replay the patch safely.

That means all of the following can be true at once:

- the local Hermes checkout still works today,
- the local customized behavior is still present today,
- but the durable patch artifact can no longer be replayed onto the latest clean upstream checkout.

If you ignore that signal and continue the upstream update anyway, the most likely failure mode is silent customization loss: the update completes, but one or more local behaviors no longer survive the upgrade.

## Required response when rehearsal finds conflicts

When `patches/rehearse-upstream-update.sh` reports `conflict` for any active manifest-listed patch:

1. Stop the upstream update flow.
2. Inspect the failing files and hunk context printed by the helper.
3. Refresh or realign the conflicting patch against the current upstream structure.
4. Re-run:
   - `python3 -m patches.local_customization_manifest --repo-root . verify --strict`
   - `python3 -m patches.local_customization_manifest --repo-root . verify-ahead`
   - `patches/apply-hermes-local-customizations.sh --status --no-tests`
   - `patches/rehearse-upstream-update.sh`
5. Continue the actual upstream merge/rebase/update only after the needed patch is no longer reported as `conflict`.

If upstream already absorbed the behavior, convert that patch from "conflict" into an intentional retirement or absorbed state rather than forcing stale hunks forward.
If upstream changed the architecture, adapt the customization to the new structure and regenerate the canonical patch artifact.

## Base-ref expectations

Inside the main local maintenance repo, the canonical baseline is `upstream/main`.
In a fresh upstream clone or overlay-CI rehearsal, the equivalent baseline is usually `origin/main`.
The helper scripts resolve this automatically when possible, but future maintainers should still interpret results with that distinction in mind.

## Governance notes

## Hotspot note: fast tool completion and delegated subtask progress on the primary status bar

`cli.py` is a hotspot file with several local customizations layered onto the tool-progress path.
Keep the `statusbar-fast-tool-refresh` patch separate from transcript-persistence and tmux-title patches.
Its scope is intentionally narrow but two-part:

1. on `tool.completed`, force `_invalidate(min_interval=0.0)` so very fast final tools (especially `todo`) cannot leave the footer stuck on stale progress like `(3/4)` or `4/5`;
2. when delegated child work exists, show `delegate_task` subtask progress at the end of status-bar line 1 without replacing the parent todo progress slot.

That second rule matters: parent todo progress and delegated subtask progress are not the same signal.
The parent `(x/y)` progress must remain intact, while delegated subtask progress such as `子任务：2/5` is appended at the far right of line 1.
Do not repurpose the parent todo-progress slot for delegated subtasks.

When refreshing this patch, validate both the focused regressions (`TestToolProgressRefresh`, delegated subtask lifecycle/placement coverage) and the surrounding `tests/cli/test_cli_status_bar.py` suite before folding it into any broader CLI artifact.

- `patches/manifest.yaml` is the single source of truth for active patch membership and order.
- `patches/archive/` is historical only.
- Governance code and helper scripts under `patches/` are themselves part of the local customization surface and must remain covered by the canonical bundle.
- Governance-green is not the same thing as upgrade-ready-green; rehearsal is the authoritative check for latest-upstream applicability.
