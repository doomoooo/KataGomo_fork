# KataGo final migration

This directory is the control point for the final integration. It contains every
cross-cutting plan, inventory, reminder, environment lock, and generated audit
record used by the migration.

## Current state

- Integration base: official `lightvector/KataGo` `master` at
  `6a1fc5de9fc253723ac475a0683bf0b9d9b7bd19` (`v1.17.2`, fetched 2026-08-07).
- Integration branch/worktree: `final-migration` in
  `/workspace/katago-final-migration`.
- Phase 1 (environment): implemented and locally validated; see
  `inventory/environment-validation.md`.
- Phase 2 (plan scanner): blocked by the explicit 4090 scanner freeze gate.
- Active optimization worktrees are read-only inputs. Do not edit, merge, clean,
  or build in them from this migration.

## Fresh Ubuntu entry point

On Ubuntu 24.04, from the repository root:

```bash
./final-migration/environment/setup.sh all
```

The entry point installs and checks the system toolchain, creates an isolated
Python environment, resolves the latest upstream source commits, builds them
locally, runs compile smokes, and builds KataGo's CUDA backend. If a new NVIDIA
driver was installed, reboot and run the same command again.
Only the Ubuntu package step elevates itself with `sudo`; source checkouts,
virtual environments, and builds retain the invoking user's ownership.

There are two intentionally different acquisition paths:

- Development resolves current upstream HEAD for source-capable optimizer
  dependencies. A local Git bundle may seed the checkout, but a GitHub fetch
  establishes the latest commit. Binary/bootstrap packages use a local
  wheelhouse first and the configured domestic PyPI mirror second.
- Deployment consumes the compiled distribution bundle first and does not
  clone or rebuild optimizer dependencies.

GitHub access prints an explicit warning because connectivity may need a
proxy. Set `HTTPS_PROXY`/`https_proxy` before invoking the script when required.

After a successful build, create the artifact intended for future machines:

```bash
./final-migration/environment/setup.sh package
```

The bundle records resolved source revisions and hashes for every shipped wheel
and CUDA-backend binary. A target can use either `setup.sh deploy BUNDLE` from
the source tree or the bundle's standalone `installer/deploy-prebuilt.sh`.

For non-installing checks on an already configured host:

```bash
./final-migration/environment/setup.sh audit
./final-migration/environment/setup.sh verify
./final-migration/environment/setup.sh build
```

Runtime records are written under `final-migration/records/`; large build and
virtual-environment data stay in ignored directories at the repository root.

## Directory map

- `PLAN.md`: phase ordering, deliverables, and acceptance criteria.
- `FREEZE-GATES.md`: hard gates protecting active optimization sessions.
- `inventory/`: observations and asset relationships, never presumed frozen.
- `environment/`: reproducible bootstrap, locks, audits, and build verification.
- `archive/`: archive contract and optional local dependency cache.
- `records/`: generated reports from setup/audit/build runs.
