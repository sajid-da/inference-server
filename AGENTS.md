# Agent rules (read first, every session)

## Permissions

Full permission to create, edit, delete files in this repo, install dependencies,
run tests/linters/Docker, create branches, commit. Ask only for secrets/API keys,
anything that costs money, or destructive actions outside this repo.

## Zero-budget rule

Everything must run for free. Use free tiers and local CPU open-source models.
Tests never call real APIs: use deterministic fakes. Real providers only via env
vars. Record unknown free-tier limits in `docs/DECISIONS.md`.

## Token discipline

- Read only needed files with targeted commands.
- Run targeted tests first and the full suite once per milestone.
- State lives in the tracking files; no unrequested dependencies or refactors.
- Reply with what changed, test result, and next step in at most five lines.

## Tracking files

Update `docs/TODO.md`, `docs/STATUS.md`, `docs/DECISIONS.md`,
`docs/METRICS.md`, and `docs/LEARN.md` at every milestone end.

## Engineering and security

Use typed Python, ruff, mypy, structured logging, and environment-based config.
Use validated inputs; do not log secrets or sensitive request data. Add public
endpoint rate limits when those endpoints exist. Justify dependencies in
`docs/DECISIONS.md`. A feature is done only with a passing test.

## Milestone loop

Read STATUS and TODO, create a short-lived branch, implement the smallest slice,
test/lint, update docs, commit, and stop. Begin the next milestone only on
`continue`.
