---
name: massh-commit
description: "Guided commit workflow for the MASSH repo. Use when: committing changes to a VarDyn or other MASSH branch, staging source changes, deciding what to gitignore, and pushing to origin."
---

<what-to-do>

Walk the user through a safe, deliberate commit to the MASSH repo by interviewing them one question at a time about scope, gitignore hygiene, and commit message, then executing the agreed steps.

Ask questions one at a time. If a question can be answered by inspecting the repo, do so instead of asking.

</what-to-do>

<steps>

## 1. Inspect current state

Run `git status` and `git diff --stat` to understand:
- Which tracked files are modified
- Which untracked files exist and their categories (source, config, notebooks, data, logs)

## 2. Resolve .gitignore hygiene first (before staging anything)

For each category of untracked files, check whether it belongs in `.gitignore`:

| Pattern | Rule |
|---------|------|
| `*.nc` | Always gitignore — NetCDF binary data, not diffable |
| `*.out` | Always gitignore — runtime log output |
| `nohup.out` | Always gitignore |
| `__pycache__/`, `*.pyc` | Already covered by existing .gitignore |
| `.ipynb_checkpoints/` | Already covered by existing .gitignore |

If any of these patterns are missing from `.gitignore`, add them before staging.

## 3. Determine what to commit

For each category of untracked files, ask the user:
- **Source files** (`mapping/src/*.py`, `mapping/models/*.py`): almost always yes
- **Tests** (`tests/*.py`): almost always yes
- **Config experiment files** (`config_*.py` at root): ask — they may be personal scratch configs
- **Notebooks** (`.ipynb`): ask — warn about embedded cell outputs bloating the repo
- **Scripts** (`run_*.sh`): ask
- **Docs** (`doc/`): ask

## 4. Agree on commit message

Summarise the changes from `git diff --stat` and propose a conventional-commit message:
`<type>: <short summary>`

Common types for MASSH: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`.

## 5. Execute

1. Add `*.nc` and `*.out` rules to `.gitignore` if missing
2. `git add` agreed files
3. Show `git status --short` for confirmation
4. `git commit -m "<agreed message>"`
5. `git push origin <current-branch>`

</steps>

<safety>

- Never `git push --force`
- Never amend already-pushed commits
- Always show `git status --short` after staging so the user can verify before committing

</safety>
