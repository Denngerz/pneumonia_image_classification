---
name: commit-workflow
description: Project procedure for committing, branching, pushing and opening Pull Requests in the pneumonia CNN comparison repo. Use whenever the user asks to commit, save changes to git, push, create a branch, or open/update a PR — and before running any git command that changes history.
---

# Commit & Pull Request workflow

This skill is written for any AI assistant (Claude Code, Codex, Cursor, Copilot, …). It mirrors the human guide
[COMMIT_GUIDE.md](../../../docs/COMMIT_GUIDE.md) — keep both consistent.

Only commit, push or open a PR when the user asked for it. If unsure, show the planned commit and ask.

## 1. Inspect the current state

```bash
git status
git branch --show-current
git diff --stat
```

- Understand every changed and untracked file before staging anything.
- If there are unrelated changes mixed together, propose splitting them into separate commits.

## 2. Make sure you are not on `main`

If the current branch is `main` (or `master`), create a branch first:

```bash
git checkout main
git pull --ff-only
git checkout -b <type>/<short-topic>
```

Branch naming: `<type>/<short-topic>`, all lowercase, words joined with `-`, where `<type>` is one of the commit types below
written in lowercase. Examples: `feat/resnet50-config`, `bugfix/data-split-seed`, `docs/project-foundation`, `exp/densenet121-run1`.

If the user is already on a feature branch for this task, stay on it.

## 3. Check for forbidden files

Never stage any of these (see also the "Never commit" list in [AGENTS.md](../../../AGENTS.md)):

| Pattern | Reason |
|---|---|
| Dataset folders / images (`chest_xray/`, `data/raw/`, bulk `*.jpeg`, `*.png` from the dataset) | Large, licensed separately, downloaded by each user |
| `*.pt`, `*.pth`, `*.ckpt`, `*.onnx`, `*.h5` | Model weights — too large |
| `kaggle.json`, `.env`, any file containing API keys or tokens | Secrets |
| `venv/`, `.venv/`, `__pycache__/`, `.ipynb_checkpoints/` | Local environment / caches |
| Any single file > ~10 MB | Ask the user first |

Checks:

```bash
git status --porcelain
git diff --cached --name-only
```

- If a forbidden file appears as untracked, suggest adding its pattern to `.gitignore` (as a separate `Chore:` commit if needed).
- If a training notebook (`.ipynb`) has large outputs, tell the user and suggest clearing outputs before committing.
- If a secret was **already pushed**, stop and tell the user to revoke/regenerate it; do not try to rewrite history yourself.

## 4. Stage specific files

Stage files explicitly by path. Do **not** use `git add -A` / `git add .` blindly.

```bash
git add path/to/file1 path/to/file2
git diff --cached
```

Review the staged diff once more before committing.

## 5. Write the commit message

Format — this is the team's required style:

```
Type: very short description

<optional body: why the change was made, details, results>
```

| Type | Use for |
|---|---|
| `Feat` | New functionality (model, metric, script) |
| `Bugfix` | Bug fix |
| `Refactor` | Restructuring without behavior change |
| `Docs` | Documentation only |
| `Exp` | Experiment configs or recorded results |
| `Data` | Data preparation code, split files |
| `Test` | Tests |
| `Chore` | Dependencies, tooling, `.gitignore`, config |

Rules:
- Type is capitalized exactly as in the table, followed by `: ` (colon + space). No scope in parentheses.
- Description: very short (a few words, ≤ 50 characters), imperative, lowercase start, no trailing period.
  ✅ `Bugfix: apply class weights to loss` — ❌ `fix(train): apply class weights to loss`, ❌ `Update`.
- Body is optional: explain *why*; for `Exp` commits include key metrics.
- One logical change per commit.
- Append any attribution trailer your tool/environment requires (e.g. `Co-Authored-By:`).

Commit using a heredoc or file so the multi-line message is preserved:

```bash
git commit -F- <<'EOF'
Feat: add DenseNet121 model

Loaded from timm with ImageNet weights; uses the shared classifier head.
EOF
```

If a pre-commit hook fails, fix the reported problem and create a **new** commit — never use `--no-verify`, never `--amend`
a commit that was already pushed.

## 6. Push

```bash
git push -u origin <branch-name>
```

Never `git push --force` / `--force-with-lease` to a shared branch and never push directly to `main`.

## 7. Open the Pull Request

Target `main`. Title in commit-message style. Body template:

```markdown
## What
Short description of the change.

## Why
Reason / related task.

## How it was tested
e.g. debug mode on CPU (N images, 1 epoch), or full Kaggle run (link).

## Results
Metrics table / plots, or "n/a".
```

With GitHub CLI available:

```bash
gh pr create --base main --title "<title>" --body-file <file>
```

Without `gh`, give the user the compare URL:
`https://github.com/Denngerz/pneumonia_image_classification/compare/main...<branch-name>`
and the filled-in description to paste.

Append any PR attribution line your tool/environment requires.

## 8. Report back

Tell the user: branch name, commit hash(es) and messages, what was pushed, PR link (or that it still needs to be opened),
and anything you deliberately left out (e.g. ignored files, notebook outputs).

## Never

- Commit or push to `main` directly
- Force-push, rebase or rewrite already-pushed history
- Skip hooks (`--no-verify`) or signing
- Commit data, weights or secrets
- Merge a PR unless the user explicitly asks
