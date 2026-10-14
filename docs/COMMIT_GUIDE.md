# How to Commit Changes (without the command line)

This guide explains how to save and share your work in this project using **GitHub Desktop**, a free app with buttons
instead of terminal commands. No previous git experience is needed.

Repository: <https://github.com/Denngerz/pneumonia_image_classification>

> Each step also shows the matching terminal command in a grey box *for reference only* — you don't need to use it.

---

## 0. Words you will see

| Word | Meaning |
|---|---|
| **Repository (repo)** | The project folder, tracked by git, stored on GitHub |
| **Clone** | Download a copy of the repo to your computer (done once) |
| **Branch** | Your own parallel copy of the project where you can work without disturbing others |
| **`main`** | The main, shared branch. Nobody works on it directly |
| **Commit** | A saved snapshot of your changes, with a short message describing them |
| **Push** | Upload your commits from your computer to GitHub |
| **Pull / Fetch** | Download other people's new commits from GitHub |
| **Pull Request (PR)** | A request to merge your branch into `main`; teammates review it first |
| **Merge conflict** | Two people changed the same lines; git asks you which version to keep |

**The golden rule:** never work on `main`. Always create your own branch → commit → push → open a Pull Request.

---

## 1. One-time setup

1. Create a free account at <https://github.com> (if you don't have one) and ask the repo owner to add you as a collaborator.
   Accept the invitation e-mail.
2. Download and install **GitHub Desktop**: <https://desktop.github.com>.
3. Open GitHub Desktop → **File → Options → Accounts** (macOS: **GitHub Desktop → Settings → Accounts**) → **Sign in** to GitHub.com.
4. In **Options → Git**, check that your name and e-mail are filled in (they will appear on your commits).

## 2. Clone the repository (one time)

1. **File → Clone repository…**
2. Open the **URL** tab, paste `https://github.com/Denngerz/pneumonia_image_classification`.
3. Choose a local folder (e.g. `Documents\VGTU\NeuralNetworks`) → **Clone**.

```
git clone https://github.com/Denngerz/pneumonia_image_classification.git
```

---

## 3. Every time you start working

### 3.1 Get the latest version of `main`

1. In the top bar, click **Current branch** and select **`main`**.
2. Click **Fetch origin**, then **Pull origin** if it appears.

```
git checkout main
git pull
```

### 3.2 Create your own branch

1. **Current branch → New branch**.
2. Name it `<type>/<short-topic>` — lowercase type (see the table in section 5), words separated by `-`. Examples:
   - `feat/resnet50-config`
   - `bugfix/data-split-seed`
   - `docs/update-plan`
   - `exp/densenet121-run1`
3. Make sure **Create branch based on… `main`** is selected → **Create branch**.

```
git checkout -b feat/resnet50-config
```

> Already have a branch for this task? Just select it in **Current branch** instead of creating a new one.

---

## 4. Make your changes

Edit files as usual (VS Code, PyCharm, Jupyter, Kaggle download, etc.). GitHub Desktop automatically shows every changed file
in the **Changes** tab on the left.

### What must NOT be committed

Before committing, look at the list of changed files. **Untick** (or never add) any of these:

- ❌ The dataset or X-ray images (`chest_xray/`, folders full of `.jpeg`)
- ❌ Model weights / checkpoints: `*.pt`, `*.pth`, `*.ckpt`, `*.onnx`
- ❌ Secrets: `kaggle.json`, API keys, passwords, `.env`
- ❌ Virtual environment folders: `venv/`, `.venv/`
- ❌ Huge files (> ~10 MB) — ask the team first
- ⚠️ Notebooks: before committing a training notebook, clear its outputs (Jupyter: **Edit → Clear All Outputs**;
  Kaggle: **Run → Clear All Outputs**). Results/visualization notebooks may keep their charts.

If something on this list keeps showing up, right-click it → **Ignore file** (or tell the team so it gets added to `.gitignore`).

---

## 5. Commit

1. In the **Changes** tab, tick only the files that belong to this change. Click a file to see exactly what changed
   (green = added lines, red = removed lines).
2. At the bottom left, fill in **Summary** (required) and optionally **Description**.
3. Click **Commit to `<your-branch>`**. Check that the button does **not** say `main`!

```
git add <file1> <file2>
git commit -m "Feat: add ResNet50 config"
```

### Commit message format

```
Type: very short description
```

The type starts with a capital letter, followed by a colon and a space.

| Type | Use it for | Example |
|---|---|---|
| `Feat` | New functionality | `Feat: add EfficientNet-B0` |
| `Bugfix` | Bug fix | `Bugfix: split data by patient` |
| `Refactor` | Code change that doesn't change behavior | `Refactor: extract metrics function` |
| `Docs` | Documentation only | `Docs: explain Kaggle setup` |
| `Exp` | Experiment configs or results | `Exp: add DenseNet121 results` |
| `Data` | Data preparation scripts, split files | `Data: generate train/val split` |
| `Test` | Tests | `Test: check split leakage` |
| `Chore` | Dependencies, config, tooling | `Chore: add timm to requirements` |

Rules for the summary:
- **Very short**: a few words, max ~50 characters, no period at the end.
- Say **what** the change does: ✅ `Bugfix: apply class weights to loss` — ❌ `Bugfix: fixed stuff`, ❌ `update`.
- Optional: use the **Description** field for *why* and details (results, links).

Make **small commits** — one logical change each. Several commits per branch are fine.

---

## 6. Push (upload) your branch

Click **Publish branch** (first time) or **Push origin** (later) in the top bar.

```
git push -u origin feat/resnet50-config
```

---

## 7. Open a Pull Request

1. After pushing, GitHub Desktop shows **Create Pull Request** (or **Branch → Create pull request**). It opens GitHub in your browser.
2. Check: **base: `main`** ← **compare: `<your-branch>`**.
3. Title: same style as a commit message.
4. Description — copy and fill in:

   ```
   ## What
   Short description of the change.

   ## Why
   Reason / related task.

   ## How it was tested
   e.g. "Ran debug mode on CPU (50 images, 1 epoch)" or "Full run on Kaggle, notebook link: …"

   ## Results (if any)
   Metrics table, plots, or "n/a".
   ```
5. On the right, add a teammate as **Reviewer** → **Create pull request**.
6. When the reviewer approves, click **Merge pull request** → **Confirm merge** → **Delete branch** (on GitHub).
7. Back in GitHub Desktop: switch to `main` → **Fetch origin / Pull origin**.

If the reviewer asks for changes: stay on the same branch, edit, commit, push again — the PR updates automatically.

---

## 8. Keeping your branch up to date

If `main` got new commits while you were working:

1. Make sure all your changes are committed.
2. **Branch → Update from `main`** (older versions: **Branch → Merge into current branch… → `main`**).
3. **Push origin**.

```
git fetch
git merge origin/main
git push
```

## 9. Merge conflicts (don't panic)

If GitHub Desktop says there are conflicts:

1. Click **Open in Visual Studio Code** (or your editor) for each conflicted file.
2. You'll see blocks like:
   ```
   <<<<<<< HEAD
   your version
   =======
   their version
   >>>>>>> main
   ```
3. Keep the correct lines, delete the `<<<<<<<`, `=======`, `>>>>>>>` markers, save.
4. Back in GitHub Desktop → **Continue merge** → **Push origin**.

Notebook (`.ipynb`) conflicts are hard to fix by hand — ask a teammate, or keep one version and redo your small change.

---

## 10. FAQ

**I committed to `main` by mistake (not pushed yet).**
**Branch → New branch** — your commit moves with you to the new branch. Then switch back to `main` and ask a teammate to help reset it, or leave it and tell the team.

**I committed a file I shouldn't have (not pushed yet).**
**History** tab → right-click the last commit → **Undo commit**. The changes return to the Changes tab; untick the file and commit again.

**I pushed a secret (e.g. `kaggle.json`).**
Tell the team immediately and **regenerate the key** on Kaggle (Account → API → Create New Token). Deleting the file in a new commit is not enough — it stays in history.

**I want to throw away my local changes to a file.**
Right-click the file in **Changes** → **Discard changes…** (this cannot be undone).

**Where do I see what others did?**
**History** tab, or the **Commits** / **Pull requests** pages on GitHub.

---

AI assistants follow the same rules, described in [.claude/skills/commit-workflow/SKILL.md](../.claude/skills/commit-workflow/SKILL.md).
