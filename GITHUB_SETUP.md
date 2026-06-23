# Pushing This Repo to GitHub

This project already has a full git history (`git log --oneline`, 30 commits).
Your GitHub repo is **<https://github.com/Aeontanvir/finetuning-llm>**.

## Push (your repo is empty, so this just works)

From inside this folder:

```bash
git remote add origin https://github.com/Aeontanvir/finetuning-llm.git
git branch -M main
git push -u origin main
```

If you see *"remote origin already exists"*:

```bash
git remote set-url origin https://github.com/Aeontanvir/finetuning-llm.git
git push -u origin main
```

## Authentication

When prompted for a password, GitHub needs a **Personal Access Token**, not your
account password:

1. github.com → Settings → Developer settings → Personal access tokens →
   Tokens (classic) → Generate new token.
2. Give it the `repo` scope.
3. Use the token as the password when `git push` asks.

(Or use SSH: `git remote set-url origin git@github.com:Aeontanvir/finetuning-llm.git`.)

## Done

Refresh <https://github.com/Aeontanvir/finetuning-llm> — all files and the commit
history will be there.

---

### If git isn't initialized (e.g. you only copied the loose files)

```bash
git init
git add .
git commit -m "chore: initial commit"
git branch -M main
git remote add origin https://github.com/Aeontanvir/finetuning-llm.git
git push -u origin main
```
