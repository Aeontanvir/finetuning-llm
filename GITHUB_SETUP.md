# Pushing This Repo to GitHub

This project already has a full git history (`git log --oneline`). To publish it:

## 1. Create an empty repo on GitHub

Go to <https://github.com/new>, name it `finetuning-llm-mastery`, and **do not**
add a README, .gitignore, or license (this repo already has them).

## 2. Connect and push

From inside this folder:

```bash
# Use 'main' as the branch name
git branch -M main

# Replace <your-username> with your GitHub username
git remote add origin https://github.com/<your-username>/finetuning-llm-mastery.git

git push -u origin main
```

If you use SSH instead of HTTPS:

```bash
git remote add origin git@github.com:<your-username>/finetuning-llm-mastery.git
git push -u origin main
```

## 3. Done

Refresh your GitHub repo page — all files and the commit history will be there.

---

### If git isn't initialized (e.g. you only copied the files)

```bash
git init
git add .
git commit -m "chore: initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/finetuning-llm-mastery.git
git push -u origin main
```
