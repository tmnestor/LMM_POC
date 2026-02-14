# Minimise .git Repository Size

## 1. Check Current Size

```bash
du -sh .git
git rev-list --count HEAD
```

## 2. Garbage Collection (Safe, Non-Destructive)

Repacks loose objects and prunes unreachable ones. No history is lost.

```bash
git gc --aggressive --prune=now
```

Check size again with `du -sh .git` — this alone often cuts 30-50%.

## 3. Find Large Blobs in History

List the 20 largest objects ever committed, including deleted files still in history:

```bash
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk '/^blob/ {print $3, $4}' \
  | sort -rn \
  | head -20
```

Output is `<size_bytes> <file_path>`. Look for:

- Images/binaries accidentally committed
- Notebook files with embedded outputs (each save = new multi-MB blob)
- Data files, model weights, CSVs

## 4. Prevent Future Notebook Bloat (nbstripout)

Strips outputs from `.ipynb` files on `git add` so only source cells are committed.

```bash
pip install nbstripout
nbstripout --install --attributes .gitattributes
```

Verify:

```bash
git config --local --list | grep filter.nbstripout
cat .gitattributes
```

Commit `.gitattributes` so it applies for all contributors (they also need `nbstripout` installed).

## 5. Remove Large Files from History (git filter-repo)

Permanently rewrites history to remove specific files. **Requires force push.**

```bash
# Install
pip install git-filter-repo

# Remove a specific file from all history
git filter-repo --invert-paths --path path/to/large_file.png

# Remove all files matching a pattern
git filter-repo --invert-paths --path-glob '*.png'

# Remove files larger than a threshold
git filter-repo --strip-blobs-bigger-than 10M
```

After `filter-repo`, the remote is unset. Re-add and force push:

```bash
git remote add origin <url>
git push --force --all
git push --force --tags
```

**Warning:** All collaborators must re-clone after a history rewrite.

## 6. Shallow Clone (Fresh Start)

Clone only the latest state with no history:

```bash
git clone --depth 1 <url> repo-shallow
```

Or truncate an existing repo's history to the last N commits:

```bash
git fetch --depth 50
git gc --prune=now
```

## 7. Preventive .gitignore

Add common large-file patterns before they get committed:

```gitignore
# Data and models
*.h5
*.pt
*.onnx
*.bin
*.safetensors
evaluation_data/

# Images
*.png
*.jpg
*.jpeg
*.tiff

# Notebook checkpoints
.ipynb_checkpoints/
```

## Quick Reference

| Action | Destructive? | Force push? | Typical savings |
|--------|-------------|-------------|-----------------|
| `git gc --aggressive` | No | No | 30-50% |
| `nbstripout` | No | No | Prevents future bloat |
| `.gitignore` | No | No | Prevents future bloat |
| `git filter-repo` | Yes | Yes | Depends on blob size |
| Shallow clone | Yes | N/A (new clone) | Up to 90% |
