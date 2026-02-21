# Rollback Guide — Composable Pipeline Deployment

## Before Deploying

Tag the known-good state before applying changes:

```bash
git tag pre-composable-pipeline
git push origin pre-composable-pipeline
```

Note the current commit hash as a secondary reference:

```bash
git log --oneline -1
```

## If Changes Regress Performance

### Option 1: Revert (preserves history, safe for shared branches)

```bash
# Undo the most recent commit with a new revert commit
git revert HEAD

# Or revert multiple commits (oldest..newest)
git revert abc1234..HEAD --no-commit
git commit -m "revert: roll back composable pipeline changes"

# Tag and push to trigger pipeline on reverted code
git tag v<next>
git push && git push --tags
```

### Option 2: Reset to safety tag (cleaner, rewrites history)

```bash
# Local: move branch back to the known-good commit
git reset --hard pre-composable-pipeline

# Remote: update to match (refuses if someone else pushed)
git push --force-with-lease
```

## Which Option to Use

| Scenario | Use |
|----------|-----|
| Shared branch, others may have pulled | **Revert** — safe, no history rewrite |
| Solo development branch, clean rollback needed | **Reset** — cleaner history |
| Unsure | **Revert** — always safe |

## After Rollback

1. Verify the pipeline runs on the reverted code
2. Investigate the regression on a separate branch
3. Remove the safety tag when no longer needed:
   ```bash
   git tag -d pre-composable-pipeline
   git push origin --delete pre-composable-pipeline
   ```
