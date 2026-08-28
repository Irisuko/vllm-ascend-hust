# Slash Commands

> The legacy `/e2e`, `/nightly`, and `/weekly` hardware commands have been retired. Pull request
> dataset validation is handled by the external fixed-machine service.

vLLM Ascend supports slash commands in pull request comments to trigger CI workflows. See the [Permission](#permission) section for who can trigger each command.

## Available Commands

### `/cherry-pick`

Cherry-pick a PR's commits onto a specified target branch and create a new PR. This is useful for backporting fixes to release branches.

**Usage:**

| Syntax | Description |
|---|---|
| `/cherry-pick <target_branch>` | Cherry-pick onto the specified branch |

**Examples:**

```text
# Cherry-pick to a release branch
/cherry-pick releases/v0.23.0

# Cherry-pick to main
/cherry-pick main
```

A new PR will be created with the title format `[Cherry-pick] <original_title> (from #<PR_NUMBER>)` and a body linking back to the original PR.

If the cherry-pick encounters merge conflicts, the command will report the failure and the cherry-pick must be done manually.

### `/revert`

Revert a merged PR by creating a new PR that reverses its changes. The revert targets the same base branch the original PR was merged into.

**Usage:**

| Syntax | Description |
|---|---|
| `/revert` | Revert this PR (no arguments needed) |

**Example:**

```text
/revert
```

A new PR will be created with the title format `[Revert] Revert "original_title" (#PR_NUMBER)` and a body linking back to the original PR and its merge commit.

Only merged PRs can be reverted. If the revert encounters merge conflicts (e.g., because the base branch has diverged significantly), the command will report the failure and the revert must be done manually.

### `/rerun`

Re-run all failed workflow runs on the current PR commit. Useful when CI jobs failed due to infrastructure issues.

**Examples:**

```text
# Re-run all failed CI workflows on this PR
/rerun
```

## Behavior

1. When you comment a slash command, a 👀 reaction is added to your comment to indicate it has been received
2. The corresponding CI workflow is triggered asynchronously
3. Upon completion, a 🎉 reaction and a summary comment are added

## Scope

| Command | PR comments | Issue comments |
|---|---|---|
| `/rerun` | ✅ | ❌ |
| `/cherry-pick` | ✅ | ❌ |
| `/revert` | ✅ | ❌ |

## Permission

| Command | Who can trigger |
|---|---|
| `/rerun` | PR author, or users with triage+ permission on the repository |
| `/cherry-pick` | PR author, or users with triage+ permission on the repository |
| `/revert` | PR author, or users with triage+ permission on the repository |

Permission is verified via the GitHub API (`repos/{owner}/{repo}/collaborators/{user}/permission`).
