# Sourced by the farm submitters. meta.json records the source commit, which
# identifies the code only if this checkout is committed, pushed and current.
# A source snapshot (not a Git checkout) is named by its commit and passes.
# ALLOW_UNSYNCED_SOURCE=1 overrides, for a deliberate one-off.
if [[ "${ALLOW_UNSYNCED_SOURCE:-0}" != 1 ]] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git fetch --quiet 2>/dev/null || true
    if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
        echo "refusing to submit: uncommitted changes in $PWD (commit and push locally, then pull here)" >&2
        exit 2
    fi
    if [[ -z "$(git branch --remotes --contains HEAD)" ]]; then
        echo "refusing to submit: $(git rev-parse --short HEAD) is on no remote branch (push it first)" >&2
        exit 2
    fi
    behind=$(git rev-list --count 'HEAD..@{upstream}' 2>/dev/null || echo 0)
    if (( behind > 0 )); then
        echo "refusing to submit: $PWD is $behind commits behind its remote branch (git pull --ff-only)" >&2
        exit 2
    fi
fi
