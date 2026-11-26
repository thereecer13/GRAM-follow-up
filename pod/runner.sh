#!/usr/bin/env bash
# Job runner for a rented GPU pod that cannot be reached by SSH.
#
# The pod's start command clones this repository and runs this script. From then on the
# only way to tell the pod what to do is to push to the branch: the runner pulls every
# 30 seconds and runs, in name order, every script in pod/jobs/ it has not finished yet.
#
#   * One job runs at a time. Its output goes to the container log and to
#     /workspace/pod_logs/<job>.log.
#   * A finished job is recorded in /workspace/pod_state/ under its name and content
#     hash, so it is not run again unless the file changes.
#   * If a job fails, the runner waits for the next push instead of running later jobs.
#   * If the pod restarts mid-job, the job is simply run again. The long jobs
#     (scripts/run_set.py) continue from their newest checkpoint.
#
# /workspace/share (the result tables and the logs) is served read-only on port 8000.
set -u

BRANCH="${GRAM_BRANCH:-claude/nifty-heisenberg-9kom7x}"
REPO=/workspace/GRAM-follow-up
STATE=/workspace/pod_state
LOGS=/workspace/pod_logs
SHARE=/workspace/share
mkdir -p "$STATE" "$LOGS" "$SHARE" "$REPO/results"
ln -sfn "$REPO/results" "$SHARE/results"
ln -sfn "$LOGS" "$SHARE/logs"
ln -sfn "$STATE" "$SHARE/state"

cd "$REPO"
(cd "$SHARE" && exec python -m http.server 8000 >/dev/null 2>&1) &

failed_at=""   # the commit at which a job failed; nothing runs until the branch moves on
while true; do
    git fetch -q origin "$BRANCH" && git reset -q --hard FETCH_HEAD
    head=$(git rev-parse HEAD)
    if [ "$head" != "$failed_at" ]; then
        for job in pod/jobs/*.sh; do
            name=$(basename "$job" .sh)
            marker="$STATE/$name.$(sha1sum "$job" | cut -c1-12)"
            [ -e "$marker.ok" ] && continue
            echo "=== [runner] $(date -u +%FT%TZ) starting $name at commit ${head:0:7}"
            bash "$job" 2>&1 | tee "$LOGS/$name.log"
            code=${PIPESTATUS[0]}
            echo "=== [runner] $(date -u +%FT%TZ) finished $name with exit code $code"
            if [ "$code" -eq 0 ]; then
                touch "$marker.ok"
            else
                echo "$code" > "$marker.failed"
                failed_at=$head
                break
            fi
        done
    fi
    date -u +%FT%TZ > "$STATE/heartbeat"
    sleep 30
done
