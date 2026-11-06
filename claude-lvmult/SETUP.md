# Setup

```bash
git init librivox-mult && cd librivox-mult    # this folder becomes the repo (/work)
cp .env.example .env && $EDITOR .env           # UID/GID = `id -u`/`id -g`, paths, key, GPUs
docker compose build                           # worker builds Kaldi: expect ~30-60 min

# models for the worker / LLM (the only containers with internet are claude + fetch)
docker compose run --rm fetch download facebook/wav2vec2-large-960h-lv60-self   # or yours
docker compose run --rm fetch download speechbrain/spkrec-ecapa-voxceleb
docker compose run --rm fetch download Qwen/Qwen3-8B

docker compose up -d claude worker
docker compose --profile llm up -d llm         # when Task 2 needs it

docker compose exec claude tmux new -A -s main
# inside:
claude --dangerously-skip-permissions          # acceptable here: firewall + ro data mounts
```

A first prompt that works with the CLAUDE.md:

> Do stage 0 and the existing-code review. Survey /data, write docs/inputs.md and
> /out/inventory.jsonl, then review /src-projects and write docs/existing-code.md.
> Stop and summarise.

## Notes

- **Spend:** use a Console workspace with a spend limit for `ANTHROPIC_API_KEY`.
- **Detach/reattach:** `Ctrl-b d`, then `docker compose exec claude tmux attach`.
  `/home/agent/.claude` is a volume, so `claude --resume` works across restarts.
- **Firewall:** domains are resolved to IPs at start-up. If something on the list
  stops working, restart the `claude` container. The list file is the repo copy
  (`docker/claude/allowed-domains.txt`), so the agent could edit it, with effect at
  the next restart. Set `FIREWALL_DOMAINS` in `.env` to a path outside the repo
  to rule that out.
- **Worker deps:** the worker has no internet. Claude will add to
  `docker/worker/requirements.txt` and ask you to `docker compose build worker`.
- **GPU split:** `WORKER_GPUS` / `LLM_GPUS`. On a single card, give both `0`
  and lower `LLM_GPU_UTIL`, or stop the LLM during wav2vec/VibeVoice jobs.
- **Escaping the sandbox:** `ssh worker` gives the agent the worker's
  permissions. The worker has no internet, but can write `/work` and `/out`.
  Only `/data` and `/src-projects` are protected (read-only).
