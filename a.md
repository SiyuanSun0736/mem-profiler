# Repeat Timing 采集与回传命令

我先核对一下 data_quality_audit.json 里的候选字段路径，这样给你的命令就能直接复制执行，而不是手工抄程序名。

Searched for regex `repeat_timing_candidates|missing_strict_baseline_programs|action_bucket|o2_o3_programs` (`**/train_set/data_quality_audit.json`), 40 results

Read [](file:///home/ssy/mem-profiler/train_set/data_quality_audit.json#730-730), lines 730 to 860

我再补一眼 `missing_strict_baseline_programs` 的 JSON 结构，确保给你的筛选命令一次就能跑通。

Read [](file:///home/ssy/mem-profiler/train_set/data_quality_audit.json#3663-3663), lines 3663 to 3726

下面按“测试机采 timing -> 传 sidecar -> 训练机重建时间评估”给你一套可以直接复制的命令。

前提假设：

1. 两台机器上的仓库路径都用 mem-profiler。
2. 这次只补 P2，不重跑 BCC collect。
3. 要用到的脚本是 collect_repeat_timing.py、build_time_score_table.py、evaluate_score_vs_time.py。

## 0. 先做一个最小 smoke test

先只验证脚本会不会再因为 output_dir 权限失败：

```bash
cd /home/ssy/mem-profiler
source .venv/bin/activate

.venv/bin/python scripts/collect_repeat_timing.py \
  --program Bullet \
  --variant O0 \
  --limit 1 \
  --dry-run
```

你现在应该看到类似输出：

```text
[info] [1/1] O0/Bullet -> data/llvm_test_suite/repeat_timing_sidecars/bcc/O0/Bullet_.../repeat_timing.json
       output_dir 不可写，已回退到 data/llvm_test_suite/repeat_timing_sidecars
```

这一步的意义是先验证：

1. 脚本已经识别到旧的 `bcc/...` 运行目录不可写。
2. sidecar 会自动写到镜像目录 `data/llvm_test_suite/repeat_timing_sidecars/...`。

如果你还想做一次真实写文件的最小测试，可以只跑 1 次正式计时：

```bash
.venv/bin/python scripts/collect_repeat_timing.py \
  --program Bullet \
  --variant O0 \
  --limit 1 \
  --warmup-count 0 \
  --repeat-count 1 \
  --overwrite
```

然后确认 sidecar 已经写出来：

```bash
find data/llvm_test_suite/repeat_timing_sidecars -path '*Bullet*/repeat_timing.json' -print
```

## 1. 在测试机上跑 repeat timing

先进入仓库并激活环境：

```bash
cd /home/ssy/mem-profiler
source .venv/bin/activate
```

先从 data_quality_audit.json 里自动取出两类程序：

1. `repeat_timing_candidates`
2. `missing_strict_baseline_programs`

```bash
mapfile -t PROGRAMS < <(.venv/bin/python - <<'PY'
import json
from pathlib import Path

audit = json.loads(Path("train_set/data_quality_audit.json").read_text())

programs = {
    row["program"]
    for row in audit["pair_difficulty"]["o2_o3_programs"]
    if row.get("action_bucket") == "repeat_timing_candidates"
}
programs |= {
    row["program"]
    for row in audit["strict_time_filter"]["missing_strict_baseline_programs"]
}

for program in sorted(programs):
    print(program)
PY
)

printf '%s\n' "${PROGRAMS[@]}"
```

先 dry-run 看目标是否对：

```bash
CMD=(.venv/bin/python scripts/collect_repeat_timing.py --overwrite)
for p in "${PROGRAMS[@]}"; do
  CMD+=(--program "$p")
done

"${CMD[@]}" --dry-run
```

如果 dry-run 没问题，正式跑：

```bash
"${CMD[@]}"
```

如果你只想先跑一小批试试，可以先加 `--limit 5`：

```bash
"${CMD[@]}" --dry-run --limit 5
"${CMD[@]}" --limit 5
```

如果你只想收紧到 `O0/O2/O3`，把命令改成这样：

```bash
CMD=(.venv/bin/python scripts/collect_repeat_timing.py --overwrite --variant O0 --variant O2 --variant O3)
for p in "${PROGRAMS[@]}"; do
  CMD+=(--program "$p")
done
"${CMD[@]}" --dry-run
"${CMD[@]}"
```

跑完以后，测试机上会新增很多 `repeat_timing.json`。

位置分两种：

1. 如果原 `output_dir` 可写，就仍然写在各自 `output_dir` 下面。
2. 如果原 `output_dir` 不可写，脚本会自动回退到镜像目录 `data/llvm_test_suite/repeat_timing_sidecars/.../repeat_timing.json`。

像你这次这种 root-owned 的旧 `bcc` 运行目录，更可能看到的是第二种。

## 2. 从测试机传哪些数据

只需要传这批 sidecar，不需要重新传 `window_metrics.jsonl`、`events.jsonl`、`scores.parquet`。

在测试机上执行：

```bash
cd /home/ssy/mem-profiler
find data/llvm_test_suite -name repeat_timing.json -print > /tmp/repeat_timing_files.txt
rsync -av --files-from=/tmp/repeat_timing_files.txt ./ USER@TRAIN_HOST:/home/ssy/mem-profiler/
```

你真正传的是这一类文件：

```bash
data/llvm_test_suite/**/repeat_timing.json
```

这会同时覆盖两类位置：

1. `data/llvm_test_suite/bcc/**/repeat_timing.json`
2. `data/llvm_test_suite/repeat_timing_sidecars/**/repeat_timing.json`

如果你只想传这次新补的候选程序，也可以先生成更小的文件列表，但通常直接传全部 `repeat_timing.json` 最省心。

## 3. 在训练机上重建时间真值和评估

登录训练机后：

```bash
cd /home/ssy/mem-profiler
source .venv/bin/activate
```

直接重建时间评分表：

```bash
.venv/bin/python scripts/build_time_score_table.py
```

再重做时间外部验证：

```bash
.venv/bin/python scripts/evaluate_score_vs_time.py
```

这两步会更新：

1. time_scores.parquet
2. time_score_filter_summary.json
3. score_time_eval.json

新的 score_time_eval.json 里会有：

1. `preferred_source_counts`
2. `repeat_backed_only`

这样你就能单独看 repeat-backed 子集上的相关性，而不是只看整体混合结果。

## 4. 一个重要边界

如果训练机和测试机的仓库绝对路径不一样，这一步要注意。

现在 build_time_score_table.py 不只会查 `run_features.parquet` 里的 `output_dir/repeat_timing.json`，也会自动查镜像目录 `data/llvm_test_suite/repeat_timing_sidecars/...`。

另外，它也兼容一种常见跨机器情况：
如果 `run_features.parquet` 里存的是另一台机器的绝对路径，但路径里仍然保留了 `data/llvm_test_suite/...` 这段相对子树，脚本会自动把它重映射回当前仓库，再去找 sidecar。

所以多数情况下，即使测试机和训练机仓库绝对路径不同，也可以直接先试：

```bash
.venv/bin/python scripts/build_time_score_table.py
.venv/bin/python scripts/evaluate_score_vs_time.py
```

只有在下面这种更强的不一致下，才建议先补做一次：

1. `run_features.parquet` 里的 `output_dir` 子路径本身已经变了，不只是仓库根目录变了。
2. 训练机上的 `data/llvm_test_suite/...` 目录结构和测试机不一致。

那时再执行：

```bash
.venv/bin/python scripts/build_run_features.py
```

然后再执行：

```bash
.venv/bin/python scripts/build_time_score_table.py
.venv/bin/python scripts/evaluate_score_vs_time.py
```

如果两台机器仓库路径都一样，通常更不需要这一步。

## 最小执行版

如果你就想要最短的一套命令，照这个顺序走：

测试机：

```bash
cd /home/ssy/mem-profiler
source .venv/bin/activate

mapfile -t PROGRAMS < <(.venv/bin/python - <<'PY'
import json
from pathlib import Path
audit = json.loads(Path("train_set/data_quality_audit.json").read_text())
programs = {
    row["program"]
    for row in audit["pair_difficulty"]["o2_o3_programs"]
    if row.get("action_bucket") == "repeat_timing_candidates"
}
programs |= {
    row["program"]
    for row in audit["strict_time_filter"]["missing_strict_baseline_programs"]
}
for program in sorted(programs):
    print(program)
PY
)

CMD=(.venv/bin/python scripts/collect_repeat_timing.py --overwrite)
for p in "${PROGRAMS[@]}"; do
  CMD+=(--program "$p")
done

"${CMD[@]}" --dry-run
"${CMD[@]}"

find data/llvm_test_suite -name repeat_timing.json -print > /tmp/repeat_timing_files.txt
rsync -av --files-from=/tmp/repeat_timing_files.txt ./ USER@TRAIN_HOST:/home/ssy/mem-profiler/
```

训练机：

```bash
cd /home/ssy/mem-profiler
source .venv/bin/activate
.venv/bin/python scripts/build_time_score_table.py
.venv/bin/python scripts/evaluate_score_vs_time.py
```

如果你要，我下一条可以直接把“只跑 28 个 repeat-timing candidates 的精确 program 名单”给你展开成一串现成的 `--program ...` 命令。
