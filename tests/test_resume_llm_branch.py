"""The repair path for LLM Full branches (decision 2026-10-05) must reproduce a branch exactly when it
replays every call, and must verify every replayed outgoing message before going live."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_full_path import job  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def resume(tmp_path, name, live_from):
    spec = tmp_path / 'ok.json'
    out = tmp_path / name / 'record.json'
    proc = subprocess.run([sys.executable, str(ROOT / 'scripts/resume_llm_branch.py'), '--job-spec', str(spec),
                           '--audit', str(tmp_path / 'ok' / 'out.audit.jsonl'), '--private-dir', str(tmp_path / name / 'private'),
                           '--out', str(out), '--live-from', str(live_from)],
                          capture_output=True, text=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'tests')})
    return proc, (json.loads(out.read_text()) if out.exists() else None)


def test_full_replay_reproduces_the_branch_and_live_resume_verifies_the_prefix(tmp_path):
    code, original = job(tmp_path, 'ok', 'python:full_path_models:Exerciser')
    assert code == 0
    proc, rec = resume(tmp_path, 'replay', 'none')
    assert proc.returncode == 0, proc.stderr[-3000:]
    full = original['branches']['full']
    assert rec['branches']['full']['public_transcript'] == full['public_transcript']
    assert Path(rec['branches']['full']['settlement']).read_bytes() == Path(full['settlement']).read_bytes()
    assert rec['repair']['live_calls'] == 0 and rec['repair']['replayed_calls_verified'] > 0
    proc, rec = resume(tmp_path, 'live', 3)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert rec['repair']['modes']['full'][:3] == ['replayed', 'replayed', 'live']
    assert set(rec['repair']['modes']['shared_day_0']) == {'replayed'}
    assert rec['status'] == 'completed'


def test_a_changed_prefix_stops_the_repair(tmp_path):
    code, _ = job(tmp_path, 'ok', 'python:full_path_models:Exerciser')
    audit = tmp_path / 'ok' / 'out.audit.jsonl'
    lines = audit.read_text().splitlines()
    recs = [json.loads(line) for line in lines]
    i = next(k for k, r in enumerate(recs) if r['label'] == 'full')
    recs[i]['messages_sha256'] = '0' * 64          # pretend the recorded message was different
    audit.write_text('\n'.join(json.dumps(r) for r in recs) + '\n')
    proc, rec = resume(tmp_path, 'bad', 'none')
    assert proc.returncode != 0 and 'differs from the recorded message' in proc.stderr
