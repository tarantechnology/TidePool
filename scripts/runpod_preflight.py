#!/usr/bin/env python3
"""Read-only GPU/storage/model-access audit. Does not download model weights.

Run inside the pod. Secrets are read only by the Hub client and are never printed.
df on Runpod's shared FUSE mount does not reveal the account's volume quota.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return dict(returncode=result.returncode, stdout=result.stdout.strip())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a fresh output path')
    report = dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  user=command(['whoami']), cwd=str(Path.cwd()), python=platform.python_version(),
                  mount=command(['findmnt', '-J', '-T', '/workspace', '-o', 'TARGET,SOURCE,FSTYPE,OPTIONS']),
                  disk=command(['df', '-h', '/workspace']), account_volume_quota_bytes=None,
                  persistence='Inspect mount source: pod volume survives stops; survival after deletion is not established',
                  gpu=command(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version', '--format=csv']),
                  gpu_processes=command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv']),
                  model_access=[])
    for path in ['/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes']:
        if Path(path).exists():
            report['cgroup_memory_limit_bytes'] = Path(path).read_text().strip()
            break
    if importlib.util.find_spec('torch'):
        import torch
        report['torch'] = dict(version=torch.__version__, cuda_runtime=torch.version.cuda,
                               cuda_available=torch.cuda.is_available())
    if importlib.util.find_spec('huggingface_hub'):
        from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url
        report['hf_auth_present'] = bool(get_token())
        for repo, filename in [('facebook/sam3.1', 'sam3.1_multiplex.pt'),
                               ('facebook/sam3', 'sam3.pt'),
                               ('facebook/sam-audio-large-tv', 'config.json'),
                               ('facebook/sam-audio-base-tv', 'config.json'),
                               ('facebook/sam-audio-small-tv', 'config.json')]:
            item = dict(repo=repo, filename=filename)
            try:
                meta = get_hf_file_metadata(hf_hub_url(repo, filename))
                item.update(accessible=True, size_bytes=meta.size, revision=meta.commit_hash)
            except Exception as exc:
                response = getattr(exc, 'response', None)
                item.update(accessible=False, error_type=type(exc).__name__,
                            http_status=getattr(response, 'status_code', None))
            report['model_access'].append(item)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
