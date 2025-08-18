import os
import torch
import argparse
import subprocess
import warnings

from svc_ensure_models import ensure_models

warnings.filterwarnings('ignore')
print("CPU Count is :", os.cpu_count())
ensure_models()
parser = argparse.ArgumentParser()
parser.add_argument("-t", type=int, default=0, help="thread count")
parser.add_argument("--custom_whisper", type=str, default=None, help="Custom whisper")
args = parser.parse_args()

if args.custom_whisper:
    whisper_cmd = "uv run prepare/preprocess_ppg.py -w data_svc/waves-16k/ -p data_svc/whisper --custom_whisper " + args.custom_whisper
else:
    whisper_cmd = "uv run prepare/preprocess_ppg.py -w data_svc/waves-16k/ -p data_svc/whisper"
commands = [
    f"uv run prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-16k -s 16000 -t {args.t}",
    f"uv run prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-48k -s 48000 -t {args.t}",
    #f"uv run prepare/preprocess_crepe.py -w data_svc/waves-16k/ -p data_svc/pitch",
    f"uv run prepare/preprocess_rmvpe.py -w data_svc/waves-16k/ -p data_svc/pitch -t {args.t}",
    whisper_cmd,
    f"uv run prepare/preprocess_hubert.py -w data_svc/waves-16k/ -v data_svc/hubert",
    f"uv run prepare/preprocess_speaker.py data_svc/waves-16k/ data_svc/speaker -t {args.t}",
    f"uv run prepare/preprocess_speaker_ave.py data_svc/speaker/ data_svc/singer",
    f"uv run prepare/preprocess_spec.py -w data_svc/waves-48k/ -s data_svc/specs -t {args.t}",
    f"uv run prepare/preprocess_train.py",
    f"uv run prepare/preprocess_zzz.py",
]

for command in commands:
    print(f"Command: {command}")

    process = subprocess.Popen(command, shell=True)
    outcode = process.wait()
    if (outcode):
        break
