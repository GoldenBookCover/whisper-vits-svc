import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmvpe.RMVPEF0Predictor import RMVPEF0Predictor

import numpy as np
import librosa
import torch
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 参考 https://github.com/ShadowLoveElysia/Whisper-vits-svc-LargeV3/blob/MiX3/prepare/preprocess_rmvpe.py


def compute_f0(filename, save, device):
    predictor = RMVPEF0Predictor(hop_length=160, f0_min=50, f0_max=1100, device=device)
    audio, sr = librosa.load(filename, sr=16000)
    assert sr == 16000
    # Load audio
    audio = torch.tensor(np.copy(audio))
    audio = audio + torch.randn_like(audio) * 0.001

    pitch, uv = predictor.compute_f0_uv(audio)

    pitch = abs(pitch) * uv
    pitch = torch.from_numpy(pitch).to(torch.float32)
    pitch = pitch.squeeze(0)
    np.save(save, pitch, allow_pickle=False)


def process_file(file, wavPath, spks, pitPath, device):
    # print(f'debug: process_file with args ({file = }, { wavPath = }, { spks = }, { pitPath = })')
    if file.endswith(".wav"):
        file = file[:-4]
        compute_f0(f"{wavPath}/{spks}/{file}.wav", f"{pitPath}/{spks}/{file}.pit", device)

def process_files_with_process_pool(wavPath, spks, pitPath, device, process_num=None):
    files = [f for f in os.listdir(f"./{wavPath}/{spks}") if f.endswith(".wav")]

    with ProcessPoolExecutor(max_workers=process_num) as executor:
        futures = [executor.submit(process_file, file, wavPath, spks, pitPath, device) for file in files]

        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing files'):
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-p", "--pit", help="pit", dest="pit")
    parser.add_argument("-t", "--thread_count", help="thread count to process, set 0 to use all cpu cores", dest="thread_count", type=int, default=1)
    args = parser.parse_args()
    print(args.wav)
    print(args.pit)
    device = torch.device('cpu')  # "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.pit, exist_ok=True)
    wavPath = args.wav
    pitPath = args.pit

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{pitPath}/{spks}", exist_ok=True)
            print(f">>>>>>>>>> {spks} <<<<<<<<<<")
            if args.thread_count == 0:
                process_num = os.cpu_count()
            else:
                process_num = args.thread_count
            print(f"Processing rmvpe with {process_num} threads")
            process_files_with_process_pool(wavPath, spks, pitPath, device, process_num)
