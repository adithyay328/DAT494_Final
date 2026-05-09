# We are workign with the ML Commons people speech dataset.
# For each one, all we are doing is a) downloading, b)
# recoding into mp3 at 96kbps, and c) putting into a directory
# called "data". After that, the pipeline is that for each one,
# a) we have an asyncio loader that just runs continuously, b)
# picks a random piece of mp3 data, and hands off to a different
# multiprocess thread. That thread cuts the mp3 into a 5 second chunk,
# and a) asyncio sends it into gemini 3.1 flash lite for audio transcription
# + checking if there is any speech at all, and b) get a 50 millisecond by
# 50 millisecond amplitude chart(should be 50 log-scale magnitudes), and return both.
# As well, it needs to queue up encoding the chunks into the EnCodec model, which runs
# every 250 milliseconds, but we can queue up a batch into it by writing into a multiprocess
# queue, and encoding the audio chunk into .wav in a temp folder, and handing it off to the model.
# After all of that, that dict is available to retrieve by the training loop
import os
import subprocess
import tempfile
import uuid

import numpy as np
import soundfile as sf
from datasets import load_dataset

DATA_DIR = "data"
MAX_SAMPLES = 100_000


def download_samples(data_dir: str = DATA_DIR, max_samples: int = MAX_SAMPLES) -> int:
    """Stream up to *max_samples* from peoples_speech, save each as 96 kbps MP3.

    Each file is named ``<uuid4>.mp3`` inside *data_dir*.
    Returns the number of files actually written.
    """
    os.makedirs(data_dir, exist_ok=True)

    ds = load_dataset(
        "MLCommons/peoples_speech", "clean", split="train", streaming=True
    )

    count = 0
    for sample in ds:
        if count >= max_samples:
            break

        audio = sample["audio"]  # dict with "array" and "sampling_rate"
        arr = np.asarray(audio["array"])

        # Skip empty / silent samples
        if arr.size == 0 or np.max(np.abs(arr)) < 1e-6:
            continue

        out_path = os.path.join(data_dir, f"{uuid.uuid4()}.mp3")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        sf.write(tmp_path, arr, audio["sampling_rate"])

        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-i", tmp_path, "-b:a", "96k", out_path],
                check=True,
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        count += 1
        if count % 100 == 0:
            print(f"  … {count} files saved")

    print(f"Done – {count} MP3 files in {data_dir}/")
    return count


if __name__ == "__main__":
    download_samples()
