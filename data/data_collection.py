import logging
import librosa, random
import numpy as np
import soundfile as sf
import sys
from pathlib import Path
from data_utils import background_noises, mappings

try:
    dataset = sys.argv[1]
    assert dataset in ['train', 'test']
except:
    print("Please run with 'train' or 'test'")
    exit(1)

timit_dir = Path(f"./timit/{dataset}/")
output_dir = Path(f"./mixed_data/{dataset}/")
sound_bible_dir = Path("./musan/noise/sound-bible/")
background_noise_dir = Path("./musan/noise/free-sound/")

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'{output_dir}/mixed.log', level=logging.INFO)

def translate_all():
    timit_dir = Path("./timit")

    for phn_file in timit_dir.rglob("*.PHN"):
        with open(phn_file, 'r') as phn:
            print(f"Reading {phn_file.__str__()[:-3]}VSS")
            with open(f"{phn_file.__str__()[:-3]}VSS", "a") as vss:
                for line in phn:
                    new_line = line.strip().split()
                    print(f"Original: {new_line}")
                    timestamps = new_line[:2]
                    phonemes = new_line[2:]
                    for i in range(len(phonemes)):
                        phonemes[i] = mappings[phonemes[i]]
                    translation = timestamps + phonemes
                    print(f"Translated to: {translation}\n")
                    vss.write(" ".join(translation))
                    vss.write('\n')
        print(f"{phn_file} has been translated.\n\n\n")

def calculate_SNR(signal_file, noise_file, targetsnr):
    '''
    SNR in dB = 10 * log10(Ps / Pn)
    Power of the signal is the clean TIMIT audio only in db.
    Power of the noise is the noise from MUSAN only in db.
    Calculate the power of each speech clip alone before mixing.
    Grab amplitude of the wav file from several samples. 
    '''

    speech, speech_sr = librosa.load(signal_file, sr=None)
    speech_duration = librosa.get_duration(path=signal_file, sr=speech_sr)
    noise_duration = librosa.get_duration(path=noise_file, sr=speech_sr)
    logging.info(f"Speech duration: {speech_duration}")

    noise, _ = librosa.load(noise_file, sr=None)
    repititions = (speech.size + noise.size - 1) // noise.size # Number of times noise must be repeated to meet speech size
    noise = np.tile(noise, repititions)[:speech.size] # Scale up or not at all, and only include size of speech to avoid spill-over.
    logging.info(f"Original noise duration: {noise_duration} seconds, Scaled to final duration: {librosa.get_duration(y=noise, sr=speech_sr)} seconds")

    speech = np.asarray(speech, dtype=np.float64)
    speech_db = 10 * np.log10(np.mean(speech ** 2))
    noise_db = 10 * np.log10(np.mean(noise ** 2))

    noise_offset = speech_db - targetsnr
    noise_scale = 10 ** ((noise_offset - noise_db) / 20)
    scaled_noise = noise * noise_scale
    scaled_noise_db = 10 * np.log10(np.mean(scaled_noise ** 2))

    noise_duration = librosa.get_duration(y=noise, sr=speech_sr)

    snr = (speech_db - scaled_noise_db)
    logging.info(f"Actual SNR: {snr} dB")

    audio_data = (speech + scaled_noise)

    return audio_data, speech_sr

if __name__ == "__main__":
    # There are 87 sound bible noises and 37 background_noises, totalling 124 noise files to choose from
    i = 0
    for signal_file in timit_dir.rglob("*.WAV"):
        logging.info('-' * 100)
        logging.info(signal_file.stem)
        selection = random.randint(0, 123)
        if selection > 86:
            # Use a background noise instead
            noise_file = background_noise_dir / f"{background_noises[selection - 87]}.wav"
        else:
            noise_file = sound_bible_dir / f'noise-sound-bible-{selection:04d}.wav'
        assert signal_file.exists(), f"The selected speech file: {signal_file} does not exist!"
        assert noise_file.exists(), f"The selected noise file: {noise_file} does not exist!"
        logging.info(f"Using {signal_file} as speech.")
        logging.info(f"Using {noise_file} as noise.")
        targetsnr = random.randint(-5, 15)
        logging.info(f"Target SNR: {targetsnr} dB")
        audio_data, speech_sr = calculate_SNR(signal_file, noise_file, targetsnr)
        sf.write(f"{output_dir}\\{dataset}_{i:04d}.wav", audio_data, speech_sr)
        logging.info(f"Mixed file: {output_dir}\\{dataset}_{i:04d}.wav written successfully.\n\n")
        i += 1
    logging.info(f"Successfully mixed all {i} {dataset}ing files.")