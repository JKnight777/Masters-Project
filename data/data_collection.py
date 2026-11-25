import logging
import librosa, random
import numpy as np
import soundfile as sf
import sys
from pathlib import Path
from data_utils import background_noises, mappings
from mfcc_extraction import extract_mfccs_sf, extract_mfccs_fw

try:
    dataset = sys.argv[1]
    snr_type = sys.argv[2]
    assert dataset in ['train', 'test']
    assert snr_type in ['mixed', 'single_level', 'clean']
    run = sys.argv[3]
    if dataset == 'train':
        if snr_type == 'single_level':
            SINGLE_LEVEL = float(sys.argv[4])
            output_dir = Path(f"./{SINGLE_LEVEL}_data_{run}/")
        else:
            output_dir = Path(f"./{snr_type}_data_{run}/")
    else:
        if snr_type == 'single_level':
            SINGLE_LEVEL = float(sys.argv[4])
            output_dir = Path(f"./Test Sets/{SINGLE_LEVEL}/")
        else:
            output_dir = Path(f"./Test Sets/{snr_type}/")
except:
    print("Please run with 'train' or 'test', and 'single_level', 'mixed', or 'clean'")
    exit(1)

npseed = np.random.SeedSequence()
rng = np.random.default_rng(npseed)

pyseed = random.randrange(sys.maxsize)
random.seed(pyseed)

timit_dir = Path(f"./timit/{dataset}/")
# output_dir = Path(f"./{snr_type}_data/{dataset}/")
sound_bible_dir = Path("./musan/noise/sound-bible/")
background_noise_dir = Path("./musan/noise/free-sound/")
silence_script = Path("./silence.VSS")

output_dir.mkdir(parents=True, exist_ok=True)

CLEAN_PER = 0.15
SILENT_PER = 0.05
NOISE_PER = 0.05

extract_mfccs = extract_mfccs_sf

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'{output_dir}/{snr_type}_{dataset}.log', level=logging.INFO)

logging.info(f"Numpy Seed: {npseed.entropy}")
logging.info(f"Python Random Seed: {pyseed}")

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
    '''

    speech, speech_sr = librosa.load(signal_file, sr=None)
    speech_duration = librosa.get_duration(path=signal_file, sr=speech_sr)
    noise_duration = librosa.get_duration(path=noise_file, sr=speech_sr)
    logging.info(f"Speech duration: {speech_duration}")
    logging.info(f"Sample rate: {speech_sr}")

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

    return audio_data, speech_sr, snr

def generate_silence(duration, sr, seed = None):
    n_samples = round(duration * sr)

    if n_samples < 1:
        raise ValueError("Duration must produce at least one sample.")

    audio = rng.standard_normal(n_samples)

    # Set the RMS amplitude to a random dBFS level.
    level_dbfs = rng.uniform(-75, -50)

    current_rms = np.sqrt(np.mean(audio ** 2))
    target_rms = 10 ** (level_dbfs / 20)
    audio *= target_rms / current_rms

    return audio

def generate_noise(duration, sr, source, seed = None):

    noise, _ = librosa.load(source, sr=None)
    noise_duration = librosa.get_duration(path=source, sr=None)
    repititions = int(duration // noise_duration) + 1 # Number of times noise must be repeated to meet speech size
    noise = np.tile(noise, repititions)[:int(duration * sr)] # Scale up or not at all, and avoid spill-over.
    
    logging.info(f"Original noise duration: {noise_duration} seconds, Scaled to final duration: {librosa.get_duration(y=noise, sr=sr)} seconds")
    
    audio = np.asarray(noise, dtype=np.float64)
    current_rms = np.sqrt(np.mean(audio ** 2))
    target_dbfs = rng.uniform(-47, -27) if snr_type == 'mixed' else -37
    target_rms = 10 ** (target_dbfs / 20)
    scaled_audio = audio * (target_rms / current_rms)

    return scaled_audio

if __name__ == "__main__":
    # There are 87 sound bible noises and 37 background_noises, totaling 124 noise files to choose from
    i = 0
    samples = list(timit_dir.rglob('*.WAV')) # Get all wav files in directory
    random.shuffle(samples) # Randomize samples
    total_samples = len(samples)

    if snr_type == 'clean':
        num_clean = total_samples
        clean = samples
        samples = []

    else:
        num_clean = int((total_samples * CLEAN_PER) // (1.0 - SILENT_PER - NOISE_PER)) # cx / 1 - a - b  
        clean = samples[:num_clean] # Separate and reserve clean for later.
        samples = samples[num_clean:]


    average_snr = 0.0

    # Mix remaining samples
    for signal_file in samples:
        logging.info('-' * 100)
        logging.info(signal_file.stem)
            
        # Mix noise with clean data.
        selection = random.randint(0, 123)
        if selection > 86:
            # Use a background noise instead.
            noise_file = background_noise_dir / f"{background_noises[selection - 87]}.wav"
        else:
            noise_file = sound_bible_dir / f'noise-sound-bible-{selection:04d}.wav'

        assert noise_file.exists(), f"The selected noise file: {noise_file} does not exist!"
        logging.info(f"Using {signal_file} as speech.")
        logging.info(f"Using {noise_file} as noise.")
        targetsnr = random.randint(-5, 15) if snr_type == 'mixed' else SINGLE_LEVEL
        logging.info(f"Target SNR: {targetsnr} dB")
        audio_data, speech_sr, snr = calculate_SNR(signal_file, noise_file, targetsnr)

        average_snr += snr

        mixed_file = f"{output_dir}\\{dataset}_{i:04d}.wav"
        sf.write(mixed_file, audio_data, speech_sr)
        logging.info(f"Mixed file: {output_dir}\\{dataset}_{i:04d}.wav written successfully.\n")

        # Once mixed file is generated, caluclate mfccs and truths using the converted script file.
        try:
            phn_script = signal_file.with_suffix('.VSS')
            assert phn_script.exists(), f"{phn_script} was not found! No data will be generated for {mixed_file}!"
            logging.info(f"Script used: {phn_script}.\n\n")
            x, y, labels, config = extract_mfccs(mixed_file, phn_script)
            # Save to compressed numpy file.
            np.savez_compressed(
                f'{output_dir}\\{dataset}_{i:04d}.npz',
                x=x,
                y=y,
                labels=labels,
                config_json = config
            )
        except Exception as e:
            logging.error(f"Something went wrong: {e}")
        i += 1

    # Calculate and Save clean samples
    logging.info('-' * 100)
    for signal_file in clean:
        logging.info(signal_file.stem)
        
        # Copy paste files and calculate MFCCs.
        dest = f"{output_dir}\\{dataset}_{i:04d}.wav"
        audio, speech_sr = librosa.load(signal_file, sr=None)
        audio_data = np.asarray(audio, dtype=np.float64)
        sf.write(dest, audio_data, speech_sr)
        logging.info(f"Clean file: {output_dir}\\{dataset}_{i:04d}.wav written successfully.\n")

        try:
            phn_script = signal_file.with_suffix('.VSS')
            assert phn_script.exists(), f"{phn_script} was not found! No data will be generated for {signal_file}!"
            logging.info(f"Script used: {phn_script}.\n\n")
            x, y, labels, config = extract_mfccs(signal_file, phn_script)
            # Save to compressed numpy file.
            np.savez_compressed(
                f'{output_dir}\\{dataset}_{i:04d}.npz',
                x=x,
                y=y,
                labels=labels,
                config_json = config
            )
        except Exception as e:
            logging.error(f"Something went wrong: {e}")

        i += 1

    logging.info('-' * 100)
    # Create 15% clean (no noise added), 5% silent, 5% noise only options as well for noise datasets. (10% silence only for clean dataset)
    if snr_type == 'clean':
        num_silent = int(((total_samples * (SILENT_PER + NOISE_PER)) / (1.0 - SILENT_PER - NOISE_PER))) # ax / (1 - a - b)
        num_noise = 0
    else:
        num_silent = int(((total_samples * SILENT_PER) / (1.0 - SILENT_PER - NOISE_PER))) # ax / (1 - a - b)
        num_noise = int(((total_samples * NOISE_PER) / (1.0 - SILENT_PER - NOISE_PER))) # ax / (1 - a - b)

    for _ in range(num_silent):
        duration = round(random.uniform(1.5, 4.0), 2)
        audio_data = generate_silence(duration=duration, sr=16000)
        dest = f"{output_dir}\\{dataset}_{i:04d}.wav"
        sf.write(dest, audio_data, 16000)
        logging.info(f"Silent file: {output_dir}\\{dataset}_{i:04d}.wav of {duration} sec was written successfully.\n")

        try:
            assert silence_script.exists(), f"{silence_script} was not found! No data will be generated for {dest}!"
            logging.info(f"Script used: {silence_script}.\n\n")
            x, y, labels, config = extract_mfccs(dest, silence_script)
            # Save to compressed numpy file.
            np.savez_compressed(
                f'{output_dir}\\{dataset}_{i:04d}.npz',
                x=x,
                y=y,
                labels=labels,
                config_json = config
            )
        except Exception as e:
            logging.error(f"Something went wrong: {e}")

        i += 1

    for _ in range(num_noise):
        selection = random.randint(0, 123)
        if selection > 86:
            # Use a background noise instead.
            noise_file = background_noise_dir / f"{background_noises[selection - 87]}.wav"
        else:
            noise_file = sound_bible_dir / f'noise-sound-bible-{selection:04d}.wav'

        assert noise_file.exists(), f"The selected noise file: {noise_file} does not exist!"

        duration = round(random.uniform(1.5, 4.0), 2)
        audio_data = generate_noise(duration=duration, source=noise_file, sr=16000)
        dest = f"{output_dir}\\{dataset}_{i:04d}.wav"
        sf.write(dest, audio_data, 16000)
        logging.info(f"Noise file: {output_dir}\\{dataset}_{i:04d}.wav of {duration} sec was written successfully.\n")

        try:
            assert silence_script.exists(), f"{silence_script} was not found! No data will be generated for {dest}!"
            logging.info(f"Script used: {silence_script}.\n\n")
            x, y, labels, config = extract_mfccs(dest, silence_script)
            # Save to compressed numpy file.
            np.savez_compressed(
                f'{output_dir}\\{dataset}_{i:04d}.npz',
                x=x,
                y=y,
                labels=labels,
                config_json = config
            )
        except Exception as e:
            logging.error(f"Something went wrong: {e}")

        i += 1

    total_samples += (num_noise + num_silent)
        
    logging.info(f"Successfully created all {i} {dataset}ing files. ({round((len(clean) / total_samples), 2) * 100}% clean, {round((num_silent / total_samples), 2) * 100}% silent, {round((num_noise / total_samples), 2) * 100}% noise).")
    if snr_type != 'clean':
        logging.info(f"Average SNR of all mixed files: {round(average_snr / len(samples), 2)} dB")