import argparse
import codecs
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path
 
import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
from omegaconf import OmegaConf
import pathlib
import os
from dotenv import load_dotenv
 
 
load_dotenv()
 
CURRENT_DIR = Path(__file__).parent.resolve()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR))))
 
from f5_tts.infer.utils_infer import (
    mel_spec_type,
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT, UNetT  # noqa: F401. used for config
 
 
parser = argparse.ArgumentParser(
    prog="python3 infer-cli.py",
    description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
    epilog="Specify options above to override one or more settings from config.",
)
 
# # config file
 
config = tomli.load(open(os.path.join(files("f5_tts").joinpath("infer/examples/basic"), "basic.toml"), "rb"))

ref_audio = "infer/examples/basic/basic_ref_en.wav"
ref_text = "Some call me nature, others call me mother nature."
 
model = os.environ.get("MODEL")

ckpt_file_it = os.environ.get("CKPT_FILE_IT")
vocab_file_it = os.environ.get("VOCAB_FILE_IT")

ckpt_file_gen = os.environ.get("CKPT_FILE_GEN")
vocab_file_gen = os.environ.get("VOCAB_FILE_GEN")


use_ema = False

ref_audio_it = os.environ.get("REF_AUDIO_IT")
ref_text_it = os.environ.get("REF_TEXT_IT")

ref_audio_en = os.environ.get("REF_AUDIO_EN")
ref_text_en = os.environ.get("REF_TEXT_EN")

ref_audio_de = os.environ.get("REF_AUDIO_DE")
ref_text_de = os.environ.get("REF_TEXT_DE")

ref_audio_fr = os.environ.get("REF_AUDIO_FR")
ref_text_fr = os.environ.get("REF_TEXT_FR")
 
 
gen_file = config.get("gen_file", "")
 
output_dir = config.get("output_dir", "tests")
output_file = config.get(
    "output_file", f"infer_cli_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav"
)
 
save_chunk = config.get("save_chunk", False)
remove_silence = config.get("remove_silence", False)
load_vocoder_from_local = config.get("load_vocoder_from_local", False)
 
vocoder_name = config.get("vocoder_name", mel_spec_type)
target_rms = config.get("target_rms", target_rms)
cross_fade_duration = config.get("cross_fade_duration", cross_fade_duration)
nfe_step = config.get("nfe_step", nfe_step)
cfg_strength = config.get("cfg_strength", cfg_strength)
sway_sampling_coef = config.get("sway_sampling_coef", sway_sampling_coef)
speed = config.get("speed", speed)
fix_duration = config.get("fix_duration", fix_duration)
 
 
# patches for pip pkg user
if "infer/examples/" in ref_audio:
    ref_audio = str(files("f5_tts").joinpath(f"{ref_audio}"))
if "infer/examples/" in gen_file:
    gen_file = str(files("f5_tts").joinpath(f"{gen_file}"))
if "voices" in config:
    for voice in config["voices"]:
        voice_ref_audio = config["voices"][voice]["ref_audio"]
        if "infer/examples/" in voice_ref_audio:
            config["voices"][voice]["ref_audio"] = str(files("f5_tts").joinpath(f"{voice_ref_audio}"))
 
 
# ignore gen_text if gen_file provided
 
if gen_file:
    gen_text = codecs.open(gen_file, "r", "utf-8").read()
 
 
# output path
 
wave_path = Path(output_dir) / output_file
# spectrogram_path = Path(output_dir) / "infer_cli_out.png"
if save_chunk:
    output_chunk_dir = os.path.join(output_dir, f"{Path(output_file).stem}_chunks")
    if not os.path.exists(output_chunk_dir):
        os.makedirs(output_chunk_dir)
 
 
# load vocoder
 
if vocoder_name == "vocos":
    vocoder_local_path = "../checkpoints/vocos-mel-24khz"
elif vocoder_name == "bigvgan":
    vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
 
vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=load_vocoder_from_local, local_path=vocoder_local_path)
 
# load TTS model
 
model_cfg = OmegaConf.load(
    config.get("model_cfg", str(files("f5_tts").joinpath(f"configs/{model}.yaml")))
).model
model_cls = globals()[model_cfg.backbone]
 
repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"
 
if model != "F5TTS_Base":
    assert vocoder_name == model_cfg.mel_spec.mel_spec_type
 
# override for previous models
if model == "F5TTS_Base":
    if vocoder_name == "vocos":
        ckpt_step = 1200000
    elif vocoder_name == "bigvgan":
        model = "F5TTS_Base_bigvgan"
        ckpt_type = "pt"
elif model == "E2TTS_Base":
    repo_name = "E2-TTS"
    ckpt_step = 1200000

if not ckpt_file_it:
    ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}"))

 
ema_model_it = load_model(model_cls, model_cfg.arch, ckpt_file_it, mel_spec_type=vocoder_name, vocab_file=vocab_file_it, use_ema=use_ema)
ema_model_gen = load_model(model_cls, model_cfg.arch, ckpt_file_gen, mel_spec_type=vocoder_name, vocab_file=vocab_file_gen, use_ema=True)
 
def normalize_text(text: str) -> str:
    """
    Function that normilize text for TTS Dataset
    Args:
        text: str -> String to be normalized
    Returns:
        str: Normalized text
    """
    text = text.lower()
    text = re.sub(r"[^\w\s\']", "", text)
    text = " ".join(text.split())
    return text
 
# inference process
def main(gen_text: str, language: str):

    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    if "voices" not in config:
        voices = {"main": main_voice}
    else:
        voices = config["voices"]
        voices["main"] = main_voice
    for voice in voices:
        voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
            voices[voice]["ref_audio"], voices[voice]["ref_text"]
        )
    
    match language:
        case "it":
            ema_model = ema_model_it
            ref_audio_ = ref_audio_it
            ref_text_ = ref_text_it 
        case "en":
            ema_model = ema_model_gen
            ref_audio_ = ref_audio_en
            ref_text_ = ref_text_en
        case "de":
            ema_model = ema_model_gen
            ref_audio_ = ref_audio_de
            ref_text_ = ref_text_de
        case "fr":
            ema_model = ema_model_gen
            ref_audio_ = ref_audio_fr
            ref_text_ = ref_text_fr
        case _:
            ema_model = ema_model_gen
            ref_audio_ = ref_audio_en
            ref_text_ = ref_text_en

    generated_audio_segments = []
    reg1 = r"(?=\[\w+\])"
    chunks = re.split(reg1, gen_text)
    reg2 = r"\[(\w+)\]"
    for text in chunks:
        if not text.strip():
            continue
        match = re.match(reg2, text)
        if match:
            voice = match[1]
        else:
            voice = "main"
        if voice not in voices:
            voice = "main"
        text = re.sub(reg2, "", text)
    
        gen_text_ = text.strip()
        
        print(ref_audio_, ref_text_)
        audio_segment, final_sample_rate, spectragram = infer_process(
            ref_audio_,
            ref_text_,
            gen_text_,
            ema_model,
            vocoder,
            mel_spec_type=vocoder_name,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
        )
        generated_audio_segments.append(audio_segment)
 
        if save_chunk:
            if len(gen_text_) > 200:
                gen_text_ = gen_text_[:200] + " ... "
            sf.write(
                os.path.join(output_chunk_dir, f"{len(generated_audio_segments)-1}_{gen_text_}.wav"),
                audio_segment,
                final_sample_rate,
            )
 
    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)
 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
 
        with open(wave_path, "wb") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            # Remove silence
            if remove_silence:
                remove_silence_for_generated_wav(f.name)
            print(f.name)
            
if __name__ == "__main__":
    main("The Natù project is dedicated to inspiring individuals to embrace a more sustainable way of living by providing eco-friendly products and raising awareness about the importance of protecting the environment.", "en")