import math
import os
from dataclasses import dataclass, field, replace
from itertools import chain
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torchaudio
from coqpit import Coqpit
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from trainer.torch import DistributedSampler, DistributedSamplerWrapper
from trainer.trainer_utils import get_optimizer, get_scheduler

from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.datasets.dataset import TTSDataset, _parse_sample
from TTS.tts.layers.glow_tts.duration_predictor import DurationPredictor
from TTS.tts.layers.vits.discriminator import VitsDiscriminator
from TTS.tts.layers.vitspitch.networks import PosteriorEncoder, ResidualCouplingBlocks, TextEncoder
#ADDITION FOR MODULAR_VITS
from TTS.tts.layers.vits.stochastic_duration_predictor import StochasticDurationPredictor
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.helpers import generate_path, maximum_path, rand_segments, segment, sequence_mask, average_over_durations
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.characters import BaseCharacters, _characters, _pad, _phonemes, _punctuations
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment
#ADDITION FOR FAST_PITCH
from TTS.tts.utils.visual import plot_avg_pitch
from TTS.tts.layers.generic.aligner import AlignmentNetwork
from TTS.tts.utils.data import prepare_data,prepare_tensor


from TTS.utils.io import load_fsspec
from TTS.utils.samplers import BucketBatchSampler
from TTS.vocoder.models.hifigan_generator import HifiganGenerator
from TTS.vocoder.utils.generic_utils import plot_results

##############################
# IO / Feature extraction
##############################

# pylint: disable=global-statement
hann_window = {}
mel_basis = {}


@torch.no_grad()
def weights_reset(m: nn.Module):
    # check if the current module has reset_parameters and if it is reset the weight
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


def get_module_weights_sum(mdl: nn.Module):
    dict_sums = {}
    for name, w in mdl.named_parameters():
        if "weight" in name:
            value = w.data.sum().item()
            dict_sums[name] = value
    return dict_sums


def load_audio(file_path):
    """Load the audio file normalized in [-1, 1]

    Return Shapes:
        - x: :math:`[1, T]`
    """
    x, sr = torchaudio.load(file_path)
    assert (x > 1).sum() + (x < -1).sum() == 0
    return x, sr


def _amp_to_db(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def _db_to_amp(x, C=1):
    return torch.exp(x) / C


def amp_to_db(magnitudes):
    output = _amp_to_db(magnitudes)
    return output


def db_to_amp(magnitudes):
    output = _db_to_amp(magnitudes)
    return output


def wav_to_spec(y, n_fft, hop_length, win_length, center=False):
    """
    Args Shapes:
        - y : :math:`[B, 1, T]`

    Return Shapes:
        - spec : :math:`[B,C,T]`
    """
    y = y.squeeze(1)

    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_length) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_length).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel(spec, n_fft, num_mels, sample_rate, fmin, fmax):
    """
    Args Shapes:
        - spec : :math:`[B,C,T]`

    Return Shapes:
        - mel : :math:`[B,C,T]`
    """
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sample_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    mel = torch.matmul(mel_basis[fmax_dtype_device], spec)
    mel = amp_to_db(mel)
    return mel


def wav_to_mel(y, n_fft, num_mels, sample_rate, hop_length, win_length, fmin, fmax, center=False):
    """
    Args Shapes:
        - y : :math:`[B, 1, T]`

    Return Shapes:
        - spec : :math:`[B,C,T]`
    """
    y = y.squeeze(1)

    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_length) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sample_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_length).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = amp_to_db(spec)
    return spec

#UPDATE PITCHTOVITS FOR FAST_PITCH
def wav_pitch_to_vits(wav_pitch):
    if isinstance(wav_pitch, list):
        return [torch.from_numpy(w)[None,:] for w in wav_pitch]
    else:
        return torch.from_numpy(wav_pitch)[None,:]
        
#############################
# CONFIGS
#############################


@dataclass
class ModularVitsAudioConfig(Coqpit):
    fft_size: int = 1024
    sample_rate: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    num_mels: int = 80
    mel_fmin: int = 0
    mel_fmax: int = None
    trim_db:int = None
    do_trim_silence:bool = False
    #UPDATE FOR FAST_PITCH
    #make sure we  call those from train script, by calling VitsPitchAudioConfig and not BaseAudioConfig
    pitch_fmax: int = 640
    pitch_fmin: int =1
    min_level_db:int = -100
    power:float = 1.5
    griffin_lim_iters:int = 60
    symmetric_norm:bool = True
    max_norm:float = 4.0
    signal_norm: bool =None
    spec_gain: float =20.0
    log_func: str ="np.log"
    ref_level_db: int = 20
    preemphasis: float = 0.0


##############################
# DATASET
##############################


def get_attribute_balancer_weights(items: list, attr_name: str, multi_dict: dict = None):
    """Create inverse frequency weights for balancing the dataset.
    Use `multi_dict` to scale relative weights."""
    attr_names_samples = np.array([item[attr_name] for item in items])
    unique_attr_names = np.unique(attr_names_samples).tolist()
    attr_idx = [unique_attr_names.index(l) for l in attr_names_samples]
    attr_count = np.array([len(np.where(attr_names_samples == l)[0]) for l in unique_attr_names])
    weight_attr = 1.0 / attr_count
    dataset_samples_weight = np.array([weight_attr[l] for l in attr_idx])
    dataset_samples_weight = dataset_samples_weight / np.linalg.norm(dataset_samples_weight)
    if multi_dict is not None:
        # check if all keys are in the multi_dict

        for k in multi_dict:
            assert k in unique_attr_names, f"{k} not in {unique_attr_names}"
        # scale weights
        multiplier_samples = np.array([multi_dict.get(item[attr_name], 1.0) for item in items])
        dataset_samples_weight *= multiplier_samples
    return (
        torch.from_numpy(dataset_samples_weight).float(),
        unique_attr_names,
        np.unique(dataset_samples_weight).tolist(),
    )


class ModularVitsDataset(TTSDataset):
    def __init__(self, model_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_id = self.tokenizer.characters.pad_id
        self.model_args = model_args

    def __getitem__(self, idx):
        item = self.samples[idx]
        raw_text = item["text"]

        wav, _ = load_audio(item["audio_file"])
        if self.model_args.encoder_sample_rate is not None:
            if wav.size(1) % self.model_args.encoder_sample_rate != 0:
                wav = wav[:, : -int(wav.size(1) % self.model_args.encoder_sample_rate)]

        wav_filename = os.path.basename(item["audio_file"])

        token_ids = self.get_token_ids(idx, item["text"])

        #ADDITION FOR FAST_PITCH
        #specific format of wav
        wav_pitch = np.asarray(self.load_wav(item["audio_file"]), dtype=np.float32)


        # after phonemization the text length may change
        # this is a shameful 🤭 hack to prevent longer phonemes
        # TODO: find a better fix
        if len(token_ids) > self.max_text_len or wav_pitch_to_vits(wav_pitch).shape[1] < self.min_audio_len:
            self.rescue_item_idx += 1
            return self.__getitem__(self.rescue_item_idx)
            
        # get f0 values
        f0 = None
        if self.compute_f0:
            f0 = self.get_f0(idx)["f0"]

        return {
            "raw_text": raw_text,
            "token_ids": token_ids,
            "token_len": len(token_ids),
            "wav_file": wav_filename,
            "speaker_name": item["speaker_name"],
            "language_name": item["language"],
            "audio_unique_name": item["audio_unique_name"],
            #ADDITION FOR FAST_PITCH
            "pitch": f0,
            #"wav": wav,
            "wav_pitch":wav_pitch
        }

    @property
    def lengths(self):
        lens = []
        for item in self.samples:
            _, wav_file, *_ = _parse_sample(item)
            audio_len = os.path.getsize(wav_file) / 16 * 8  # assuming 16bit audio
            lens.append(audio_len)
        return lens

    def collate_fn(self, batch):
        """
        Return Shapes:
            - tokens: :math:`[B, T]`
            - token_lens :math:`[B]`
            - token_rel_lens :math:`[B]`
            - waveform: :math:`[B, 1, T]`
            - waveform_lens: :math:`[B]`
            - waveform_rel_lens: :math:`[B]`
            - speaker_names: :math:`[B]`
            - language_names: :math:`[B]`
            - audiofile_paths: :math:`[B]`
            - raw_texts: :math:`[B]`
            - audio_unique_names: :math:`[B]`
        """
        # convert list of dicts to dict of lists
        B = len(batch)
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        _, ids_sorted_decreasing = torch.sort(
            #UPDATE PITCHTOVITS
            torch.LongTensor([x.size(1) for x in wav_pitch_to_vits(batch["wav_pitch"])]), dim=0, descending=True
        )

        max_text_len = max([len(x) for x in batch["token_ids"]])
        token_lens = torch.LongTensor(batch["token_len"])
        token_rel_lens = token_lens / token_lens.max()

        #UPDATE PITCHTOVITS
        wav_lens = [w.shape[1] for w in wav_pitch_to_vits(batch["wav_pitch"])]
        wav_lens = torch.LongTensor(wav_lens)
        wav_lens_max = torch.max(wav_lens)
        wav_rel_lens = wav_lens / wav_lens_max

        token_padded = torch.LongTensor(B, max_text_len)
        wav_padded = torch.FloatTensor(B, 1, wav_lens_max)
        token_padded = token_padded.zero_() + self.pad_id
        wav_padded = wav_padded.zero_() + self.pad_id
        for i in range(len(ids_sorted_decreasing)):
            token_ids = batch["token_ids"][i]
            token_padded[i, : batch["token_len"][i]] = torch.LongTensor(token_ids)

            #UPDATE PITCHTOVITS
            wav = wav_pitch_to_vits(batch["wav_pitch"])[i]
            wav_padded[i, :, : wav.size(1)] = torch.FloatTensor(wav)
        
        #ADDITION FOR FAST_PITCH
        # compute features
        #mel = [self.ap.melspectrogram(np.asarray(w, dtype=np.float32)).astype("float32") for w in batch["wav"]]
        mel = [self.ap.melspectrogram(w).astype("float32") for w in batch["wav_pitch"]]
        mel_lengths = [m.shape[1] for m in mel]
        mel_lengths = torch.LongTensor(mel_lengths)
        # PAD features with longest instance
        mel = prepare_tensor(mel, self.outputs_per_step)
        # B x D x T --> B x T x D
        mel = mel.transpose(0, 2, 1)
        mel = torch.FloatTensor(mel).contiguous()   
        # format F0
        if self.compute_f0 and self.training_phase==2:
            pitch = prepare_data(batch["pitch"])
            assert mel.shape[1] == pitch.shape[1], f"[!] {mel.shape} vs {pitch.shape}"
            pitch = torch.FloatTensor(pitch)[:, None, :].contiguous()  # B x 1 xT
        else:
            pitch = None

        return {
            "tokens": token_padded,
            "token_lens": token_lens,
            "token_rel_lens": token_rel_lens,
            "waveform": wav_padded,  # (B x T)
            "waveform_lens": wav_lens,  # (B)
            "waveform_rel_lens": wav_rel_lens,
            "speaker_names": batch["speaker_name"],
            "language_names": batch["language_name"],
            "audio_files": batch["wav_file"],
            "raw_text": batch["raw_text"],
            "audio_unique_names": batch["audio_unique_name"],
            #ADDITION FOR FAST_PITCH
            "pitch" : pitch,
            "mel_input" : mel,
            "mel_lengths":mel_lengths
        }


##############################
# MODEL DEFINITION
##############################


@dataclass
class ModularVitsArgs(Coqpit):
    """VITS model arguments.

    Args:

        num_chars (int):
            Number of characters in the vocabulary. Defaults to 100.

        out_channels (int):
            Number of output channels of the decoder. Defaults to 513.

        spec_segment_size (int):
            Decoder input segment size. Defaults to 32 `(32 * hoplength = waveform length)`.

        hidden_channels (int):
            Number of hidden channels of the model. Defaults to 192.

        hidden_channels_ffn_text_encoder (int):
            Number of hidden channels of the feed-forward layers of the text encoder transformer. Defaults to 256.

        num_heads_text_encoder (int):
            Number of attention heads of the text encoder transformer. Defaults to 2.

        num_layers_text_encoder (int):
            Number of transformer layers in the text encoder. Defaults to 6.

        kernel_size_text_encoder (int):
            Kernel size of the text encoder transformer FFN layers. Defaults to 3.

        dropout_p_text_encoder (float):
            Dropout rate of the text encoder. Defaults to 0.1.

        dropout_p_duration_predictor (float):
            Dropout rate of the duration predictor. Defaults to 0.1.

        kernel_size_posterior_encoder (int):
            Kernel size of the posterior encoder's WaveNet layers. Defaults to 5.

        dilatation_posterior_encoder (int):
            Dilation rate of the posterior encoder's WaveNet layers. Defaults to 1.

        num_layers_posterior_encoder (int):
            Number of posterior encoder's WaveNet layers. Defaults to 16.

        kernel_size_flow (int):
            Kernel size of the Residual Coupling layers of the flow network. Defaults to 5.

        dilatation_flow (int):
            Dilation rate of the Residual Coupling WaveNet layers of the flow network. Defaults to 1.

        num_layers_flow (int):
            Number of Residual Coupling WaveNet layers of the flow network. Defaults to 6.

        resblock_type_decoder (str):
            Type of the residual block in the decoder network. Defaults to "1".

        resblock_kernel_sizes_decoder (List[int]):
            Kernel sizes of the residual blocks in the decoder network. Defaults to `[3, 7, 11]`.

        resblock_dilation_sizes_decoder (List[List[int]]):
            Dilation sizes of the residual blocks in the decoder network. Defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`.

        upsample_rates_decoder (List[int]):
            Upsampling rates for each concecutive upsampling layer in the decoder network. The multiply of these
            values must be equal to the kop length used for computing spectrograms. Defaults to `[8, 8, 2, 2]`.

        upsample_initial_channel_decoder (int):
            Number of hidden channels of the first upsampling convolution layer of the decoder network. Defaults to 512.

        upsample_kernel_sizes_decoder (List[int]):
            Kernel sizes for each upsampling layer of the decoder network. Defaults to `[16, 16, 4, 4]`.

        periods_multi_period_discriminator (List[int]):
            Periods values for Vits Multi-Period Discriminator. Defaults to `[2, 3, 5, 7, 11]`.

        use_sdp (bool):
            Use Stochastic Duration Predictor. Defaults to True.

        noise_scale (float):
            Noise scale used for the sample noise tensor in training. Defaults to 1.0.

        inference_noise_scale (float):
            Noise scale used for the sample noise tensor in inference. Defaults to 0.667.

        length_scale (float):
            Scale factor for the predicted duration values. Smaller values result faster speech. Defaults to 1.

        noise_scale_dp (float):
            Noise scale used by the Stochastic Duration Predictor sample noise in training. Defaults to 1.0.

        inference_noise_scale_dp (float):
            Noise scale for the Stochastic Duration Predictor in inference. Defaults to 0.8.

        max_inference_len (int):
            Maximum inference length to limit the memory use. Defaults to None.

        init_discriminator (bool):
            Initialize the disciminator network if set True. Set False for inference. Defaults to True.

        use_spectral_norm_disriminator (bool):
            Use spectral normalization over weight norm in the discriminator. Defaults to False.

        use_speaker_embedding (bool):
            Enable/Disable speaker embedding for multi-speaker models. Defaults to False.

        num_speakers (int):
            Number of speakers for the speaker embedding layer. Defaults to 0.

        speakers_file (str):
            Path to the speaker mapping file for the Speaker Manager. Defaults to None.

        speaker_embedding_channels (int):
            Number of speaker embedding channels. Defaults to 256.

        use_d_vector_file (bool):
            Enable/Disable the use of d-vectors for multi-speaker training. Defaults to False.

        d_vector_file (str):
            Path to the file including pre-computed speaker embeddings. Defaults to None.

        d_vector_dim (int):
            Number of d-vector channels. Defaults to 0.

        detach_dp_input (bool):
            Detach duration predictor's input from the network for stopping the gradients. Defaults to True.

        use_language_embedding (bool):
            Enable/Disable language embedding for multilingual models. Defaults to False.

        embedded_language_dim (int):
            Number of language embedding channels. Defaults to 4.

        num_languages (int):
            Number of languages for the language embedding layer. Defaults to 0.

        language_ids_file (str):
            Path to the language mapping file for the Language Manager. Defaults to None.

        use_speaker_encoder_as_loss (bool):
            Enable/Disable Speaker Consistency Loss (SCL). Defaults to False.

        speaker_encoder_config_path (str):
            Path to the file speaker encoder config file, to use for SCL. Defaults to "".

        speaker_encoder_model_path (str):
            Path to the file speaker encoder checkpoint file, to use for SCL. Defaults to "".

        condition_dp_on_speaker (bool):
            Condition the duration predictor on the speaker embedding. Defaults to True.

        freeze_encoder (bool):
            Freeze the encoder weigths during training. Defaults to False.

        freeze_DP (bool):
            Freeze the duration predictor weigths during training. Defaults to False.

        freeze_PE (bool):
            Freeze the posterior encoder weigths during training. Defaults to False.

        freeze_flow_encoder (bool):
            Freeze the flow encoder weigths during training. Defaults to False.

        freeze_waveform_decoder (bool):
            Freeze the waveform decoder weigths during training. Defaults to False.

        encoder_sample_rate (int):
            If not None this sample rate will be used for training the Posterior Encoder,
            flow, text_encoder and duration predictor. The decoder part (vocoder) will be
            trained with the `config.audio.sample_rate`. Defaults to None.

        interpolate_z (bool):
            If `encoder_sample_rate` not None and  this parameter True the nearest interpolation
            will be used to upsampling the latent variable z with the sampling rate `encoder_sample_rate`
            to the `config.audio.sample_rate`. If it is False you will need to add extra
            `upsample_rates_decoder` to match the shape. Defaults to True.

        pitch_predictor_hidden_channels (int):
            Number of hidden channels in the pitch predictor. Defaults to 256.

        pitch_predictor_dropout_p (float):
            Dropout rate for the pitch predictor. Defaults to 0.1.

        pitch_predictor_kernel_size (int):
            Kernel size of conv layers in the pitch predictor. Defaults to 3.

        pitch_embedding_kernel_size (int):
            Kernel size of the projection layer in the pitch predictor. Defaults to 3.
        
        use_aligner (bool):
            Whether to use aligner network to learn the text to speech alignment or use pre-computed durations.
            If set False, durations should be computed by `TTS/bin/compute_attention_masks.py` and path to the
            pre-computed durations must be provided to `config.datasets[0].meta_file_attn_mask`. Defaults to True.

    """

    num_chars: int = 100
    out_channels: int = 513
    spec_segment_size: int = 32
    hidden_channels: int = 192
    hidden_channels_ffn_text_encoder: int = 768
    num_heads_text_encoder: int = 2
    num_layers_text_encoder: int = 6
    kernel_size_text_encoder: int = 3
    dropout_p_text_encoder: float = 0.1
    dropout_p_duration_predictor: float = 0.5
    kernel_size_posterior_encoder: int = 5
    dilation_rate_posterior_encoder: int = 1
    num_layers_posterior_encoder: int = 16
    kernel_size_flow: int = 5
    dilation_rate_flow: int = 1
    num_layers_flow: int = 4
    resblock_type_decoder: str = "1"
    resblock_kernel_sizes_decoder: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes_decoder: List[List[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    upsample_rates_decoder: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_initial_channel_decoder: int = 512
    upsample_kernel_sizes_decoder: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    periods_multi_period_discriminator: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    use_sdp: bool = True
    noise_scale: float = 1.0
    inference_noise_scale: float = 0.667
    length_scale: float = 1
    noise_scale_dp: float = 1.0
    inference_noise_scale_dp: float = 1.0
    max_inference_len: int = None
    init_discriminator: bool = True
    use_spectral_norm_disriminator: bool = False
    use_speaker_embedding: bool = False
    num_speakers: int = 0
    speakers_file: str = None
    d_vector_file: str = None
    #MODIFICATION FOR FAST_PITCH
    speaker_embedding_channels: int = 192 #256
    use_d_vector_file: bool = False
    d_vector_dim: int = 0
    detach_dp_input: bool = True
    use_language_embedding: bool = False
    embedded_language_dim: int = 4
    num_languages: int = 0
    language_ids_file: str = None
    use_speaker_encoder_as_loss: bool = False
    speaker_encoder_config_path: str = ""
    speaker_encoder_model_path: str = ""
    condition_dp_on_speaker: bool = True
    #freeze_encoder: bool = False
    #freeze_DP: bool = False
    #freeze_PE: bool = False
    #freeze_flow_decoder: bool = False
    #freeze_waveform_decoder: bool = False
    encoder_sample_rate: int = None
    interpolate_z: bool = True
    reinit_DP: bool = False
    reinit_text_encoder: bool = False
    #ADDITION FOR FAST_PITCH
    use_aligner: bool = True
    pitch_predictor_hidden_channels: int = 256
    pitch_predictor_kernel_size: int = 3
    pitch_predictor_dropout_p: float = 0.1
    pitch_embedding_kernel_size: int = 3
    aligner_out_channels=80
    aligner_hidden_channels=192
    compute_f0: bool = True
    f0_cache_path: str = None
    #ADDITION FOR MODULAR_VITS
    training_phase: int = None


class ModularVits(BaseTTS):
    """VITS TTS model

    Paper::
        https://arxiv.org/pdf/2106.06103.pdf

    Paper Abstract::
        Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel
        sampling have been proposed, but their sample quality does not match that of two-stage TTS systems.
        In this work, we present a parallel endto-end TTS method that generates more natural sounding audio than
        current two-stage models. Our method adopts variational inference augmented with normalizing flows and
        an adversarial training process, which improves the expressive power of generative modeling. We also propose a
        stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the
        uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the
        natural one-to-many relationship in which a text input can be spoken in multiple ways
        with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS)
        on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly
        available TTS systems and achieves a MOS comparable to ground truth.

    Check :class:`TTS.tts.configs.vits_config.VitsConfig` for class arguments.

    Examples:
        >>> from TTS.tts.configs.vits_config import VitsConfig
        >>> from TTS.tts.models.vits import Vits
        >>> config = VitsConfig()
        >>> model = Vits(config)
    """

    def __init__(
        self,
        config: Coqpit,
        ap: "AudioProcessor" = None,
        tokenizer: "TTSTokenizer" = None,
        speaker_manager: SpeakerManager = None,
        language_manager: LanguageManager = None,
    ):

        super().__init__(config, ap, tokenizer, speaker_manager, language_manager)

        self.init_multispeaker(config)
        self.init_multilingual(config)
        self.init_upsampling()

        self.length_scale = self.args.length_scale
        self.noise_scale = self.args.noise_scale
        self.inference_noise_scale = self.args.inference_noise_scale
        self.inference_noise_scale_dp = self.args.inference_noise_scale_dp
        self.noise_scale_dp = self.args.noise_scale_dp
        self.max_inference_len = self.args.max_inference_len
        self.spec_segment_size = self.args.spec_segment_size
        #ADDITION FOR FAST_PITCH
        self.use_aligner = self.args.use_aligner
        self.compute_f0 = self.config.compute_f0
        self.f0_cache_path=self.config.f0_cache_path
        #ADDITION FOR MODULAR_VITS
        self.training_phase=self.config.training_phase
        if self.training_phase==1:
            self.freeze_pitch_predictor=True
            self.freeze_pitch_embedding=False
            self.freeze_pitch_encoding=True
            self.freeze_pitch_conv1d=True
            self.freeze_pitch_aligner=False
            self.freeze_prior_encoder=True
            self.freeze_post_encoder=True
            self.freeze_SDP=True
            self.freeze_flow_decoder=True
            self.freeze_waveform_decoder=True
            self.freeze_disc=True
        elif self.training_phase==2:           
            self.freeze_pitch_predictor=False
            self.freeze_pitch_embedding=True
            self.freeze_pitch_encoding=False
            self.freeze_pitch_conv1d=True
            self.freeze_pitch_aligner=True
            self.freeze_prior_encoder=True
            self.freeze_post_encoder=True
            self.freeze_SDP=True
            self.freeze_flow_decoder=True
            self.freeze_waveform_decoder=True
            self.freeze_disc=True
        elif self.training_phase==3:
            self.freeze_pitch_predictor=True
            self.freeze_pitch_embedding=True
            self.freeze_pitch_encoding=True
            self.freeze_pitch_conv1d=False
            self.freeze_pitch_aligner=True
            self.freeze_prior_encoder=False
            self.freeze_post_encoder=False
            self.freeze_SDP=False
            self.freeze_flow_decoder=False
            self.freeze_waveform_decoder=False
            self.freeze_disc=False
            #we also freeze emb_g, to be adapted for each phase in later implementation
            self.freeze_emb_g=True

        elif self.training_phase==4:
            self.freeze_pitch_predictor=True
            self.freeze_pitch_embedding=True
            self.freeze_pitch_encoding=True
            self.freeze_pitch_conv1d=True
            self.freeze_pitch_aligner=True
            self.freeze_prior_encoder=False
            self.freeze_post_encoder=True
            self.freeze_SDP=True
            self.freeze_flow_decoder=False
            self.freeze_waveform_decoder=False
            self.freeze_disc=False
            #we also freeze emb_g, to be adapted for each phase in later implementation
            self.freeze_emb_g=True
            
            
        elif self.training_phase==5:
            self.freeze_pitch_predictor=True
            self.freeze_pitch_embedding=True
            self.freeze_pitch_encoding=True
            self.freeze_pitch_conv1d=True
            self.freeze_pitch_aligner=True
            self.freeze_prior_encoder=True
            self.freeze_post_encoder=True
            self.freeze_SDP=False
            self.freeze_flow_decoder=False
            self.freeze_waveform_decoder=True
            self.freeze_disc=True
            #we also freeze emb_g, to be adapted for each phase in later implementation
            self.freeze_emb_g=True
            
        elif self.training_phase==6:
            self.freeze_pitch_predictor=True
            self.freeze_pitch_embedding=True
            self.freeze_pitch_encoding=True
            self.freeze_pitch_conv1d=True
            self.freeze_pitch_aligner=True
            self.freeze_prior_encoder=True
            self.freeze_post_encoder=True
            self.freeze_SDP=True
            self.freeze_flow_decoder=True
            self.freeze_waveform_decoder=False
            self.freeze_disc=False
            #we also freeze emb_g, to be adapted for each phase in later implementation
            self.freeze_emb_g=True

        self.text_encoder = TextEncoder(
            self.args.num_chars,
            self.args.hidden_channels,
            self.args.hidden_channels,
            self.args.hidden_channels_ffn_text_encoder,
            self.args.num_heads_text_encoder,
            self.args.num_layers_text_encoder,
            self.args.kernel_size_text_encoder,
            self.args.dropout_p_text_encoder,
            language_emb_dim=self.embedded_language_dim,
        )

        self.posterior_encoder = PosteriorEncoder(
            self.args.out_channels,
            self.args.hidden_channels,
            self.args.hidden_channels,
            kernel_size=self.args.kernel_size_posterior_encoder,
            dilation_rate=self.args.dilation_rate_posterior_encoder,
            num_layers=self.args.num_layers_posterior_encoder,
            cond_channels=self.embedded_speaker_dim,
        )

        self.flow = ResidualCouplingBlocks(
            self.args.hidden_channels,
            self.args.hidden_channels,
            kernel_size=self.args.kernel_size_flow,
            dilation_rate=self.args.dilation_rate_flow,
            num_layers=self.args.num_layers_flow,
            cond_channels=self.embedded_speaker_dim,
        )

        if self.args.use_sdp:
            self.duration_predictor = StochasticDurationPredictor(
                self.args.hidden_channels,
                192,
                3,
                self.args.dropout_p_duration_predictor,
                4,
                cond_channels=self.embedded_speaker_dim if self.args.condition_dp_on_speaker else 0,
                language_emb_dim=self.embedded_language_dim,
            )
        else:
            self.duration_predictor = DurationPredictor(
                self.args.hidden_channels,
                256,
                3,
                self.args.dropout_p_duration_predictor,
                cond_channels=self.embedded_speaker_dim,
                language_emb_dim=self.embedded_language_dim,
            )

        self.waveform_decoder = HifiganGenerator(
            self.args.hidden_channels,
            1,
            self.args.resblock_type_decoder,
            self.args.resblock_dilation_sizes_decoder,
            self.args.resblock_kernel_sizes_decoder,
            self.args.upsample_kernel_sizes_decoder,
            self.args.upsample_initial_channel_decoder,
            self.args.upsample_rates_decoder,
            inference_padding=0,
            cond_channels=self.embedded_speaker_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

        if self.args.init_discriminator:
            self.disc = VitsDiscriminator(
                periods=self.args.periods_multi_period_discriminator,
                use_spectral_norm=self.args.use_spectral_norm_disriminator,
            )
            
        #ADDITION FOR FAST_PITCH
        self.pitch_predictor = DurationPredictor(
        #MODIFICATION FOR FAST_PITCH
            #self.args.hidden_channels + self.embedded_speaker_dim,
            self.args.hidden_channels,
            self.args.pitch_predictor_hidden_channels,
            self.args.pitch_predictor_kernel_size,
            self.args.pitch_predictor_dropout_p,
            #addition for speaker embedding
            cond_channels=self.embedded_speaker_dim,
        )
        self.pitch_conv1d = nn.Conv1d(
            1,
            self.args.hidden_channels,
            kernel_size=self.args.pitch_embedding_kernel_size,
            padding=int((self.args.pitch_embedding_kernel_size - 1) / 2),
            )
        if self.args.use_aligner:
            self.pitch_aligner = AlignmentNetwork(
                in_query_channels=self.args.aligner_out_channels, 
                in_key_channels=self.args.aligner_hidden_channels
            )
            
        self.pitch_text_encoder = TextEncoder(
            self.args.num_chars,
            self.args.hidden_channels,
            self.args.hidden_channels,
            self.args.hidden_channels_ffn_text_encoder,
            self.args.num_heads_text_encoder,
            self.args.num_layers_text_encoder,
            self.args.kernel_size_text_encoder,
            self.args.dropout_p_text_encoder,
            language_emb_dim=self.embedded_language_dim,
        )       

    @property
    def device(self):
        return next(self.parameters()).device

    def init_multispeaker(self, config: Coqpit):
        """Initialize multi-speaker modules of a model. A model can be trained either with a speaker embedding layer
        or with external `d_vectors` computed from a speaker encoder model.

        You must provide a `speaker_manager` at initialization to set up the multi-speaker modules.

        Args:
            config (Coqpit): Model configuration.
            data (List, optional): Dataset items to infer number of speakers. Defaults to None.
        """
        self.embedded_speaker_dim = 0
        self.num_speakers = self.args.num_speakers
        self.audio_transform = None

        if self.speaker_manager:
            self.num_speakers = self.speaker_manager.num_speakers

        if self.args.use_speaker_embedding:
            self._init_speaker_embedding()

        if self.args.use_d_vector_file:
            self._init_d_vector()

        # TODO: make this a function
        if self.args.use_speaker_encoder_as_loss:
            if self.speaker_manager.encoder is None and (
                not self.args.speaker_encoder_model_path or not self.args.speaker_encoder_config_path
            ):
                raise RuntimeError(
                    " [!] To use the speaker consistency loss (SCL) you need to specify speaker_encoder_model_path and speaker_encoder_config_path !!"
                )

            self.speaker_manager.encoder.eval()
            print(" > External Speaker Encoder Loaded !!")

            if (
                hasattr(self.speaker_manager.encoder, "audio_config")
                and self.config.audio.sample_rate != self.speaker_manager.encoder.audio_config["sample_rate"]
            ):
                self.audio_transform = torchaudio.transforms.Resample(
                    orig_freq=self.config.audio.sample_rate,
                    new_freq=self.speaker_manager.encoder.audio_config["sample_rate"],
                )

    def _init_speaker_embedding(self):
        # pylint: disable=attribute-defined-outside-init
        if self.num_speakers > 0:
            print(" > initialization of speaker-embedding layers.")
            self.embedded_speaker_dim = self.args.speaker_embedding_channels
            self.emb_g = nn.Embedding(self.num_speakers, self.embedded_speaker_dim)

    def _init_d_vector(self):
        # pylint: disable=attribute-defined-outside-init
        if hasattr(self, "emb_g"):
            raise ValueError("[!] Speaker embedding layer already initialized before d_vector settings.")
        self.embedded_speaker_dim = self.args.d_vector_dim

    def init_multilingual(self, config: Coqpit):
        """Initialize multilingual modules of a model.

        Args:
            config (Coqpit): Model configuration.
        """
        if self.args.language_ids_file is not None:
            self.language_manager = LanguageManager(language_ids_file_path=config.language_ids_file)

        if self.args.use_language_embedding and self.language_manager:
            print(" > initialization of language-embedding layers.")
            self.num_languages = self.language_manager.num_languages
            self.embedded_language_dim = self.args.embedded_language_dim
            self.emb_l = nn.Embedding(self.num_languages, self.embedded_language_dim)
            torch.nn.init.xavier_uniform_(self.emb_l.weight)
        else:
            self.embedded_language_dim = 0

    def init_upsampling(self):
        """
        Initialize upsampling modules of a model.
        """
        if self.args.encoder_sample_rate:
            self.interpolate_factor = self.config.audio["sample_rate"] / self.args.encoder_sample_rate
            self.audio_resampler = torchaudio.transforms.Resample(
                orig_freq=self.config.audio["sample_rate"], new_freq=self.args.encoder_sample_rate
            )  # pylint: disable=W0201

    def on_epoch_start(self, trainer):  # pylint: disable=W0613
        """Freeze layers at the beginning of an epoch"""
        self._freeze_layers()
        # set the device of speaker encoder
        if self.args.use_speaker_encoder_as_loss:
            self.speaker_manager.encoder = self.speaker_manager.encoder.to(self.device)

    def on_init_end(self, trainer):  # pylint: disable=W0613
        """Reinit layes if needed"""
        if self.args.reinit_DP:
            before_dict = get_module_weights_sum(self.duration_predictor)
            # Applies weights_reset recursively to every submodule of the duration predictor
            self.duration_predictor.apply(fn=weights_reset)
            after_dict = get_module_weights_sum(self.duration_predictor)
            for key, value in after_dict.items():
                if value == before_dict[key]:
                    raise RuntimeError(" [!] The weights of Duration Predictor was not reinit check it !")
            print(" > Duration Predictor was reinit.")

        if self.args.reinit_text_encoder:
            before_dict = get_module_weights_sum(self.text_encoder)
            # Applies weights_reset recursively to every submodule of the duration predictor
            self.text_encoder.apply(fn=weights_reset)
            after_dict = get_module_weights_sum(self.text_encoder)
            for key, value in after_dict.items():
                if value == before_dict[key]:
                    raise RuntimeError(" [!] The weights of Text Encoder was not reinit check it !")
            print(" > Text Encoder was reinit.")

    def get_aux_input(self, aux_input: Dict):
        sid, g, lid, _ = self._set_cond_input(aux_input)
        return {"speaker_ids": sid, "style_wav": None, "d_vectors": g, "language_ids": lid}

    def _freeze_layers(self):
    
        if self.freeze_prior_encoder:
            print("freezing prior encoder")
            for param in self.text_encoder.parameters():
                param.requires_grad = False

            if hasattr(self, "emb_l"):
                for param in self.emb_l.parameters():
                    param.requires_grad = False

        if self.freeze_post_encoder:
            print("freezing post encoder")
            for param in self.posterior_encoder.parameters():
                param.requires_grad = False

        if self.freeze_SDP:
            print("freezing SDP")
            for param in self.duration_predictor.parameters():
                param.requires_grad = False

        if self.freeze_flow_decoder:
            print("freezing flow decoder")
            for param in self.flow.parameters():
                param.requires_grad = False

        if self.freeze_waveform_decoder:
            print("freezing waveform decoder")
            for param in self.waveform_decoder.parameters():
                param.requires_grad = False
        
        if self.freeze_pitch_predictor:
            print("freezing pitch predictor")
            for param in self.pitch_predictor.parameters():
                param.requires_grad = False
        
        if self.freeze_pitch_embedding:
            print("freezing pitch text embedder")
            for name, param in self.pitch_text_encoder.named_parameters():
                if name=="emb.weight":
                    param.requires_grad=False
                
        if self.freeze_pitch_encoding:
            print("freezing pitch text encoder")
            for name, param in self.pitch_text_encoder.named_parameters():
                if name!="emb.weight":
                    param.requires_grad=False
        
        if self.freeze_pitch_conv1d:
            print("freezing pitch conv1d")
            for param in self.pitch_conv1d.parameters():
                param.requires_grad = False
        
        if self.freeze_pitch_aligner:
            print("freezing pitch aligner")
            for param in self.pitch_aligner.parameters():
                param.requires_grad = False
                
        if self.freeze_disc:
            print("freezing discriminator")
            for param in self.disc.parameters():
                param.requires_grad = False

        #we also freeze emb_g, to be adapted for each phase in later implementation
        if self.freeze_emb_g:
            print("freezing emb_g")
            for param in self.emb_g.parameters():
                param.requires_grad = False
                
    @staticmethod
    def _set_cond_input(aux_input: Dict):
        """Set the speaker conditioning input based on the multi-speaker mode."""
        sid, g, lid, durations = None, None, None, None
        if "speaker_ids" in aux_input and aux_input["speaker_ids"] is not None:
            sid = aux_input["speaker_ids"]
            if sid.ndim == 0:
                sid = sid.unsqueeze_(0)
        if "d_vectors" in aux_input and aux_input["d_vectors"] is not None:
            g = F.normalize(aux_input["d_vectors"]).unsqueeze(-1)
            if g.ndim == 2:
                g = g.unsqueeze_(0)

        if "language_ids" in aux_input and aux_input["language_ids"] is not None:
            lid = aux_input["language_ids"]
            if lid.ndim == 0:
                lid = lid.unsqueeze_(0)

        if "durations" in aux_input and aux_input["durations"] is not None:
            durations = aux_input["durations"]

        return sid, g, lid, durations

    def _set_speaker_input(self, aux_input: Dict):
        d_vectors = aux_input.get("d_vectors", None)
        speaker_ids = aux_input.get("speaker_ids", None)

        if d_vectors is not None and speaker_ids is not None:
            raise ValueError("[!] Cannot use d-vectors and speaker-ids together.")

        if speaker_ids is not None and not hasattr(self, "emb_g"):
            raise ValueError("[!] Cannot use speaker-ids without enabling speaker embedding.")

        g = speaker_ids if speaker_ids is not None else d_vectors
        return g

    def forward_mas(self, outputs, z_p, m_p, logs_p, x, x_mask, y_mask, g, lang_emb):
        # find the alignment path
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        with torch.no_grad():
            o_scale = torch.exp(-2 * logs_p)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1]).unsqueeze(-1)  # [b, t, 1]
            logp2 = torch.einsum("klm, kln -> kmn", [o_scale, -0.5 * (z_p**2)])
            logp3 = torch.einsum("klm, kln -> kmn", [m_p * o_scale, z_p])
            logp4 = torch.sum(-0.5 * (m_p**2) * o_scale, [1]).unsqueeze(-1)  # [b, t, 1]
            logp = logp2 + logp3 + logp1 + logp4
            attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()  # [b, 1, t, t']

        # duration predictor
        attn_durations = attn.sum(3)
        if self.args.use_sdp:
            loss_duration = self.duration_predictor(
                x.detach() if self.args.detach_dp_input else x,
                x_mask,
                attn_durations,
                g=g.detach() if self.args.detach_dp_input and g is not None else g,
                lang_emb=lang_emb.detach() if self.args.detach_dp_input and lang_emb is not None else lang_emb,
            )
            loss_duration = loss_duration / torch.sum(x_mask)
        else:
            attn_log_durations = torch.log(attn_durations + 1e-6) * x_mask
            log_durations = self.duration_predictor(
                x.detach() if self.args.detach_dp_input else x,
                x_mask,
                g=g.detach() if self.args.detach_dp_input and g is not None else g,
                lang_emb=lang_emb.detach() if self.args.detach_dp_input and lang_emb is not None else lang_emb,
            )
            loss_duration = torch.sum((log_durations - attn_log_durations) ** 2, [1, 2]) / torch.sum(x_mask)
        outputs["loss_duration"] = loss_duration
        return outputs, attn
        
    #ADDITION FOR FAST_PITCH
    def _forward_pitch_predictor(
        self,
        o_en: torch.FloatTensor,
        x_mask: torch.IntTensor,
        pitch: torch.FloatTensor = None,
        dr: torch.IntTensor = None,
        #ADDITION FOR MULTISPEAKERS
        g: torch.FloatTensor = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Pitch predictor forward pass.

        1. Predict pitch from encoder outputs.
        2. In training - Compute average pitch values for each input character from the ground truth pitch values.
        3. Embed average pitch values.

        Args:
            o_en (torch.FloatTensor): Encoder output.
            x_mask (torch.IntTensor): Input sequence mask.
            pitch (torch.FloatTensor, optional): Ground truth pitch values. Defaults to None.
            dr (torch.IntTensor, optional): Ground truth durations. Defaults to None.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: Pitch embedding, pitch prediction.

        Shapes:
            - o_en: :math:`(B, C, T_{en})`
            - x_mask: :math:`(B, 1, T_{en})`
            - pitch: :math:`(B, 1, T_{de})`
            - dr: :math:`(B, T_{en})`
        """
        o_pitch=self.pitch_predictor(
            o_en,
            x_mask,
            g=g.detach() if self.args.detach_dp_input and g is not None else g
        )
        
        if pitch is not None:
            avg_pitch = average_over_durations(pitch, dr)
            o_pitch_emb = self.pitch_conv1d(avg_pitch)
            return o_pitch_emb, o_pitch, avg_pitch
        o_pitch_emb = self.pitch_conv1d(o_pitch)
        return o_pitch_emb, o_pitch

    #ADDITION FOR FAST_PITCH
    def _forward_pitch_aligner(
        self, x: torch.FloatTensor, y: torch.FloatTensor, x_mask: torch.IntTensor, y_mask: torch.IntTensor
    ) -> Tuple[torch.IntTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Aligner forward pass.

        1. Compute a mask to apply to the attention map.
        2. Run the alignment network.
        3. Apply MAS to compute the hard alignment map.
        4. Compute the durations from the hard alignment map.

        Args:
            x (torch.FloatTensor): Input sequence.
            y (torch.FloatTensor): Output sequence.
            x_mask (torch.IntTensor): Input sequence mask.
            y_mask (torch.IntTensor): Output sequence mask.

        Returns:
            Tuple[torch.IntTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
                Durations from the hard alignment map, soft alignment potentials, log scale alignment potentials,
                hard alignment map.

        Shapes:
            - x: :math:`[B, T_en, C_en]`
            - y: :math:`[B, T_de, C_de]`
            - x_mask: :math:`[B, 1, T_en]`
            - y_mask: :math:`[B, 1, T_de]`

            - o_alignment_dur: :math:`[B, T_en]`
            - alignment_soft: :math:`[B, T_en, T_de]`
            - alignment_logprob: :math:`[B, 1, T_de, T_en]`
            - alignment_mas: :math:`[B, T_en, T_de]`
        """
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        alignment_soft, alignment_logprob = self.pitch_aligner(y.transpose(1, 2), x.transpose(1, 2), x_mask, None)
        alignment_mas = maximum_path(
            alignment_soft.squeeze(1).transpose(1, 2).contiguous(), attn_mask.squeeze(1).contiguous()
        )
        o_alignment_dur = torch.sum(alignment_mas, -1).int()
        alignment_soft = alignment_soft.squeeze(1).transpose(1, 2)
        return o_alignment_dur, alignment_soft, alignment_logprob, alignment_mas

  
    def upsampling_z(self, z, slice_ids=None, y_lengths=None, y_mask=None):
        spec_segment_size = self.spec_segment_size
        if self.args.encoder_sample_rate:
            # recompute the slices and spec_segment_size if needed
            slice_ids = slice_ids * int(self.interpolate_factor) if slice_ids is not None else slice_ids
            spec_segment_size = spec_segment_size * int(self.interpolate_factor)
            # interpolate z if needed
            if self.args.interpolate_z:
                z = torch.nn.functional.interpolate(z, scale_factor=[self.interpolate_factor], mode="linear").squeeze(0)
                # recompute the mask if needed
                if y_lengths is not None and y_mask is not None:
                    y_mask = (
                        sequence_mask(y_lengths * self.interpolate_factor, None).to(y_mask.dtype).unsqueeze(1)
                    )  # [B, 1, T_dec_resampled]

        return z, spec_segment_size, slice_ids, y_mask
    
    
    def forward(  # pylint: disable=dangerous-default-value
        self,
        training_phase: int,
        **kwargs
    ) -> Dict:
        if training_phase==1:
            return self.forward_phase_1(
                x=kwargs.get('x'),
                x_lengths=kwargs.get('x_lengths'),
                mel_input=kwargs.get('mel_input'),
                mel_lens=kwargs.get('mel_lens'),
                aux_input=kwargs.get('aux_input')
            )
            
        elif training_phase==2:
            return self.forward_phase_2(
                    x=kwargs.get('x'),
                    x_lengths=kwargs.get('x_lengths'),
                    pitch=kwargs.get('pitch'),
                    mel_input=kwargs.get('mel_input'),
                    mel_lens=kwargs.get('mel_lens'),
                    aux_input=kwargs.get('aux_input')
            )  
            
        elif training_phase==3:
            return self.forward_phase_3(
                    x=kwargs.get('x'),
                    x_lengths=kwargs.get('x_lengths'),
                    y = kwargs.get('y'),
                    y_lengths= kwargs.get('y_lengths'),
                    waveform = kwargs.get('waveform'),
                    mel_input=kwargs.get('mel_input'),
                    mel_lens=kwargs.get('mel_lens'),
                    aux_input=kwargs.get('aux_input')
            )
            
        elif training_phase==4:
            return self.forward_phase_4(
                    x=kwargs.get('x'),
                    x_lengths=kwargs.get('x_lengths'),
                    y = kwargs.get('y'),
                    y_lengths= kwargs.get('y_lengths'),
                    waveform = kwargs.get('waveform'),
                    mel_input=kwargs.get('mel_input'),
                    mel_lens=kwargs.get('mel_lens'),
                    aux_input=kwargs.get('aux_input'))
        
        elif training_phase==5:
            return self.forward_phase_5(
                    x=kwargs.get('x'),
                    x_lengths=kwargs.get('x_lengths'),
                    y = kwargs.get('y'),
                    y_lengths= kwargs.get('y_lengths'),
                    aux_input=kwargs.get('aux_input')
            )        

        elif training_phase==6:
            return self.forward_phase_6(
                    y = kwargs.get('y'),
                    y_lengths= kwargs.get('y_lengths'),
                    waveform = kwargs.get('waveform'),
                    aux_input=kwargs.get('aux_input')
            )     

        
        raise ValueError(" [!] Unexpected training_phase {} in forward function.".format(training_phase))
        

    
    #Modular_vits forward pass for the phase 1
    def forward_phase_1(self,x,x_lengths,mel_input,mel_lens,aux_input):
        sid, g, lid, _ = self._set_cond_input(aux_input)

        # language embedding
        lang_emb = None
        if self.args.use_language_embedding and lid is not None:
            lang_emb = self.emb_l(lid).unsqueeze(-1)
      
        #Text Embedding for pitch aligner purpose only
        #x_mask, x_emb = self.pitch_text_embedder(x, x_lengths, lang_emb=lang_emb)        
        _, _, _, x_mask, x_emb = self.pitch_text_encoder(x, x_lengths, lang_emb=lang_emb)
        
        
        # pitch duration calculation with generic aligner
        if self.use_aligner:
            mel_mask = torch.unsqueeze(sequence_mask(mel_lens, mel_input.shape[1]), 1).float()
            o_alignment_dur, alignment_soft, alignment_logprob, alignment_mas = self._forward_pitch_aligner(
                x_emb, mel_input, x_mask, mel_mask
            )
            alignment_soft = alignment_soft.transpose(1, 2)
            alignment_mas = alignment_mas.transpose(1, 2)
        else:
            raise Exception("No aligner is used to compute pitch duration. Make sure use_aligner=True")
        outputs={
            "o_alignment_dur": o_alignment_dur,
            "alignment_logprob": alignment_logprob,
            "alignment_soft": alignment_soft,
            "alignment_mas": alignment_mas,
            }
        return outputs        

    #Modular_vits forward pass for the phase 2
    def forward_phase_2(
        self,
        x: torch.tensor,
        x_lengths: torch.tensor,
        mel_input: torch.FloatTensor,
        mel_lens: torch.tensor,
        pitch: torch.FloatTensor,
        aux_input={"d_vectors": None, "speaker_ids": None, "language_ids": None},
    ) -> Dict:
    
        sid, g, lid, _ = self._set_cond_input(aux_input)
        # speaker embedding
        if self.args.use_speaker_embedding and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]

        # language embedding
        lang_emb = None
        if self.args.use_language_embedding and lid is not None:
            lang_emb = self.emb_l(lid).unsqueeze(-1)
      
        #Regular Text Encoding as per core vits but dedicated to pitch
        x, m_p, logs_p, x_mask, x_emb = self.pitch_text_encoder(x, x_lengths, lang_emb=lang_emb)
        #x_mask, x_emb = self.pitch_text_embedder(x, x_lengths, lang_emb=lang_emb)
        
        # pitch duration calculation with generic aligner
        mel_mask = torch.unsqueeze(sequence_mask(mel_lens, mel_input.shape[1]), 1).float()
        o_alignment_dur, alignment_soft, alignment_logprob, alignment_mas = self._forward_pitch_aligner(
            x_emb, mel_input, x_mask, mel_mask
        )

        #Pitch predictor pass
        o_pitch_emb, o_pitch, avg_pitch = self._forward_pitch_predictor(x, x_mask, pitch, o_alignment_dur,g=g)
    
        return {"pitch_avg": o_pitch,"pitch_avg_gt": avg_pitch}


    
    #Modular_vits forward pass for the phase 3
    def forward_phase_3(  # pylint: disable=dangerous-default-value
        self,
        x: torch.tensor,
        x_lengths: torch.tensor,
        y: torch.tensor,
        y_lengths: torch.tensor,
        waveform: torch.tensor,
        #ADDITION FOR FAST_PITCH
        dr: torch.IntTensor = None,
        mel_input: torch.FloatTensor = None,
        mel_lens: torch.tensor = None,
        aux_input={"d_vectors": None, "speaker_ids": None, "language_ids": None},
    ) -> Dict:
        """Forward pass of the model.

        Args:
            x (torch.tensor): Batch of input character sequence IDs.
            x_lengths (torch.tensor): Batch of input character sequence lengths.
            y (torch.tensor): Batch of input spectrograms.
            y_lengths (torch.tensor): Batch of input spectrogram lengths.
            waveform (torch.tensor): Batch of ground truth waveforms per sample.
            aux_input (dict, optional): Auxiliary inputs for multi-speaker and multi-lingual training.
                Defaults to {"d_vectors": None, "speaker_ids": None, "language_ids": None}.

        Returns:
            Dict: model outputs keyed by the output name.

        Shapes:
            - x: :math:`[B, T_seq]`
            - x_lengths: :math:`[B]`
            - y: :math:`[B, C, T_spec]`
            - y_lengths: :math:`[B]`
            - waveform: :math:`[B, 1, T_wav]`
            - d_vectors: :math:`[B, C, 1]`
            - speaker_ids: :math:`[B]`
            - language_ids: :math:`[B]`

        Return Shapes:
            - model_outputs: :math:`[B, 1, T_wav]`
            - alignments: :math:`[B, T_seq, T_dec]`
            - z: :math:`[B, C, T_dec]`
            - z_p: :math:`[B, C, T_dec]`
            - m_p: :math:`[B, C, T_dec]`
            - logs_p: :math:`[B, C, T_dec]`
            - m_q: :math:`[B, C, T_dec]`
            - logs_q: :math:`[B, C, T_dec]`
            - waveform_seg: :math:`[B, 1, spec_seg_size * hop_length]`
            - gt_spk_emb: :math:`[B, 1, speaker_encoder.proj_dim]`
            - syn_spk_emb: :math:`[B, 1, speaker_encoder.proj_dim]`
        """
        outputs = {}
        sid, g, lid, _ = self._set_cond_input(aux_input)
        # speaker embedding
        if self.args.use_speaker_embedding and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]

        # language embedding
        lang_emb = None
        if self.args.use_language_embedding and lid is not None:
            lang_emb = self.emb_l(lid).unsqueeze(-1)
            
            
        #COMPUTATION OF THE PITCH EMBEDDING TO BE PASSED TO CORE VITS       
        #Text Embedding for pitch aligner purpose only
        #x_mask, x_emb = self.pitch_text_embedder(x, x_lengths, lang_emb=lang_emb)        
        o_p_e, m_p, logs_p, x_mask, x_emb = self.pitch_text_encoder(x, x_lengths, lang_emb=lang_emb)
        #Pitch predictor pass
        o_pitch_emb, o_pitch= self._forward_pitch_predictor(
            o_en=o_p_e, 
            x_mask=x_mask, 
            g=g
        )
               
        #CORE VITS COMPUTATION
        # Text Encoding
        x, m_p, logs_p, x_mask, x_emb = self.text_encoder(x, x_lengths, lang_emb=lang_emb)
        #Adding the inputs from pitch embedding
        x = x + o_pitch_emb
       
        # posterior encoder
        z, m_q, logs_q, y_mask = self.posterior_encoder(y, y_lengths, g=g)

        # flow layers
        z_p = self.flow(z, y_mask, g=g)

        # duration predictor
        outputs, attn = self.forward_mas(outputs, z_p, m_p, logs_p, x, x_mask, y_mask, g=g, lang_emb=lang_emb)

        # expand prior
        m_p = torch.einsum("klmn, kjm -> kjn", [attn, m_p])
        logs_p = torch.einsum("klmn, kjm -> kjn", [attn, logs_p])

        # select a random feature segment for the waveform decoder
        z_slice, slice_ids = rand_segments(z, y_lengths, self.spec_segment_size, let_short_samples=True, pad_short=True)

        # interpolate z if needed
        z_slice, spec_segment_size, slice_ids, _ = self.upsampling_z(z_slice, slice_ids=slice_ids)

        o = self.waveform_decoder(z_slice, g=g)

        wav_seg = segment(
            waveform,
            slice_ids * self.config.audio.hop_length,
            spec_segment_size * self.config.audio.hop_length,
            pad_short=True,
        )

        if self.args.use_speaker_encoder_as_loss and self.speaker_manager.encoder is not None:
            # concate generated and GT waveforms
            wavs_batch = torch.cat((wav_seg, o), dim=0)

            # resample audio to speaker encoder sample_rate
            # pylint: disable=W0105
            if self.audio_transform is not None:
                wavs_batch = self.audio_transform(wavs_batch)

            pred_embs = self.speaker_manager.encoder.forward(wavs_batch, l2_norm=True)

            # split generated and GT speaker embeddings
            gt_spk_emb, syn_spk_emb = torch.chunk(pred_embs, 2, dim=0)
        else:
            gt_spk_emb, syn_spk_emb = None, None

        outputs.update(
            {
                "model_outputs": o,
                "alignments": attn.squeeze(1),
                "m_p": m_p,
                "logs_p": logs_p,
                "z": z,
                "z_p": z_p,
                "m_q": m_q,
                "logs_q": logs_q,
                "waveform_seg": wav_seg,
                "gt_spk_emb": gt_spk_emb,
                "syn_spk_emb": syn_spk_emb,
                "slice_ids": slice_ids,
            }
        )
        return outputs
        
    #Modular_vits forward pass for the phase 4
    def forward_phase_4(  # pylint: disable=dangerous-default-value
        self,
        x: torch.tensor,
        x_lengths: torch.tensor,
        y: torch.tensor,
        y_lengths: torch.tensor,
        waveform: torch.tensor,
        #ADDITION FOR FAST_PITCH
        dr: torch.IntTensor = None,
        mel_input: torch.FloatTensor = None,
        mel_lens: torch.tensor = None,
        aux_input={"d_vectors": None, "speaker_ids": None, "language_ids": None},
    ) -> Dict:   
    
        sid, g, lid, _ = self._set_cond_input(aux_input)
        # speaker embedding
        if self.args.use_speaker_embedding and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]

        # language embedding
        lang_emb = None
        if self.args.use_language_embedding and lid is not None:
            lang_emb = self.emb_l(lid).unsqueeze(-1)
            
            
        #COMPUTATION OF THE PITCH EMBEDDING TO BE PASSED TO CORE VITS       
        #Text Encoding for pitch aligner purpose only
        o_p_e, m_p, logs_p, x_mask, x_emb = self.pitch_text_encoder(x, x_lengths, lang_emb=lang_emb)
        #Pitch predictor pass
        o_pitch_emb, o_pitch= self._forward_pitch_predictor(
            o_en=o_p_e, 
            x_mask=x_mask, 
            g=g
        )
               
        #CORE VITS COMPUTATION
        # Text Encoding
        x, m_p, logs_p, x_mask, x_emb = self.text_encoder(x, x_lengths, lang_emb=lang_emb)
        #Adding the inputs from pitch embedding
        x = x + o_pitch_emb
        
        #SDP Calculation
        if dr is None:
            if self.args.use_sdp:
                logw = self.duration_predictor(
                    x,
                    x_mask,
                    g=g if self.args.condition_dp_on_speaker else None,
                    reverse=True,
                    noise_scale=self.inference_noise_scale_dp,
                    lang_emb=lang_emb,
                )
            else:
                logw = self.duration_predictor(
                    x, x_mask, g=g if self.args.condition_dp_on_speaker else None, lang_emb=lang_emb
                )
            w = torch.exp(logw) * x_mask * self.length_scale
        else:
            assert durations.shape[-1] == x.shape[-1]
            w = durations.unsqueeze(0)
            
        
        #MAS AND Z COMPUTATION
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype).unsqueeze(1)  # [B, 1, T_dec]

        attn_mask = x_mask * y_mask.transpose(1, 2)  # [B, 1, T_enc] * [B, T_dec, 1]
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1).transpose(1, 2))

        m_p = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * self.inference_noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        # upsampling if needed
        #z_up, _, _, y_mask = self.upsampling_z(z, y_lengths=y_lengths, y_mask=y_mask)
       
        #SLICES FOR WAVEFORM SEGMENTS
        # select a random feature segment for the waveform decoder
        z_slice, slice_ids = rand_segments(z, y_lengths, self.spec_segment_size, let_short_samples=True, pad_short=True)

        # interpolate z if needed
        z_slice, spec_segment_size, slice_ids, y_mask = self.upsampling_z(z_slice, slice_ids=slice_ids, y_lengths=y_lengths, y_mask=y_mask)
        
        #WAVEFORM DECODER
        o = self.waveform_decoder(z_slice, g=g)
        
        wav_seg = segment(
            waveform,
            slice_ids * self.config.audio.hop_length,
            spec_segment_size * self.config.audio.hop_length,
            pad_short=True,
        ) 
        
        
        if self.args.use_speaker_encoder_as_loss and self.speaker_manager.encoder is not None:
            # concate generated and GT waveforms
            wavs_batch = torch.cat((wav_seg, o), dim=0)

            # resample audio to speaker encoder sample_rate
            # pylint: disable=W0105
            if self.audio_transform is not None:
                wavs_batch = self.audio_transform(wavs_batch)

            pred_embs = self.speaker_manager.encoder.forward(wavs_batch, l2_norm=True)

            # split generated and GT speaker embeddings
            gt_spk_emb, syn_spk_emb = torch.chunk(pred_embs, 2, dim=0)
        else:
            gt_spk_emb, syn_spk_emb = None, None
        
        outputs = {
            "model_outputs": o,
            "waveform_seg": wav_seg,
            "gt_spk_emb": gt_spk_emb,
            "syn_spk_emb": syn_spk_emb,
            "slice_ids" : slice_ids,
        }
        return outputs
    

    #Modular_vits forward pass for the phase 5 (inspired by forward of phase 3
    def forward_phase_5(  # pylint: disable=dangerous-default-value
        self,
        x: torch.tensor,
        x_lengths: torch.tensor,
        y: torch.tensor,
        y_lengths: torch.tensor,
        aux_input={"d_vectors": None, "speaker_ids": None, "language_ids": None},
    ) -> Dict:
        outputs = {}
        sid, g, lid, _ = self._set_cond_input(aux_input)
        # speaker embedding
        if self.args.use_speaker_embedding and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]

        # language embedding
        lang_emb = None
        if self.args.use_language_embedding and lid is not None:
            lang_emb = self.emb_l(lid).unsqueeze(-1)
            
            
        #COMPUTATION OF THE PITCH EMBEDDING TO BE PASSED TO CORE VITS       
        #Text Embedding for pitch aligner purpose only
        #x_mask, x_emb = self.pitch_text_embedder(x, x_lengths, lang_emb=lang_emb)        
        o_p_e, m_p, logs_p, x_mask, x_emb = self.pitch_text_encoder(x, x_lengths, lang_emb=lang_emb)
        #Pitch predictor pass
        o_pitch_emb, o_pitch= self._forward_pitch_predictor(
            o_en=o_p_e, 
            x_mask=x_mask, 
            g=g
        )
               
        #CORE VITS COMPUTATION
        # Text Encoding
        x, m_p, logs_p, x_mask, x_emb = self.text_encoder(x, x_lengths, lang_emb=lang_emb)
        #Adding the inputs from pitch embedding
        x = x + o_pitch_emb
       
        # posterior encoder
        z, m_q, logs_q, y_mask = self.posterior_encoder(y, y_lengths, g=g)

        # flow layers
        z_p = self.flow(z, y_mask, g=g)

        # duration predictor
        outputs, attn = self.forward_mas(outputs, z_p, m_p, logs_p, x, x_mask, y_mask, g=g, lang_emb=lang_emb)

        # expand prior
        m_p = torch.einsum("klmn, kjm -> kjn", [attn, m_p])
        logs_p = torch.einsum("klmn, kjm -> kjn", [attn, logs_p])

        if self.args.use_speaker_encoder_as_loss and self.speaker_manager.encoder is not None:
            raise Exception("using speaker encoder as loss is not implemented for this phase")
        else:
            gt_spk_emb, syn_spk_emb = None, None

        outputs.update(
            {
                "alignments": attn.squeeze(1),
                "m_p": m_p,
                "logs_p": logs_p,
                "z_p": z_p,
                "logs_q": logs_q,
                "gt_spk_emb": gt_spk_emb,
                "syn_spk_emb": syn_spk_emb,
            }
        )
        return outputs



    #Modular_vits forward pass for the phase 6
    def forward_phase_6(  # pylint: disable=dangerous-default-value
        self,
        y: torch.tensor,
        y_lengths: torch.tensor,
        waveform: torch.tensor,
        aux_input={"d_vectors": None, "speaker_ids": None, "language_ids": None},
    ) -> Dict:
        outputs = {}
        sid, g, lid, _ = self._set_cond_input(aux_input)
        # speaker embedding
        if self.args.use_speaker_embedding and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]

        # language embedding
        lang_emb = None
        if self.args.use_language_embedding and lid is not None:
            lang_emb = self.emb_l(lid).unsqueeze(-1)
            
        # posterior encoder
        z, _, _, _ = self.posterior_encoder(y, y_lengths, g=g)

        # select a random feature segment for the waveform decoder
        z_slice, slice_ids = rand_segments(z, y_lengths, self.spec_segment_size, let_short_samples=True, pad_short=True)

        # interpolate z if needed
        z_slice, spec_segment_size, slice_ids, _ = self.upsampling_z(z_slice, slice_ids=slice_ids)

        o = self.waveform_decoder(z_slice, g=g)

        wav_seg = segment(
            waveform,
            slice_ids * self.config.audio.hop_length,
            spec_segment_size * self.config.audio.hop_length,
            pad_short=True,
        )

        if self.args.use_speaker_encoder_as_loss and self.speaker_manager.encoder is not None:
            # concate generated and GT waveforms
            wavs_batch = torch.cat((wav_seg, o), dim=0)

            # resample audio to speaker encoder sample_rate
            # pylint: disable=W0105
            if self.audio_transform is not None:
                wavs_batch = self.audio_transform(wavs_batch)

            pred_embs = self.speaker_manager.encoder.forward(wavs_batch, l2_norm=True)

            # split generated and GT speaker embeddings
            gt_spk_emb, syn_spk_emb = torch.chunk(pred_embs, 2, dim=0)
        else:
            gt_spk_emb, syn_spk_emb = None, None

        outputs.update(
            {
                "model_outputs": o,
                "waveform_seg": wav_seg,
                "gt_spk_emb": gt_spk_emb,
                "syn_spk_emb": syn_spk_emb,
                "slice_ids": slice_ids,
            }
        )
        return outputs

    @staticmethod
    def _set_x_lengths(x, aux_input):
        if "x_lengths" in aux_input and aux_input["x_lengths"] is not None:
            return aux_input["x_lengths"]
        return torch.tensor(x.shape[1:2]).to(x.device)

    @torch.no_grad()
    def inference(
        self,
        x,
        aux_input={"x_lengths": None, "d_vectors": None, "speaker_ids": None, "language_ids": None, "durations": None},
    ):  # pylint: disable=dangerous-default-value
        """
        Note:
            To run in batch mode, provide `x_lengths` else model assumes that the batch size is 1.

        Shapes:
            - x: :math:`[B, T_seq]`
            - x_lengths: :math:`[B]`
            - d_vectors: :math:`[B, C]`
            - speaker_ids: :math:`[B]`

        Return Shapes:
            - model_outputs: :math:`[B, 1, T_wav]`
            - alignments: :math:`[B, T_seq, T_dec]`
            - z: :math:`[B, C, T_dec]`
            - z_p: :math:`[B, C, T_dec]`
            - m_p: :math:`[B, C, T_dec]`
            - logs_p: :math:`[B, C, T_dec]`
        """
        sid, g, lid, durations = self._set_cond_input(aux_input)
        x_lengths = self._set_x_lengths(x, aux_input)

        # speaker embedding
        if self.args.use_speaker_embedding and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)

        # language embedding
        lang_emb = None
        if self.args.use_language_embedding and lid is not None:
            lang_emb = self.emb_l(lid).unsqueeze(-1)


        #COMPUTATION OF THE PITCH EMBEDDING TO BE PASSED TO CORE VITS       
        #Text Encoding for pitch aligner purpose only
        o_p_e, m_p, logs_p, x_mask, x_emb = self.pitch_text_encoder(x, x_lengths, lang_emb=lang_emb)
        #Pitch predictor pass
        o_pitch_emb, o_pitch= self._forward_pitch_predictor(
            o_en=o_p_e, 
            x_mask=x_mask, 
            g=g
        )
               
        #CORE VITS COMPUTATION
        # Text Encoding
        x, m_p, logs_p, x_mask, x_emb = self.text_encoder(x, x_lengths, lang_emb=lang_emb)
        #Adding the inputs from pitch embedding
        x = x + o_pitch_emb


        if durations is None:
            if self.args.use_sdp:
                logw = self.duration_predictor(
                    x,
                    x_mask,
                    g=g if self.args.condition_dp_on_speaker else None,
                    reverse=True,
                    noise_scale=self.inference_noise_scale_dp,
                    lang_emb=lang_emb,
                )
            else:
                logw = self.duration_predictor(
                    x, x_mask, g=g if self.args.condition_dp_on_speaker else None, lang_emb=lang_emb
                )
            w = torch.exp(logw) * x_mask * self.length_scale
        else:
            assert durations.shape[-1] == x.shape[-1]
            w = durations.unsqueeze(0)

        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype).unsqueeze(1)  # [B, 1, T_dec]

        attn_mask = x_mask * y_mask.transpose(1, 2)  # [B, 1, T_enc] * [B, T_dec, 1]
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1).transpose(1, 2))

        m_p = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * self.inference_noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        # upsampling if needed
        z, _, _, y_mask = self.upsampling_z(z, y_lengths=y_lengths, y_mask=y_mask)

        o = self.waveform_decoder((z * y_mask)[:, :, : self.max_inference_len], g=g)

        outputs = {
            "model_outputs": o,
            "alignments": attn.squeeze(1),
            "durations": w_ceil,
            "z": z,
            "z_p": z_p,
            "m_p": m_p,
            "logs_p": logs_p,
            "y_mask": y_mask,
            #ADDITION FOR FAST_PITCH
            "pitch": o_pitch
        }
        return outputs

    @torch.no_grad()
    def inference_voice_conversion(
        self, reference_wav, speaker_id=None, d_vector=None, reference_speaker_id=None, reference_d_vector=None
    ):
        """Inference for voice conversion

        Args:
            reference_wav (Tensor): Reference wavform. Tensor of shape [B, T]
            speaker_id (Tensor): speaker_id of the target speaker. Tensor of shape [B]
            d_vector (Tensor): d_vector embedding of target speaker. Tensor of shape `[B, C]`
            reference_speaker_id (Tensor): speaker_id of the reference_wav speaker. Tensor of shape [B]
            reference_d_vector (Tensor): d_vector embedding of the reference_wav speaker. Tensor of shape `[B, C]`
        """
        # compute spectrograms
        y = wav_to_spec(
            reference_wav,
            self.config.audio.fft_size,
            self.config.audio.hop_length,
            self.config.audio.win_length,
            center=False,
        )
        y_lengths = torch.tensor([y.size(-1)]).to(y.device)
        speaker_cond_src = reference_speaker_id if reference_speaker_id is not None else reference_d_vector
        speaker_cond_tgt = speaker_id if speaker_id is not None else d_vector
        wav, _, _ = self.voice_conversion(y, y_lengths, speaker_cond_src, speaker_cond_tgt)
        return wav

    def voice_conversion(self, y, y_lengths, speaker_cond_src, speaker_cond_tgt):
        """Forward pass for voice conversion

        TODO: create an end-point for voice conversion

        Args:
            y (Tensor): Reference spectrograms. Tensor of shape [B, T, C]
            y_lengths (Tensor): Length of each reference spectrogram. Tensor of shape [B]
            speaker_cond_src (Tensor): Reference speaker ID. Tensor of shape [B,]
            speaker_cond_tgt (Tensor): Target speaker ID. Tensor of shape [B,]
        """
        assert self.num_speakers > 0, "num_speakers have to be larger than 0."
        # speaker embedding
        if self.args.use_speaker_embedding and not self.args.use_d_vector_file:
            g_src = self.emb_g(speaker_cond_src).unsqueeze(-1)
            g_tgt = self.emb_g(speaker_cond_tgt).unsqueeze(-1)
        elif not self.args.use_speaker_embedding and self.args.use_d_vector_file:
            g_src = F.normalize(speaker_cond_src).unsqueeze(-1)
            g_tgt = F.normalize(speaker_cond_tgt).unsqueeze(-1)
        else:
            raise RuntimeError(" [!] Voice conversion is only supported on multi-speaker models.")

        z, _, _, y_mask = self.posterior_encoder(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.waveform_decoder(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)

    def train_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int, evaluating: bool = False) -> Tuple[Dict, Dict]:
        """Perform a single training step. Run the model forward pass and compute losses.

        Args:
            batch (Dict): Input tensors.
            criterion (nn.Module): Loss layer designed for the model.
            optimizer_idx (int): Index of optimizer to use. 0 for the generator and 1 for the discriminator networks.
            evaluating (bool): different training_step performed in case we are performing an evaluation for certain phasese

        Returns:
            Tuple[Dict, Dict]: Model ouputs and computed losses.
        """
        #loading batch that will be used for any phase
        tokens = batch["tokens"]
        token_lengths = batch["token_lens"]
        mel_input = batch["mel_input"]
        mel_lens=batch["mel_lengths"]
        d_vectors = batch["d_vectors"]
        speaker_ids = batch["speaker_ids"]
        language_ids = batch["language_ids"]
 
        #PHASE 1 : PITCH ALIGNER
        if self.training_phase==1:

            #pitch aligner pass
            outputs=self.forward(
                training_phase=self.training_phase,
                x=tokens,
                x_lengths=token_lengths,
                mel_input=mel_input,
                mel_lens=mel_lens,
                aux_input={"d_vectors": d_vectors, "speaker_ids": speaker_ids, "language_ids": language_ids}
            )
                
            #Loss computation adapted from forwardtts loss
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    decoder_output_lens=mel_lens,
                    input_lens=token_lengths,
                    alignment_logprob=outputs['alignment_logprob'],
                    alignment_hard=outputs['alignment_mas'],
                    alignment_soft=outputs['alignment_soft'],
                    binary_loss_weight=self.binary_loss_weight,
            )
            
            return outputs, loss_dict
            
        #PHASE 2 : PITCH PREDICTOR
        elif self.training_phase==2:
            #loading batch
            pitch = batch["pitch"]

            outputs = self.forward(
                training_phase=self.training_phase,
                x=tokens,
                x_lengths=token_lengths,
                pitch=pitch,
                mel_input=mel_input,
                mel_lens=mel_lens,
                aux_input={"d_vectors": d_vectors, "speaker_ids": speaker_ids, "language_ids": language_ids},
            )
            
            #Loss computation adapted from forwardtts loss
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    pitch_output=outputs["pitch_avg"],
                    pitch_target=outputs["pitch_avg_gt"],
                    input_lens=token_lengths,
            )
            
            return outputs, loss_dict
            
            
        #PHASE 3 : CORE VITS
        #we also use this training_phase in case of evaluation for phases 5 and 6
        elif self.training_phase ==3 or evaluating:
            #we need to use different criterion if we are running an evaluatio
            if evaluating and not(self.training_phase ==3):
                from TTS.tts.layers.losses import (  # pylint: disable=import-outside-toplevel
                VitsDiscriminatorLoss,
                VitsGeneratorLoss,
                )
                eval_criterion_0=VitsDiscriminatorLoss(self.config)
                eval_criterion_1=VitsGeneratorLoss(self.config)


            #loading batch
            spec_lens = batch["spec_lens"]

            if optimizer_idx == 0:
                spec = batch["spec"]
                waveform = batch["waveform"]
                
                # generator pass
                outputs = self.forward(
                    training_phase=3,
                    x=tokens,
                    x_lengths=token_lengths,
                    y = spec,
                    y_lengths= spec_lens,
                    waveform = waveform,
                    mel_input=mel_input,
                    mel_lens=mel_lens,
                    aux_input={"d_vectors": d_vectors, "speaker_ids": speaker_ids, "language_ids": language_ids},
                )

                # cache tensors for the generator pass
                self.model_outputs_cache = outputs  # pylint: disable=attribute-defined-outside-init

                # compute scores and features
                scores_disc_fake, _, scores_disc_real, _ = self.disc(
                    outputs["model_outputs"].detach(), outputs["waveform_seg"]
                )

                # compute loss
                if evaluating and not(self.training_phase ==3):
                    with autocast(enabled=False):  # use float32 for the criterion
                        loss_dict = eval_criterion_0(
                            scores_disc_real,
                            scores_disc_fake,
                        )
                else:
                    with autocast(enabled=False):  # use float32 for the criterion
                        loss_dict = criterion[optimizer_idx](
                            scores_disc_real,
                            scores_disc_fake,
                        )
                    #return only if not evaluating
                    return outputs, loss_dict
                
            #we pursue the step if we are in eval mode
            if optimizer_idx == 1 or evaluating:
                #loading batch
                mel = batch["mel"]

                # compute melspec segment
                with autocast(enabled=False):

                    if self.args.encoder_sample_rate:
                        spec_segment_size = self.spec_segment_size * int(self.interpolate_factor)
                    else:
                        spec_segment_size = self.spec_segment_size

                    mel_slice = segment(
                        mel.float(), self.model_outputs_cache["slice_ids"], spec_segment_size, pad_short=True
                    )
                    mel_slice_hat = wav_to_mel(
                        y=self.model_outputs_cache["model_outputs"].float(),
                        n_fft=self.config.audio.fft_size,
                        sample_rate=self.config.audio.sample_rate,
                        num_mels=self.config.audio.num_mels,
                        hop_length=self.config.audio.hop_length,
                        win_length=self.config.audio.win_length,
                        fmin=self.config.audio.mel_fmin,
                        fmax=self.config.audio.mel_fmax,
                        center=False,
                    )

                # compute discriminator scores and features
                scores_disc_fake, feats_disc_fake, _, feats_disc_real = self.disc(
                    self.model_outputs_cache["model_outputs"], self.model_outputs_cache["waveform_seg"]
                )

                # compute losses
                if evaluating and not(self.training_phase ==3):
                    #we may not have created loss_dict yet, so we need to initialize (for example for phase 6)
                    if not('loss_dict' in locals()):
                        loss_dict={}
                    with autocast(enabled=False):  # use float32 for the criterion
                        loss_dict.update(
                            eval_criterion_1(
                                mel_slice=mel_slice.float(),
                                mel_slice_hat=mel_slice_hat.float(),
                                z_p=self.model_outputs_cache["z_p"].float(),
                                logs_q=self.model_outputs_cache["logs_q"].float(),
                                m_p=self.model_outputs_cache["m_p"].float(),
                                logs_p=self.model_outputs_cache["logs_p"].float(),
                                z_len=spec_lens,
                                scores_disc_fake=scores_disc_fake,
                                feats_disc_fake=feats_disc_fake,
                                feats_disc_real=feats_disc_real,
                                loss_duration=self.model_outputs_cache["loss_duration"],
                                use_speaker_encoder_as_loss=self.args.use_speaker_encoder_as_loss,
                                gt_spk_emb=self.model_outputs_cache["gt_spk_emb"],
                                syn_spk_emb=self.model_outputs_cache["syn_spk_emb"],
                                )
                            )
                else:
                    with autocast(enabled=False):  # use float32 for the criterion
                        loss_dict = criterion[optimizer_idx](
                            mel_slice=mel_slice.float(),
                            mel_slice_hat=mel_slice_hat.float(),
                            z_p=self.model_outputs_cache["z_p"].float(),
                            logs_q=self.model_outputs_cache["logs_q"].float(),
                            m_p=self.model_outputs_cache["m_p"].float(),
                            logs_p=self.model_outputs_cache["logs_p"].float(),
                            z_len=spec_lens,
                            scores_disc_fake=scores_disc_fake,
                            feats_disc_fake=feats_disc_fake,
                            feats_disc_real=feats_disc_real,
                            loss_duration=self.model_outputs_cache["loss_duration"],
                            use_speaker_encoder_as_loss=self.args.use_speaker_encoder_as_loss,
                            gt_spk_emb=self.model_outputs_cache["gt_spk_emb"],
                            syn_spk_emb=self.model_outputs_cache["syn_spk_emb"],
                        )

                return self.model_outputs_cache, loss_dict
        
        #PHASE 4 : CORE VITS
        elif self.training_phase ==4 and not(evaluating):
            #loading batch
            spec_lens = batch["spec_lens"]

            if optimizer_idx == 0:
                spec = batch["spec"]
                waveform = batch["waveform"]
                
                # generator pass
                outputs = self.forward(
                    training_phase=self.training_phase,
                    x=tokens,
                    x_lengths=token_lengths,
                    y = spec,
                    y_lengths= spec_lens,
                    waveform = waveform,
                    mel_input=mel_input,
                    mel_lens=mel_lens,
                    aux_input={"d_vectors": d_vectors, "speaker_ids": speaker_ids, "language_ids": language_ids},
                )

                # cache tensors for the generator pass
                self.model_outputs_cache = outputs  # pylint: disable=attribute-defined-outside-init

                # compute scores and features
                scores_disc_fake, _, scores_disc_real, _ = self.disc(
                    outputs["model_outputs"].detach(), outputs["waveform_seg"]
                )

                # compute loss
                with autocast(enabled=False):  # use float32 for the criterion
                    loss_dict = criterion[optimizer_idx](
                        scores_disc_real,
                        scores_disc_fake,
                    )
                return outputs, loss_dict
            if optimizer_idx == 1:
                #loading batch
                mel = batch["mel"]

                # compute melspec segment
                with autocast(enabled=False):

                    if self.args.encoder_sample_rate:
                        spec_segment_size = self.spec_segment_size * int(self.interpolate_factor)
                    else:
                        spec_segment_size = self.spec_segment_size

                mel_slice = segment(
                    mel.float(), self.model_outputs_cache["slice_ids"], spec_segment_size, pad_short=True
                )
                mel_slice_hat = wav_to_mel(
                    y=self.model_outputs_cache["model_outputs"].float(),
                    n_fft=self.config.audio.fft_size,
                    sample_rate=self.config.audio.sample_rate,
                    num_mels=self.config.audio.num_mels,
                    hop_length=self.config.audio.hop_length,
                    win_length=self.config.audio.win_length,
                    fmin=self.config.audio.mel_fmin,
                    fmax=self.config.audio.mel_fmax,
                    center=False,
                )
                # compute discriminator scores and features
                scores_disc_fake, feats_disc_fake, _, feats_disc_real = self.disc(
                    self.model_outputs_cache["model_outputs"], self.model_outputs_cache["waveform_seg"]
                )

                # compute losses
                with autocast(enabled=False):  # use float32 for the criterion
                    loss_dict = criterion[optimizer_idx](
                        mel_slice=mel_slice.float(),
                        mel_slice_hat=mel_slice_hat.float(),
                        z_len=spec_lens,
                        scores_disc_fake=scores_disc_fake,
                        feats_disc_fake=feats_disc_fake,
                        feats_disc_real=feats_disc_real,
                        use_speaker_encoder_as_loss=self.args.use_speaker_encoder_as_loss,
                        gt_spk_emb=self.model_outputs_cache["gt_spk_emb"],
                        syn_spk_emb=self.model_outputs_cache["syn_spk_emb"],
                    )

                return self.model_outputs_cache, loss_dict
                
        #PHASE  : VITS DURAL
        elif self.training_phase ==5 and not(evaluating):
            #loading batch
            spec_lens = batch["spec_lens"]
            spec = batch["spec"]
            
            # generator pass
            outputs = self.forward(
                training_phase=self.training_phase,
                x=tokens,
                x_lengths=token_lengths,
                y = spec,
                y_lengths= spec_lens,
                aux_input={"d_vectors": d_vectors, "speaker_ids": speaker_ids, "language_ids": language_ids},
            )

            # compute losses
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    z_p=outputs["z_p"].float(),
                    logs_q=outputs["logs_q"].float(),
                    m_p=outputs["m_p"].float(),
                    logs_p=outputs["logs_p"].float(),
                    z_len=spec_lens,
                    loss_duration=outputs["loss_duration"],
                    use_speaker_encoder_as_loss=self.args.use_speaker_encoder_as_loss,
                    gt_spk_emb=outputs["gt_spk_emb"],
                    syn_spk_emb=outputs["syn_spk_emb"],
                )

            return outputs, loss_dict
 
       #PHASE 6 : Just the discriminator  and waveform decoder
        elif self.training_phase ==6 and not(evaluating):
            #loading batch
            spec_lens = batch["spec_lens"]

            if optimizer_idx == 0:
                spec = batch["spec"]
                waveform = batch["waveform"]
                
                # generator pass
                outputs = self.forward(
                    training_phase=self.training_phase,
                    y = spec,
                    y_lengths= spec_lens,
                    waveform = waveform,
                    aux_input={"d_vectors": d_vectors, "speaker_ids": speaker_ids, "language_ids": language_ids},
                )

                # cache tensors for the generator pass
                self.model_outputs_cache = outputs  # pylint: disable=attribute-defined-outside-init

                # compute scores and features
                scores_disc_fake, _, scores_disc_real, _ = self.disc(
                    outputs["model_outputs"].detach(), outputs["waveform_seg"]
                )

                # compute loss
                with autocast(enabled=False):  # use float32 for the criterion
                    loss_dict = criterion[optimizer_idx](
                        scores_disc_real,
                        scores_disc_fake,
                    )
                return outputs, loss_dict
            if optimizer_idx == 1:
                #loading batch
                mel = batch["mel"]

                # compute melspec segment
                with autocast(enabled=False):

                    if self.args.encoder_sample_rate:
                        spec_segment_size = self.spec_segment_size * int(self.interpolate_factor)
                    else:
                        spec_segment_size = self.spec_segment_size

                    mel_slice = segment(
                        mel.float(), self.model_outputs_cache["slice_ids"], spec_segment_size, pad_short=True
                    )
                    mel_slice_hat = wav_to_mel(
                        y=self.model_outputs_cache["model_outputs"].float(),
                        n_fft=self.config.audio.fft_size,
                        sample_rate=self.config.audio.sample_rate,
                        num_mels=self.config.audio.num_mels,
                        hop_length=self.config.audio.hop_length,
                        win_length=self.config.audio.win_length,
                        fmin=self.config.audio.mel_fmin,
                        fmax=self.config.audio.mel_fmax,
                        center=False,
                    )

                # compute discriminator scores and features
                scores_disc_fake, feats_disc_fake, _, feats_disc_real = self.disc(
                    self.model_outputs_cache["model_outputs"], self.model_outputs_cache["waveform_seg"]
                )

                # compute losses
                with autocast(enabled=False):  # use float32 for the criterion
                    loss_dict = criterion[optimizer_idx](
                        mel_slice=mel_slice.float(),
                        mel_slice_hat=mel_slice_hat.float(),
                        scores_disc_fake=scores_disc_fake,
                        feats_disc_fake=feats_disc_fake,
                        feats_disc_real=feats_disc_real,
                        use_speaker_encoder_as_loss=self.args.use_speaker_encoder_as_loss,
                        gt_spk_emb=self.model_outputs_cache["gt_spk_emb"],
                        syn_spk_emb=self.model_outputs_cache["syn_spk_emb"],
                    )

                return self.model_outputs_cache, loss_dict
            
        raise ValueError(" [!] Unexpected `optimizer_idx`.")


    def _log(self, ap, batch, outputs, name_prefix="train"):  # pylint: disable=unused-argument,no-self-use
        
        if self.training_phase==2:
            pitch_avg = abs(outputs[1]["pitch_avg_gt"][0, 0].data.cpu().numpy())
            pitch_avg_hat = abs(outputs[1]["pitch_avg"][0, 0].data.cpu().numpy())
            tokens = self.tokenizer.decode(batch["tokens"][0].data.cpu().numpy())
            pitch_figures = {
                "pitch_ground_truth": plot_avg_pitch(pitch_avg, tokens, output_fig=False),
                "pitch_avg_predicted": plot_avg_pitch(pitch_avg_hat, tokens, output_fig=False),
            }
            figures.update(pitch_figures)
            return figures
        
        elif self.training_phase==3:
            y_hat = outputs[1]["model_outputs"]
            y = outputs[1]["waveform_seg"]
            figures = plot_results(y_hat, y, ap, name_prefix)
            sample_voice = y_hat[0].squeeze(0).detach().cpu().numpy()
            audios = {f"{name_prefix}/audio": sample_voice}

            alignments = outputs[1]["alignments"]
            align_img = alignments[0].data.cpu().numpy().T

            figures.update(
                {
                    "alignment": plot_alignment(align_img, output_fig=False),
                }
            )
            return figures, audios
            
        elif self.training_phase==4:
            y_hat = outputs[1]["model_outputs"]
            y = outputs[1]["waveform_seg"]
            figures = plot_results(y_hat, y, ap, name_prefix)
            sample_voice = y_hat[0].squeeze(0).detach().cpu().numpy()
            audios = {f"{name_prefix}/audio": sample_voice}

            return figures, audios
            
        elif self.training_phase==5:

            alignments = outputs[0]["alignments"]
            align_img = alignments[0].data.cpu().numpy().T

            figures={"alignment": plot_alignment(align_img, output_fig=False),}
            return figures        
        
        elif self.training_phase==6:
            y_hat = outputs[1]["model_outputs"]
            y = outputs[1]["waveform_seg"]
            figures = plot_results(y_hat, y, ap, name_prefix)
            sample_voice = y_hat[0].squeeze(0).detach().cpu().numpy()
            audios = {f"{name_prefix}/audio": sample_voice}

            return figures, audios

        
        raise ValueError(" [!] No logs associated with current trainig phase {}.".format(str(self.training_phase)))

    def train_log(
        self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int
    ):  # pylint: disable=no-self-use
        """Create visualizations and waveform examples.

        For example, here you can plot spectrograms and generate sample sample waveforms from these spectrograms to
        be projected onto Tensorboard.

        Args:
            ap (AudioProcessor): audio processor used at training.
            batch (Dict): Model inputs used at the previous training step.
            outputs (Dict): Model outputs generated at the previoud training step.

        Returns:
            Tuple[Dict, np.ndarray]: training plots and output waveform.
        """
        if self.training_phase==1:
            print("nothing to plot in phase 1....yet!")
        
        elif self.training_phase==2:
            print("nothing to plot in phase 2....yet!")
            #figures = self._log(self.ap, batch, outputs, "train")
            #logger.train_figures(steps, figures)
        
        elif self.training_phase==3 or self.training_phase==4 or self.training_phase==6:
            figures, audios = self._log(self.ap, batch, outputs, "train")
            logger.train_figures(steps, figures)
            logger.train_audios(steps, audios, self.ap.sample_rate)
            
        elif self.training_phase==5:
            figures = self._log(self.ap, batch, outputs, "train")
            logger.train_figures(steps, figures)
            


    @torch.no_grad()
    def eval_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        #we added one arg : evaluating = True, so that we use a more complete training_step
        return self.train_step(batch, criterion, optimizer_idx,evaluating=True)

    def eval_log(self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int) -> None:
        if self.training_phase==1:
            print("No figures or audio to log in phase 1")
            
        elif self.training_phase==2:
            print("No figures or audio to log in phase 2")
            
        elif self.training_phase==3 or self.training_phase==4 or self.training_phase==6:
            figures, audios = self._log(self.ap, batch, outputs, "eval")
            logger.eval_figures(steps, figures)
            logger.eval_audios(steps, audios, self.ap.sample_rate)

        elif self.training_phase==5:
            figures = self._log(self.ap, batch, outputs, "eval")
            logger.train_figures(steps, figures)


    def get_aux_input_from_test_sentences(self, sentence_info):
        if hasattr(self.config, "model_args"):
            config = self.config.model_args
        else:
            config = self.config

        # extract speaker and language info
        text, speaker_name, style_wav, language_name = None, None, None, None

        if isinstance(sentence_info, list):
            if len(sentence_info) == 1:
                text = sentence_info[0]
            elif len(sentence_info) == 2:
                text, speaker_name = sentence_info
            elif len(sentence_info) == 3:
                text, speaker_name, style_wav = sentence_info
            elif len(sentence_info) == 4:
                text, speaker_name, style_wav, language_name = sentence_info
        else:
            text = sentence_info

        # get speaker  id/d_vector
        speaker_id, d_vector, language_id = None, None, None
        if hasattr(self, "speaker_manager"):
            if config.use_d_vector_file:
                if speaker_name is None:
                    d_vector = self.speaker_manager.get_random_embedding()
                else:
                    d_vector = self.speaker_manager.get_mean_embedding(speaker_name, num_samples=None, randomize=False)
            elif config.use_speaker_embedding:
                if speaker_name is None:
                    speaker_id = self.speaker_manager.get_random_id()
                else:
                    speaker_id = self.speaker_manager.name_to_id[speaker_name]

        # get language id
        if hasattr(self, "language_manager") and config.use_language_embedding and language_name is not None:
            language_id = self.language_manager.name_to_id[language_name]

        return {
            "text": text,
            "speaker_id": speaker_id,
            "style_wav": style_wav,
            "d_vector": d_vector,
            "language_id": language_id,
            "language_name": language_name,
        }

    @torch.no_grad()
    def test_run(self, assets) -> Tuple[Dict, Dict]:
        """Generic test run for `tts` models used by `Trainer`.

        You can override this for a different behaviour.

        Returns:
            Tuple[Dict, Dict]: Test figures and audios to be projected to Tensorboard.
        """
        print(" | > Synthesizing test sentences.")
        test_audios = {}
        test_figures = {}
        test_sentences = self.config.test_sentences
        for idx, s_info in enumerate(test_sentences):
            aux_inputs = self.get_aux_input_from_test_sentences(s_info)
            wav, alignment, _, _ = synthesis(
                self,
                aux_inputs["text"],
                self.config,
                "cuda" in str(next(self.parameters()).device),
                speaker_id=aux_inputs["speaker_id"],
                d_vector=aux_inputs["d_vector"],
                style_wav=aux_inputs["style_wav"],
                language_id=aux_inputs["language_id"],
                use_griffin_lim=True,
                do_trim_silence=False,
            ).values()
            test_audios["{}-audio".format(idx)] = wav
            test_figures["{}-alignment".format(idx)] = plot_alignment(alignment.T, output_fig=False)
        return {"figures": test_figures, "audios": test_audios}

    def test_log(
        self, outputs: dict, logger: "Logger", assets: dict, steps: int  # pylint: disable=unused-argument
    ) -> None:
        logger.test_audios(steps, outputs["audios"], self.ap.sample_rate)
        logger.test_figures(steps, outputs["figures"])

    def format_batch(self, batch: Dict) -> Dict:
        """Compute speaker, langugage IDs and d_vector for the batch if necessary."""
        speaker_ids = None
        language_ids = None
        d_vectors = None

        # get numerical speaker ids from speaker names
        if self.speaker_manager is not None and self.speaker_manager.name_to_id and self.args.use_speaker_embedding:
            speaker_ids = [self.speaker_manager.name_to_id[sn] for sn in batch["speaker_names"]]

        if speaker_ids is not None:
            speaker_ids = torch.LongTensor(speaker_ids)
            batch["speaker_ids"] = speaker_ids

        # get d_vectors from audio file names
        if self.speaker_manager is not None and self.speaker_manager.embeddings and self.args.use_d_vector_file:
            d_vector_mapping = self.speaker_manager.embeddings
            d_vectors = [d_vector_mapping[w]["embedding"] for w in batch["audio_unique_names"]]
            d_vectors = torch.FloatTensor(d_vectors)

        # get language ids from language names
        if self.language_manager is not None and self.language_manager.name_to_id and self.args.use_language_embedding:
            language_ids = [self.language_manager.name_to_id[ln] for ln in batch["language_names"]]

        if language_ids is not None:
            language_ids = torch.LongTensor(language_ids)

        batch["language_ids"] = language_ids
        batch["d_vectors"] = d_vectors
        batch["speaker_ids"] = speaker_ids
        return batch

    def format_batch_on_device(self, batch):
        """Compute spectrograms on the device."""
        ac = self.config.audio

        if self.args.encoder_sample_rate:
            wav = self.audio_resampler(batch["waveform"])
        else:
            wav = batch["waveform"]

        # compute spectrograms
        batch["spec"] = wav_to_spec(wav, ac.fft_size, ac.hop_length, ac.win_length, center=False)

        if self.args.encoder_sample_rate:
            # recompute spec with high sampling rate to the loss
            spec_mel = wav_to_spec(batch["waveform"], ac.fft_size, ac.hop_length, ac.win_length, center=False)
            # remove extra stft frames if needed
            if spec_mel.size(2) > int(batch["spec"].size(2) * self.interpolate_factor):
                spec_mel = spec_mel[:, :, : int(batch["spec"].size(2) * self.interpolate_factor)]
            else:
                batch["spec"] = batch["spec"][:, :, : int(spec_mel.size(2) / self.interpolate_factor)]
        else:
            spec_mel = batch["spec"]

        batch["mel"] = spec_to_mel(
            spec=spec_mel,
            n_fft=ac.fft_size,
            num_mels=ac.num_mels,
            sample_rate=ac.sample_rate,
            fmin=ac.mel_fmin,
            fmax=ac.mel_fmax,
        )

        if self.args.encoder_sample_rate:
            assert batch["spec"].shape[2] == int(
                batch["mel"].shape[2] / self.interpolate_factor
            ), f"{batch['spec'].shape[2]}, {batch['mel'].shape[2]}"
        else:
            assert batch["spec"].shape[2] == batch["mel"].shape[2], f"{batch['spec'].shape[2]}, {batch['mel'].shape[2]}"

        # compute spectrogram frame lengths
        batch["spec_lens"] = (batch["spec"].shape[2] * batch["waveform_rel_lens"]).int()
        batch["mel_lens"] = (batch["mel"].shape[2] * batch["waveform_rel_lens"]).int()

        if self.args.encoder_sample_rate:
            assert (batch["spec_lens"] - (batch["mel_lens"] / self.interpolate_factor).int()).sum() == 0
        else:
            assert (batch["spec_lens"] - batch["mel_lens"]).sum() == 0

        # zero the padding frames
        batch["spec"] = batch["spec"] * sequence_mask(batch["spec_lens"]).unsqueeze(1)
        batch["mel"] = batch["mel"] * sequence_mask(batch["mel_lens"]).unsqueeze(1)
        return batch

    def get_sampler(self, config: Coqpit, dataset: TTSDataset, num_gpus=1, is_eval=False):
        weights = None
        data_items = dataset.samples
        if getattr(config, "use_weighted_sampler", False):
            for attr_name, alpha in config.weighted_sampler_attrs.items():
                print(f" > Using weighted sampler for attribute '{attr_name}' with alpha '{alpha}'")
                multi_dict = config.weighted_sampler_multipliers.get(attr_name, None)
                print(multi_dict)
                weights, attr_names, attr_weights = get_attribute_balancer_weights(
                    attr_name=attr_name, items=data_items, multi_dict=multi_dict
                )
                weights = weights * alpha
                print(f" > Attribute weights for '{attr_names}' \n | > {attr_weights}")

        # input_audio_lenghts = [os.path.getsize(x["audio_file"]) for x in data_items]

        if weights is not None:
            w_sampler = WeightedRandomSampler(weights, len(weights))
            batch_sampler = BucketBatchSampler(
                w_sampler,
                data=data_items,
                batch_size=config.eval_batch_size if is_eval else config.batch_size,
                sort_key=lambda x: os.path.getsize(x["audio_file"]),
                drop_last=True,
            )
        else:
            batch_sampler = None
        # sampler for DDP
        if batch_sampler is None:
            batch_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        else:  # If a sampler is already defined use this sampler and DDP sampler together
            batch_sampler = (
                DistributedSamplerWrapper(batch_sampler) if num_gpus > 1 else batch_sampler
            )  # TODO: check batch_sampler with multi-gpu
        return batch_sampler

    def get_data_loader(
        self,
        config: Coqpit,
        assets: Dict,
        is_eval: bool,
        samples: Union[List[Dict], List[List]],
        verbose: bool,
        num_gpus: int,
        rank: int = None,
    ) -> "DataLoader":
        if is_eval and not config.run_eval:
            loader = None
        else:
            # init dataloader
            dataset = ModularVitsDataset(
                model_args=self.args,
                samples=samples,
                batch_group_size=0 if is_eval else config.batch_group_size * config.batch_size,
                min_text_len=config.min_text_len,
                max_text_len=config.max_text_len,
                min_audio_len=config.min_audio_len,
                max_audio_len=config.max_audio_len,
                phoneme_cache_path=config.phoneme_cache_path,
                precompute_num_workers=config.precompute_num_workers,
                verbose=verbose,
                tokenizer=self.tokenizer,
                start_by_longest=config.start_by_longest,
                #ADDITION FOR FAST_PITCH
                compute_f0=self.compute_f0,
                f0_cache_path=self.f0_cache_path,
                ap=self.ap,
                training_phase=self.training_phase
            )

            # wait all the DDP process to be ready
            if num_gpus > 1:
                dist.barrier()

            # sort input sequences from short to long
            dataset.preprocess_samples()

            # get samplers
            sampler = self.get_sampler(config, dataset, num_gpus)
            if sampler is None:
                loader = DataLoader(
                    dataset,
                    batch_size=config.eval_batch_size if is_eval else config.batch_size,
                    shuffle=False,  # shuffle is done in the dataset.
                    collate_fn=dataset.collate_fn,
                    drop_last=False,  # setting this False might cause issues in AMP training.
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=False,
                )
            else:
                #workaround for multi GPUs (cf https://github.com/coqui-ai/TTS/issues/1986)
                if num_gpus == 1 :
                    loader = DataLoader(
                        dataset,
                        batch_sampler=sampler,
                        collate_fn=dataset.collate_fn,
                        num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                        pin_memory=False,
                    )
                else:
                    loader = DataLoader(
                        dataset,
                        #batch_sampler=sampler,
                        sampler=sampler,
                        batch_size=config.eval_batch_size if is_eval else config.batch_size,
                        collate_fn=dataset.collate_fn,
                        num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                        pin_memory=False,
                    )
        return loader

    def get_optimizer(self) -> List:
        """Initiate and return the GAN optimizers based on the config parameters.
        It returnes 2 optimizers in a list. First one is for the generator and the second one is for the discriminator.
        Returns:
            List: optimizers.
        """
        if self.training_phase==1:
            print("Using the same optimizer as FastPitch")
            pitch_align_encod_params=chain(self.pitch_aligner.parameters(), self.pitch_text_encoder.parameters())
            optimizer_pitch_aligner = get_optimizer(self.config.optimizer_pitch_aligner, self.config.optimizer_pitch_aligner_params, self.config.lr_pitch_aligner, parameters = pitch_align_encod_params)
            return [optimizer_pitch_aligner]
            
        elif self.training_phase==2:
            print("Using the same optimizer as FastPitch")
            pitch_pred_encod_params=chain(self.pitch_predictor.parameters(), self.pitch_text_encoder.parameters())
            optimizer_pitch_predictor=get_optimizer(self.config.optimizer_pitch_predictor, self.config.optimizer_pitch_predictor_params, self.config.lr_pitch_predictor, parameters = pitch_pred_encod_params)
            return [optimizer_pitch_predictor]


        elif self.training_phase==3:
            print("Using regular VITS optimizers")
            # discriminator optimizer
            optimizer0 = get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr_disc, self.disc)

            #generator parameters
            #gen_parameters = chain(params for k, params in self.named_parameters() if not k.startswith("disc."))
            gen_parameters=chain(
                self.waveform_decoder.parameters(),
                self.posterior_encoder.parameters(),
                self.flow.parameters(),
                self.duration_predictor.parameters(),
                self.text_encoder.parameters(),
                self.pitch_conv1d.parameters()
                )
            optimizer1 = get_optimizer(
                self.config.optimizer, self.config.optimizer_params, self.config.lr_gen, parameters=gen_parameters
            )
            return [optimizer0, optimizer1]



        elif self.training_phase==4:
            print("Using regular VITS optimizers")
            # select generator parameters
            optimizer0 = get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr_disc, self.disc)

            #gen_parameters = chain(params for k, params in self.named_parameters() if not k.startswith("disc."))
            gen_parameters=chain(
                self.waveform_decoder.parameters(),
                #self.posterior_encoder.parameters(),
                self.flow.parameters(),
                #self.duration_predictor.parameters(),
                self.text_encoder.parameters(),
                #self.pitch_conv1d.parameters()
                )
            optimizer1 = get_optimizer(
                self.config.optimizer, self.config.optimizer_params, self.config.lr_gen, parameters=gen_parameters
            )
            return [optimizer0, optimizer1]

        elif self.training_phase==5:
            print("Using regular VITS Generator Parameters")
            # select generator parameters useful for this phase
            dur_parameters=chain(
                self.duration_predictor.parameters(),
                self.flow.parameters()
            )
            # select generator parameters useful for this phase
            optimizer_dural = get_optimizer(
                self.config.optimizer, self.config.optimizer_params, self.config.lr_gen, parameters=dur_parameters
            )
            return [optimizer_dural]


        elif self.training_phase==6:
            print("Using regular VITS optimizers only for discriminator and waveform decoder")
            # discriminator optimizer
            optimizer0 = get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr_disc, self.disc)

            #generator optimizer
            optimizer1 = get_optimizer(
                self.config.optimizer, self.config.optimizer_params, self.config.lr_gen, self.waveform_decoder
            )
            return [optimizer0, optimizer1]

        raise RuntimeError(
            " [!] No optimizer associated with training phase {}".format(self.training_phase)
        )
        
    def get_lr(self) -> List:
        """Set the initial learning rates for each optimizer.

        Returns:
            List: learning rates for each optimizer.
        """
        if self.training_phase==1:
            return [self.config.lr_pitch_aligner]
                
        elif self.training_phase==2:
            return [self.config.lr_pitch_predictor]
        
        elif self.training_phase==3 or self.training_phase==4 or self.training_phase==6:
            return [self.config.lr_disc, self.config.lr_gen]
         
        elif self.training_phase==5:
            return [self.config.lr_gen]

        raise RuntimeError(
            " [!] No learning rate associated with training phase {}".format(self.training_phase)
        )
        

    def get_scheduler(self, optimizer) -> List:
        """Set the schedulers for each optimizer.

        Args:
            optimizer (List[`torch.optim.Optimizer`]): List of optimizers.

        Returns:
            List: Schedulers, one for each optimizer.
        """
        if self.training_phase==1:
            scheduler_pitch_aligner=get_scheduler(self.config.lr_scheduler_pitch_aligner, self.config.lr_scheduler_pitch_aligner_params, optimizer[0])
            return [scheduler_pitch_aligner]
 
        elif self.training_phase==2:
            scheduler_pitch_predictor=get_scheduler(self.config.lr_scheduler_pitch_predictor, self.config.lr_scheduler_pitch_predictor_params, optimizer[0])
            return  [scheduler_pitch_predictor]
 
        elif self.training_phase==3 or self.training_phase==4 or self.training_phase==6:
            scheduler_G = get_scheduler(self.config.lr_scheduler_gen, self.config.lr_scheduler_gen_params, optimizer[0])
            scheduler_D = get_scheduler(self.config.lr_scheduler_disc, self.config.lr_scheduler_disc_params, optimizer[1])
            return[scheduler_D,scheduler_G]
            
        elif self.training_phase==5:
            scheduler_G = get_scheduler(self.config.lr_scheduler_gen, self.config.lr_scheduler_gen_params, optimizer[0])
            return[scheduler_G]

        raise RuntimeError(
            " [!] No scheduler(s) associated with training phase {}".format(self.training_phase)
        )

    def get_criterion(self):
        """Get criterions for each optimizer. The index in the output list matches the optimizer idx used in
        `train_step()`"""
        
        if self.training_phase==1:
            from TTS.tts.layers.losses import (PitchAlignerLoss) 
            print("Launching PitchAlignerloss as criterion")
            return [PitchAlignerLoss(self.config)]
        
        elif self.training_phase==2:
            from TTS.tts.layers.losses import (PitchPredictorLoss)
            print("launching Pitch loss as criterion")
            return [PitchPredictorLoss(self.config)]
            
        elif self.training_phase==3:        
            from TTS.tts.layers.losses import (  # pylint: disable=import-outside-toplevel
                VitsDiscriminatorLoss,
                VitsGeneratorLoss,
            )
            print("launching VitsDiscriminatorLoss and VitsGeneratorLoss as criterion")
            return [VitsDiscriminatorLoss(self.config), VitsGeneratorLoss(self.config)]        

        elif self.training_phase==4:        
            from TTS.tts.layers.losses import (  # pylint: disable=import-outside-toplevel
                VitsDiscriminatorLoss,
                VitsReducedGeneratorLoss,
            )
            print("launching VitsDiscriminatorLoss and VitsReducedGeneratorLoss as criterion")
            return [VitsDiscriminatorLoss(self.config), VitsReducedGeneratorLoss(self.config)]  

        elif self.training_phase==5:        
            from TTS.tts.layers.losses import (  # pylint: disable=import-outside-toplevel
                VitsDuralLoss,
            )
            print("launching VitsDuralLoss as criterion")
            return [VitsDuralLoss(self.config)]   
        
        elif self.training_phase==6:        
            from TTS.tts.layers.losses import (  # pylint: disable=import-outside-toplevel
                VitsDiscriminatorLoss,
                VitsWaveformDecoderLoss,
            )
            print("launching VitsDiscriminatorLoss and VitsWaveformDecoderLoss as criterion")
            return [VitsDiscriminatorLoss(self.config), VitsWaveformDecoderLoss(self.config)]   
        
        else:
            print("No criterion to launch yet")
            return []

    def load_checkpoint(
        self,
        config,
        checkpoint_path,
        eval=False,
        strict=True,
    ):  # pylint: disable=unused-argument, redefined-builtin
        """Load the model checkpoint and setup for training or inference"""
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"))
        # compat band-aid for the pre-trained models to not use the encoder baked into the model
        # TODO: consider baking the speaker encoder into the model and call it from there.
        # as it is probably easier for model distribution.
        state["model"] = {k: v for k, v in state["model"].items() if "speaker_encoder" not in k}

        if self.args.encoder_sample_rate is not None and eval:
            # audio resampler is not used in inference time
            self.audio_resampler = None

        # handle fine-tuning from a checkpoint with additional speakers
        if hasattr(self, "emb_g") and state["model"]["emb_g.weight"].shape != self.emb_g.weight.shape:
            num_new_speakers = self.emb_g.weight.shape[0] - state["model"]["emb_g.weight"].shape[0]
            print(f" > Loading checkpoint with {num_new_speakers} additional speakers.")
            emb_g = state["model"]["emb_g.weight"]
            new_row = torch.randn(num_new_speakers, emb_g.shape[1])
            emb_g = torch.cat([emb_g, new_row], axis=0)
            state["model"]["emb_g.weight"] = emb_g
        # load the model weights
        self.load_state_dict(state["model"], strict=strict)

        if eval:
            self.eval()
            assert not self.training

    @staticmethod
    def init_from_config(config: "ModularVitsConfig", samples: Union[List[List], List[Dict]] = None, verbose=True):
        """Initiate model from config

        Args:
            config (VitsConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        from TTS.utils.audio import AudioProcessor

        upsample_rate = torch.prod(torch.as_tensor(config.model_args.upsample_rates_decoder)).item()

        if not config.model_args.encoder_sample_rate:
            assert (
                upsample_rate == config.audio.hop_length
            ), f" [!] Product of upsample rates must be equal to the hop length - {upsample_rate} vs {config.audio.hop_length}"
        else:
            encoder_to_vocoder_upsampling_factor = config.audio.sample_rate / config.model_args.encoder_sample_rate
            effective_hop_length = config.audio.hop_length * encoder_to_vocoder_upsampling_factor
            assert (
                upsample_rate == effective_hop_length
            ), f" [!] Product of upsample rates must be equal to the hop length - {upsample_rate} vs {effective_hop_length}"

        ap = AudioProcessor.init_from_config(config, verbose=verbose)
        tokenizer, new_config = TTSTokenizer.init_from_config(config)
        speaker_manager = SpeakerManager.init_from_config(config, samples)
        language_manager = LanguageManager.init_from_config(config)

        if config.model_args.speaker_encoder_model_path:
            speaker_manager.init_encoder(
                config.model_args.speaker_encoder_model_path, config.model_args.speaker_encoder_config_path
            )
        return ModularVits(new_config, ap, tokenizer, speaker_manager, language_manager)

    #ADDITION FOR FAST_PITCH
    def on_train_step_start(self, trainer):
        """Schedule binary loss weight."""
        if trainer.epochs_done>0:
            self.binary_loss_weight = min(trainer.epochs_done / self.config.binary_loss_warmup_epochs, 1.0) * 1.0
        else:
            self.binary_loss_weight = 0.0
##################################
# VITS CHARACTERS
##################################


class ModularVitsCharacters(BaseCharacters):
    """Characters class for VITs model for compatibility with pre-trained models"""

    def __init__(
        self,
        graphemes: str = _characters,
        punctuations: str = _punctuations,
        pad: str = _pad,
        ipa_characters: str = _phonemes,
    ) -> None:
        if ipa_characters is not None:
            graphemes += ipa_characters
        super().__init__(graphemes, punctuations, pad, None, None, "<BLNK>", is_unique=False, is_sorted=True)

    def _create_vocab(self):
        self._vocab = [self._pad] + list(self._punctuations) + list(self._characters) + [self._blank]
        self._char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        # pylint: disable=unnecessary-comprehension
        self._id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

    @staticmethod
    def init_from_config(config: Coqpit):
        if config.characters is not None:
            _pad = config.characters["pad"]
            _punctuations = config.characters["punctuations"]
            _letters = config.characters["characters"]
            _letters_ipa = config.characters["phonemes"]
            return (
                ModularVitsCharacters(graphemes=_letters, ipa_characters=_letters_ipa, punctuations=_punctuations, pad=_pad),
                config,
            )
        characters = ModularVitsCharacters()
        new_config = replace(config, characters=characters.to_config())
        return characters, new_config

    def to_config(self) -> "CharactersConfig":
        return CharactersConfig(
            characters=self._characters,
            punctuations=self._punctuations,
            pad=self._pad,
            eos=None,
            bos=None,
            blank=self._blank,
            is_unique=False,
            is_sorted=True,
        )
