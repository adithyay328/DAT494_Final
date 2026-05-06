"""EnCodec-based audio codec: MP3 → tokens → MP3."""

import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio


class Codec:
    """Wraps Meta's EnCodec 24 kHz model for encode/decode of audio files."""

    def __init__(self, bandwidth: float = 6.0):
        """
        Args:
            bandwidth: Target bandwidth in kbps. Controls the number of
                codebooks used (and therefore compression). Common values:
                1.5, 3.0, 6.0, 12.0, 24.0.
        """
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.sample_rate = self.model.sample_rate  # 24000
        self.channels = self.model.channels  # 1

    # ------------------------------------------------------------------
    # Encode: audio file → discrete tokens
    # ------------------------------------------------------------------
    def encode(self, audio_path: str) -> torch.Tensor:
        """Load an audio file and return EnCodec token codes.

        Args:
            audio_path: Path to an input audio file (MP3, WAV, etc.).

        Returns:
            Tensor of shape ``[1, n_codebooks, T]`` containing discrete
            codebook indices.
        """
        wav, sr = torchaudio.load(audio_path)
        wav = convert_audio(wav, sr, self.sample_rate, self.channels)
        wav = wav.unsqueeze(0)  # [1, C, T]

        with torch.no_grad():
            encoded_frames = self.model.encode(wav)

        # Each frame is (codes, scale); codes shape [B, K, T_frame]
        codes = torch.cat([frame[0] for frame in encoded_frames], dim=-1)
        return codes

    # ------------------------------------------------------------------
    # Encode continuous: audio file → continuous embeddings
    # ------------------------------------------------------------------
    def encode_continuous(self, audio_path: str) -> torch.Tensor:
        """Load an audio file and return post-quantization continuous embeddings.

        Args:
            audio_path: Path to an input audio file (MP3, WAV, etc.).

        Returns:
            Tensor of shape ``[1, 128, T]`` — the quantized continuous
            embeddings that the decoder consumes.
        """
        wav, sr = torchaudio.load(audio_path)
        wav = convert_audio(wav, sr, self.sample_rate, self.channels)
        wav = wav.unsqueeze(0)  # [1, C, T]

        with torch.no_grad():
            emb = self.model.encoder(wav)
            codes = self.model.quantizer.encode(
                emb, self.model.frame_rate, self.model.bandwidth
            )
            emb_q = self.model.quantizer.decode(codes)
        return emb_q

    # ------------------------------------------------------------------
    # Decode continuous: continuous embeddings → audio file
    # ------------------------------------------------------------------
    def decode_continuous(self, emb: torch.Tensor, output_path: str) -> None:
        """Decode continuous embeddings back to audio and save to disk.

        Args:
            emb: Tensor of shape ``[1, 128, T]`` (as returned by
                :meth:`encode_continuous`).
            output_path: Destination file path.
        """
        with torch.no_grad():
            audio = self.model.decoder(emb)
        audio = audio.squeeze(0)
        torchaudio.save(output_path, audio.cpu(), self.sample_rate)

    # ------------------------------------------------------------------
    # Decode: discrete tokens → audio file
    # ------------------------------------------------------------------
    def decode(self, codes: torch.Tensor, output_path: str) -> None:
        """Decode token codes back to audio and save to disk.

        Args:
            codes: Tensor of shape ``[1, n_codebooks, T]`` (as returned
                by :meth:`encode`).
            output_path: Destination file path. Format is inferred from
                the extension (e.g. ``.mp3``, ``.wav``).
        """
        # Reconstruct the frames list expected by model.decode
        frames = [(codes, None)]

        with torch.no_grad():
            audio = self.model.decode(frames)

        # audio shape: [B, C, T] → drop batch dim for saving
        audio = audio.squeeze(0)
        torchaudio.save(output_path, audio.cpu(), self.sample_rate)
