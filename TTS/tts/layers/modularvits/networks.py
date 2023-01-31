import math
import torch
from torch import nn
from TTS.tts.utils.helpers import sequence_mask

LRELU_SLOPE = 0.1


class TextEmbedderForPitch(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        hidden_channels: int,
    ):
        """Text Embedder for the pitch section in ModularVITS model.

        Args:
            n_vocab (int): Number of characters for the embedding layer.
            hidden_channels (int): Number of channels for the hidden layers.
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.emb = nn.Embedding(n_vocab, hidden_channels)

    def forward(self, x, x_lengths, lang_emb=None):
        """
        Shapes:
            - x: :math:`[B, T]`
            - x_length: :math:`[B]`
        """
        assert x.shape[0] == x_lengths.shape[0]
        #ADDITION FOR FAST_PITCH
        x_emb=self.emb(x)
        x = x_emb * math.sqrt(self.hidden_channels)  # [b, t, h]
        #x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]

        # concat the lang emb in embedding chars
        if lang_emb is not None:
            x = torch.cat((x, lang_emb.transpose(2, 1).expand(x.size(0), x.size(1), -1)), dim=-1)

        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)  # [b, 1, t]

        return x_mask, x_emb