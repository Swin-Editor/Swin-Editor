import torch
from torch import nn
from einops import rearrange, repeat
import torch.nn.functional as F
from diffusers.models.attention import CrossAttention, FeedForward, AdaLayerNorm


class SparseCausalAttention(CrossAttention):
    """
    A class that extends CrossAttention and implements a Sparse Causal Attention mechanism.
    @param CrossAttention - The base class for the attention mechanism.
    @method forward - The forward method for the SparseCausalAttention class.
    @param hidden_states - The hidden states of the model.
    @param encoder_hidden_states - The hidden states of the encoder.
    @param attention_mask - The attention mask for masking out certain positions.
    @param video_length - The length of the video sequence.
    @return The output hidden states after applying the attention mechanism.
    """
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        former_frame_index = torch.arange(video_length) - 1
        former_frame_index[0] = 0
        former_frame_index_1 = torch.arange(video_length) - 1
        former_frame_index_1[0] = 0
        former_frame_index_1[2] = 0

        key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        key = torch.cat([key[:, [0] * video_length], key[:, former_frame_index], key[:, former_frame_index_1]], dim=2)
        key = rearrange(key, "b f d c -> (b f) d c")

        value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        value = torch.cat([value[:, [0] * video_length], value[:, former_frame_index], value[:, former_frame_index_1]], dim=2)
        value = rearrange(value, "b f d c -> (b f) d c")

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states
        


class SwinAttention(CrossAttention):
    """
    Define a SwinAttention class that inherits from CrossAttention. Implement the forward method for the SwinAttention class.
    @param hidden_states - The input hidden states
    @param encoder_hidden_states - The hidden states from the encoder
    @param attention_mask - The attention mask
    @param window_size - The size of the attention window
    @param video_length - The length of the video
    @param mask - The mask for the attention
    @return The updated hidden states
    """
    def forward(self,hidden_states, encoder_hidden_states=None, attention_mask=None, window_size=3, video_length=None,mask=None):
        batch_size, seq_length, features = hidden_states.size()

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)


        # Pad input sequences to handle edge cases
        padding = (window_size - 1) // 2
        padded_query = F.pad(query, (0, 0,padding, padding))
        padded_key = F.pad(key, (0, 0, padding, padding))
        padded_value = F.pad(value, (0, 0, padding, padding))
        # Define the shifted windows
        windows = []
        for i in range(window_size):
            windows.append(padded_query[:, i:i + seq_length, :])
        # Compute attention scores
        scores = torch.stack([torch.baddbmm(torch.empty(windows[i].shape[0], windows[i].shape[1], padded_key.shape[1], dtype=windows[i].dtype, device=windows[i].device), windows[i], padded_key.transpose(2,1),beta=0,alpha=1/8) for i in range(window_size)],dim=2).mean(dim=2)
        scores /= torch.sqrt(torch.tensor(features, dtype=torch.float32))
        # Apply mask if necessary
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

    

        # Compute softmax and weighted sum
        attn_weights = F.softmax(scores, dim=-1)
    
        attended_values = torch.bmm(attn_weights, padded_value)
        # reshape hidden_states
        attended_values = self.reshape_batch_dim_to_heads(attended_values)

        # linear proj
        hidden_states = self.to_out[0](attended_values)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
    
        return hidden_states
    
class SpatialTemporalAttention(CrossAttention):
    """
    Define a class `SpatialTemporalAttention` that inherits from `CrossAttention`. Implement a method `forward_dense_attn` that calculates attention between hidden states and encoder hidden states.
    @param hidden_states - The hidden states of the model
    @param encoder_hidden_states - The hidden states of the encoder
    @param attention_mask - The attention mask
    @param video_length - The length of the video
    @return The result of the attention calculation.
    """
    def forward_dense_attn(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        key = rearrange(key, "(b f) n d -> b f n d", f=video_length)
        key = key.unsqueeze(1).repeat(1, video_length, 1, 1, 1)  # (b f f n d)
        key = rearrange(key, "b f g n d -> (b f) (g n) d")

        value = rearrange(value, "(b f) n d -> b f n d", f=video_length)
        value = value.unsqueeze(1).repeat(1, video_length, 1, 1, 1)  # (b f f n d)
        value = rearrange(value, "b f g n d -> (b f) (g n) d")

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states
    
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, normal_infer=False):
        if normal_infer:
            return super().forward(
                hidden_states=hidden_states, 
                encoder_hidden_states=encoder_hidden_states, 
                attention_mask=attention_mask, 
                # video_length=video_length,
            )
        else:
            return self.forward_dense_attn(
                hidden_states=hidden_states, 
                encoder_hidden_states=encoder_hidden_states, 
                attention_mask=attention_mask, 
                video_length=video_length,
            )