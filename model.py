```python
from math import sqrt

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModelForCausalLM
from layers.Embed import PatchEmbedding
import transformers


class Model(nn.Module):
    def __init__(self, llm, tokenizer):
        super(Model, self).__init__()

        # Model hyperparameters
        self.seq_len = 3
        self.d_ff = 3
        self.top_k = 5
        self.d_llm = 4096
        self.patch_len = 1
        self.stride = 1
        self.d_model = 64
        self.n_heads = 8
        self.enc_in = 1
        self.attention_dropout = 0.1

        # Prompt template with system instruction
        self.TEMPLATE = (
            "[INST] <<SYS>>\n"
            "You are a helpful assistant. 你是一个乐于助人的助手。\n"
            "<</SYS>>\n\n"
            "{instruction} [/INST]"
        )

        self.tokenizer = tokenizer
        self.llama = llm

        # Ensure pad_token is set
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = "[PAD]"
            self.tokenizer.add_special_tokens({"pad_token": pad_token})
            self.tokenizer.pad_token = pad_token

        self.dropout = nn.Dropout(self.attention_dropout).to(self.llama.device)

        # Patch embedding layer for time-series input
        self.patch_embedding1 = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, self.attention_dropout
        ).to(self.llama.device)

        # Prototype embedding layer: map LLM vocabulary to a lower-dimensional token set
        self.word_embeddings = self.llama.get_input_embeddings().weight.to(self.llama.device)
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 192
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens, dtype=self.llama.dtype).to(self.llama.device)

        # Reprogramming layer to align time-series embeddings with LLM embedding space
        self.reprogramming_layer = ReprogrammingLayer(
            self.d_model, self.n_heads, self.d_ff, self.d_llm, dtype=self.llama.dtype
        ).to(self.llama.device)

        self.patch_nums = int((self.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

    def forward(self, inputs_dict):
        # Prepare source embeddings (prototype tokens)
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0).to(self.llama.device)

        series = inputs_dict["tensor"]
        prompt = self.TEMPLATE.format(instruction=inputs_dict["instruction"])
        prompt = prompt.split("<unk>")

        # Encode the initial text prompt
        input_ids = self.tokenizer(prompt[0], return_tensors="pt").input_ids.to(self.llama.device)
        input_embeds = self.llama.get_input_embeddings()(input_ids)

        # For each additional segment (time-series + next part of the prompt)
        if len(prompt) > 1:
            for i in range(1, len(prompt)):
                tag = series[i - 1]["item"]
                x_enc = torch.tensor([series[i - 1]["tensor"]]).to(self.llama.device).to(torch.float16)

                # Compute patch embeddings from time-series data
                enc_out, n_vars = self.patch_embedding1(x_enc)
                enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

                # Encode the next text segment (skip leading newline tokens)
                input_ids = self.tokenizer("\n" + prompt[i], return_tensors="pt").input_ids[:, 2:]
                prompt_embeds = self.llama.get_input_embeddings()(input_ids.to(self.llama.device))

                # Concatenate embeddings: [previous text][time-series][next text]
                input_embeds = torch.cat([input_embeds, enc_out, prompt_embeds], dim=1)

        # Prepare labels tensor for LLM training (masking all but the target segment)
        batch_size, seq_len, _ = input_embeds.shape
        labels = -100 * torch.ones((batch_size, seq_len), dtype=torch.int32).to(self.llama.device)

        # Encode the output text (target) and append EOS token
        output = inputs_dict.get("output", "")
        output_ids = self.tokenizer("\n" + output, return_tensors="pt").input_ids[:, 2:].to(self.llama.device)
        eos_token_id = self.tokenizer.eos_token_id
        output_ids = torch.cat([output_ids, torch.tensor([[eos_token_id]]).to(self.llama.device)], dim=1)
        output_embeds = self.llama.get_input_embeddings()(output_ids)

        # Concatenate output embeddings to the input embeddings
        input_embeds = torch.cat([input_embeds, output_embeds], dim=1).to(self.llama.device)

        # Extend labels to cover the output segment
        labels = torch.cat([labels, output_ids.view(batch_size, -1)], dim=1).to(self.llama.device)

        # Forward pass through the LLM
        out = self.llama(
            inputs_embeds=input_embeds.half().to(self.llama.device),
            labels=labels.long().to(self.llama.device),
        )
        return out

    def generate(self, inputs_dict):
        # Prepare source embeddings (prototype tokens)
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0).to(self.llama.device)

        series = inputs_dict["tensor"]
        prompt = self.TEMPLATE.format(instruction=inputs_dict["instruction"])
        prompt = prompt.split("<unk>")

        # Encode the initial text prompt
        input_ids = self.tokenizer(prompt[0], return_tensors="pt").input_ids.to(self.llama.device)
        input_embeds = self.llama.get_input_embeddings()(input_ids)

        # For each additional segment (time-series + next part of the prompt)
        if len(prompt) > 1:
            for i in range(1, len(prompt)):
                tag = series[i - 1]["item"]
                x_enc = torch.tensor([series[i - 1]["tensor"]]).to(self.llama.device).to(torch.float16)
                x_enc = (x_enc - x_enc.mean()) / torch.sqrt(torch.var(x_enc) + 1e-5)

                # Compute patch embeddings from time-series data
                enc_out, n_vars = self.patch_embedding1(x_enc)
                enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

                # Encode the next text segment (skip leading newline tokens)
                input_ids = self.tokenizer("\n" + prompt[i], return_tensors="pt").input_ids[:, 2:]
                prompt_embeds = self.llama.get_input_embeddings()(input_ids.to(self.llama.device))

                # Concatenate embeddings: [previous text][time-series][next text]
                input_embeds = torch.cat([input_embeds, enc_out, prompt_embeds], dim=1)

        # Generate continuation from the LLM
        outputs = self.llama.generate(
            inputs_embeds=input_embeds.half().to(self.llama.device),
            max_new_tokens=512,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1, dtype=torch.float16):
        super(ReprogrammingLayer, self).__init__()

        # Projection dimensions
        d_keys = d_keys or (d_model // n_heads)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads, dtype=dtype)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads, dtype=dtype)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads, dtype=dtype)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm, dtype=dtype)

        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        """
        Perform a multi-head attention-like reprogramming:
        - target_embedding: [B, L, d_model]
        - source_embedding: [S, d_llm] (keys)
        - value_embedding: [S, d_llm] (values)
        """
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        # Project to query/key/value spaces and reshape for multi-head
        target_q = self.query_projection(target_embedding).view(B, L, H, -1)       # [B, L, H, E]
        source_k = self.key_projection(source_embedding).view(S, H, -1)            # [S, H, E]
        source_v = self.value_projection(value_embedding).view(S, H, -1)           # [S, H, E]

        # Compute attention scores and apply softmax
        scale = 1.0 / sqrt(target_q.shape[-1])
        scores = torch.einsum("blhe,she->bhls", target_q, source_k)                # [B, H, L, S]
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))                 # [B, H, L, S]

        # Weighted sum of values
        reprogramming_embedding = torch.einsum("bhls,she->blhe", attn, source_v)    # [B, L, H, E]
        reprogramming_embedding = reprogramming_embedding.reshape(B, L, -1)         # [B, L, H*E]

        # Project back to LLM embedding dimension
        return self.out_projection(reprogramming_embedding)                         # [B, L, d_llm]
```
