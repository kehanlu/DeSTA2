from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, WhisperForConditionalGeneration, PretrainedConfig, PreTrainedModel, BertConfig, AutoProcessor
from transformers.models.bert.modeling_bert import BertEncoder
from torch import nn
import torch
import os
import librosa

class Desta2Config(PretrainedConfig):
    model_type = "DestaModel"

    def __init__(
        self,
        llama_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        whisper_model_id="openai/whisper-small",
        prompt_size=64,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.llama_model_id = llama_model_id
        self.whisper_model_id = whisper_model_id
        self.prompt_size = prompt_size

        self.whisper_config = AutoConfig.from_pretrained(self.whisper_model_id)
        self.llama_config = AutoConfig.from_pretrained(self.llama_model_id)

class QformerConnector(PreTrainedModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        
        
        if self.cfg.whisper_model_id == "openai/whisper-medium":
            self.target_layer_ids = [5, 11, 17, 23]
        elif self.cfg.whisper_model_id == "openai/whisper-small":
            self.target_layer_ids = [2, 5, 8, 11]
        elif self.cfg.whisper_model_id == "openai/whisper-tiny":
            self.target_layer_ids = [0,1,2,3]
        elif self.cfg.whisper_model_id == "openai/whisper-large-v3":
            self.target_layer_ids = [3, 7, 11, 15, 19, 23, 27, 31]
        else:
            raise NotImplementedError(f"model_id {self.cfg.whisper_model_id} not implemented")


        self.layer_prompts = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.cfg.prompt_size, self.cfg.whisper_config.d_model)) for _ in range(len(self.target_layer_ids))]
        )
        
        
        # (prompt_size, target_layers)
        self.layer_weights = nn.Parameter(torch.zeros(self.cfg.prompt_size, len(self.target_layer_ids), dtype=torch.float))

        qformer_config = BertConfig()
        qformer_config.num_hidden_layers = 2
        qformer_config.num_attention_heads = self.cfg.whisper_config.encoder_attention_heads
        qformer_config.hidden_size = self.cfg.whisper_config.d_model
        qformer_config.add_cross_attention = True
        qformer_config.is_decoder = True

        self.qformer = BertEncoder(qformer_config)
        self.proj = nn.Sequential(
                nn.LayerNorm(self.cfg.whisper_config.d_model),
                nn.Linear(self.cfg.whisper_config.d_model, self.cfg.llama_config.hidden_size) # project to llama hidden size
            )
    
    def forward(self, encoder_hidden_states):
        layer_prompt_outputs = []
        for idx, encoder_hidden_state in enumerate(encoder_hidden_states):
            if idx in self.target_layer_ids:
                layer_prompt = self.layer_prompts[self.target_layer_ids.index(idx)].expand(encoder_hidden_state.size(0), -1, -1)
                qformer_output = self.qformer(
                    hidden_states=layer_prompt,
                    encoder_hidden_states=encoder_hidden_state,
                )
                layer_prompt_output = qformer_output.last_hidden_state
                layer_prompt_outputs.append(layer_prompt_output)
        
        layer_prompt_outputs = torch.stack(layer_prompt_outputs, dim=0)
        layer_prompt_outputs = layer_prompt_outputs.permute(1, 2, 0, 3)
        
        self.norm_weights = torch.nn.functional.softmax(self.layer_weights, dim=-1).unsqueeze(-1)
        
        output = (layer_prompt_outputs * self.norm_weights).sum(dim=2) # (b, prompt_size, d_model)
        
        output = self.proj(output)
        
        return output

class SpeechPerception(PreTrainedModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

        self.whisper = WhisperForConditionalGeneration.from_pretrained(cfg.whisper_model_id)
        self.processor = AutoProcessor.from_pretrained(cfg.whisper_model_id)

        self.connector = QformerConnector(cfg)

    def generate(self, input_features):
        input_features = input_features.to(self.whisper.device)
        
        outputs = self.whisper.generate(input_features=input_features, return_dict_in_generate=True, output_hidden_states=True) # here we use default generate config for whisper

        transcriptions = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        speech_features = self.connector(outputs.encoder_hidden_states)

        return transcriptions, speech_features


class DestaModel(PreTrainedModel):
    config_class = Desta2Config

    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.speech_perception = SpeechPerception(config)
        self.llama = AutoModelForCausalLM.from_pretrained(config.llama_model_id, torch_dtype=torch.bfloat16, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(config.llama_model_id, **kwargs)
        

    def chat(self, messages, max_new_tokens=128, do_sample=True, temperature=0.6, top_p=0.9):
        """
        messages: list of dicts with keys "role" and "content"
        ```
        [
            {"role": "system", "content": "You are a helpful voice assistant."},
            {"role": "audio", "content": "<path_to_audio_file>"},
            {"role": "user", "content": "Describe the audio."}
        ]
        ```
        """

        audio_path, input_features = self.load_audio(messages)
        transcription, audio_features = self.speech_perception.generate(input_features)
        inputs, audio_position = self.process_text(messages, audio_path, transcription)

        inputs_embeds, attention_mask = self.prepare_llm_input(
            input_ids=inputs.input_ids, 
            attention_mask=inputs.attention_mask, 
            audio_position=audio_position,
            audio_features=audio_features
        )

        outputs = self.llama.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        return outputs

    def process_text(self, messages, audio_path, transcription):
        context = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        left_text, right_text = context.split(audio_path)
        right_text = transcription + right_text # 
        
        audio_position = len(self.tokenizer.tokenize(left_text))
        context = left_text + right_text

        inputs = self.tokenizer(context, return_tensors="pt")

        return inputs, audio_position


    def prepare_llm_input(self, input_ids, attention_mask, audio_position, audio_features):
        input_ids = input_ids.to(self.llama.device)
        attention_mask = attention_mask.to(self.llama.device)
        audio_features = audio_features.to(self.llama.device)
        audio_feature_length = audio_features.size(1)

        inputs_embeds = self.llama.model.embed_tokens(input_ids) # [bs, seq_len, hidden_size]


        inputs_embeds = torch.cat([inputs_embeds[0, :audio_position], audio_features[0, :], inputs_embeds[0, audio_position:]], dim=0)
        attention_mask = torch.cat([attention_mask[0, :audio_position], torch.ones([ audio_feature_length], dtype=torch.long, device=self.llama.device), attention_mask[0, audio_position:]], dim=0)

        inputs_embeds = inputs_embeds.to(self.llama.dtype)
        attention_mask = attention_mask.to(self.llama.dtype)
        return inputs_embeds.unsqueeze(0), attention_mask.unsqueeze(0)

    
    def load_audio(self, messages):
        audio_path = None
        for message in messages:
            if message["role"] == "audio" and audio_path is not None:
                raise ValueError("Multiple audio file paths found in messages. We only support one audio file per message at this moment.")
            if message["role"] == "audio":
                audio_path = message["content"]
        if audio_path is None:
            raise ValueError("No audio file path found in messages")
        audio, ori_sr = librosa.load(audio_path)
        audio = librosa.resample(audio, orig_sr=ori_sr, target_sr=16000)
        input_features = self.speech_perception.processor(audio, sampling_rate=16000, return_tensors="pt").input_features

        return audio_path, input_features
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config=None,**kwargs):
        config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = cls(config, **kwargs)

        if os.path.isdir(pretrained_model_name_or_path):
            model.speech_perception.connector.load_state_dict(
                torch.load(os.path.join(pretrained_model_name_or_path, "qformer_connector.pth"))
            )
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="qformer_connector.pth")
            model.speech_perception.connector.load_state_dict(
                torch.load(path)
            )

        return model
    