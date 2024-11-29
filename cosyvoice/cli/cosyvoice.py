# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
import torch
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel
from cosyvoice.utils.file_utils import logging


class CosyVoice:
    

    default_speaker_keys = ["flow_embedding", "llm_embedding", "llm_prompt_speech_token",
                            "llm_prompt_speech_token_len", "flow_prompt_speech_token",
                            "flow_prompt_speech_token_len", "prompt_speech_feat",
                            "prompt_speech_feat_len", "prompt_text", "prompt_text_len"]
    
    def __init__(self, model_dir, load_jit=True, load_onnx=False, fp16=True):
        instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          instruct,
                                          configs['allowed_special'])
        if torch.cuda.is_available() is False and (fp16 is True or load_jit is True):
            load_jit = False
            fp16 = False
            logging.warning('cpu do not support fp16 and jit, force set to False')
        self.model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/llm.text_encoder.fp16.zip'.format(model_dir),
                                '{}/llm.llm.fp16.zip'.format(model_dir),
                                '{}/flow.encoder.fp32.zip'.format(model_dir))
        if load_onnx:
            self.model.load_onnx('{}/flow.decoder.estimator.fp32.onnx'.format(model_dir))
        del configs

    def list_avaliable_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks
    
    def update_model_input(self,model_input, newspk, keys):
        for key in keys:
            if key in newspk:
                model_input[key] = newspk[key]

    def inference_sft1(self, tts_text, spk_id, stream=False, speed=1.0):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True)):
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0, speaker_data=None,update_keys=None):
        """
        合成方法，逐步返回音频段数据及其对应信息。

        参数:
            tts_text (str): 输入文本，待合成的内容。
            spk_id (str): 说话人 ID，支持默认预训练音色（如 '中文女', '中文男'）。
            stream (bool): 是否以流式方式返回合成结果。
            speed (float): 语速调整参数，默认值为 1.0（正常速度）。
            speaker_data (Optional[dict]): 如果提供，应该是 `torch.load()` 的返回值，表示自定义音色模型。
                                    通常是一个包含模型权重和特定参数的字典，例如:
                                    {
                                        "flow_embedding": Tensor,
                                        "llm_embedding": Tensor,
                                        ...
                                    }

        返回:
            generator: 每次生成一个字典，包含以下键值对:
                - "audio_chunk": Tensor, 当前生成的音频段。
                - "subtitle": dict, 对应的字幕信息，包含开始时间、结束时间和文本内容。
        """
        # 文本归一化
        text_segments = self.frontend.text_normalize(tts_text, split=True)
        start_time = time.time()
        for segment in tqdm(text_segments):
            # 模型输入处理
            if speaker_data:
                model_input = self.frontend.frontend_sft(segment, spk_id)
                self.update_model_input(model_input,speaker_data,self.default_speaker_keys)
            else:
                model_input = self.frontend.frontend_sft(segment, spk_id)

            if update_keys:
                self.update_model_input(model_input,speaker_data,update_keys)

            # 合成音频并处理输出
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                tts_speech = model_output['tts_speech']  # 当前段的音频数据
                
                yield {                    
                    "tts_speech": tts_speech,  # 音频段数据
                    "text_chunk": segment
                }
                start_time = time.time()

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, stream=False, speed=1.0):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True)):
            if len(i) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_cross_lingual(self, tts_text, prompt_speech_16k, stream=False, speed=1.0):
        if self.frontend.instruct is True:
            raise ValueError('{} do not support cross_lingual inference'.format(self.model_dir))
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True)):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_speech_16k)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_instruct(self, tts_text, spk_id, instruct_text, stream=False, speed=1.0):
        if self.frontend.instruct is False:
            raise ValueError('{} do not support instruct inference'.format(self.model_dir))
        instruct_text = self.frontend.text_normalize(instruct_text, split=False)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True)):
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_vc(self, source_speech_16k, prompt_speech_16k, stream=False, speed=1.0):
        model_input = self.frontend.frontend_vc(source_speech_16k, prompt_speech_16k)
        start_time = time.time()
        for model_output in self.model.vc(**model_input, stream=stream, speed=speed):
            speech_len = model_output['tts_speech'].shape[1] / 22050
            logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
            yield model_output
            start_time = time.time()
