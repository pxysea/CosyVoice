# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
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
"""CosyVoice WebUI（支持 CosyVoice / CosyVoice2 / CosyVoice3，兼容新版 torchaudio / gradio）

功能:
  - 语音合成: 预训练音色(v1)、3s极速复刻、跨语种复刻、自然语言控制(instruct / instruct2)
  - 音色库: 把 prompt 音频 + 文本保存成命名音色（持久化到磁盘），可列表、试听、删除，合成时直接选用
  - CosyVoice3 前缀自动补全: 'You are a helpful assistant.<|endofprompt|>' 等
  - 每次合成自动保存到 outputs/，历史记录可回放，提供下载
  - 语音转换(VC)
  - 随机种子（可固定复现）、调速、流式推理、文本正则化开关

用法:
  python webui.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B --port 8007
"""
import os
import sys
import re
import json
import time
import random
import argparse
import threading
from datetime import datetime

import numpy as np
import soundfile as sf
import torch
import gradio as gr

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed

# ----------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------
MODE_SFT = '预训练音色'
MODE_ZS = '3s极速复刻'
MODE_CL = '跨语种复刻'
MODE_INS = '自然语言控制'

END_TAG = '<|endofprompt|>'
V3_PREFIX = 'You are a helpful assistant.' + END_TAG
MAX_PROMPT_SEC = 30          # prompt 音频最长 30s（frontend 里有断言）
MIN_PROMPT_SR = 16000        # prompt 音频采样率下限
DEFAULT_TEXT = '我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。'
NO_VOICE_LABEL = '— 不使用（使用下方上传/录制的 prompt）—'

# 模型是线程共享的，推理与音色增删都要串行
INFER_LOCK = threading.Lock()
LIB_LOCK = threading.Lock()

# 运行期全局状态，在 init_runtime() 里赋值
cosyvoice = None
MODEL_KIND = 'v1'            # v1 = CosyVoice, v2 = CosyVoice2, v3 = CosyVoice3
SAMPLE_RATE = 22050
LIB_DIR = ''
OUT_DIR = ''
LIB_IDS = set()              # 来自音色库目录的音色名（可删除）


# ----------------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------------
def _skip():
    """gradio 里表示"该输出保持不变"，兼容不同版本"""
    return gr.skip() if hasattr(gr, 'skip') else gr.update()


def is_zero_shot_spk(info):
    return isinstance(info, dict) and 'prompt_speech_feat' in info


def is_sft_spk(info):
    return isinstance(info, dict) and 'embedding' in info and 'prompt_speech_feat' not in info


def sft_speakers():
    return [k for k, v in cosyvoice.frontend.spk2info.items() if is_sft_spk(v)]


def saved_voices():
    return sorted(k for k, v in cosyvoice.frontend.spk2info.items() if is_zero_shot_spk(v))


def voice_choices():
    return [(NO_VOICE_LABEL, '')] + [(n, n) for n in saved_voices()]


def check_audio(path, label='prompt音频'):
    """检查音频可读、采样率、时长；不合格直接抛 ValueError（带中文提示）"""
    if not path:
        raise ValueError('{}为空，您是否忘记输入{}？'.format(label, label))
    try:
        info = sf.info(path)
    except Exception as e:
        raise ValueError('无法读取{}（支持 wav / flac / mp3 / ogg，不支持 m4a、aac）：{}'.format(label, e))
    if info.samplerate < MIN_PROMPT_SR:
        raise ValueError('{}采样率{}低于{}'.format(label, info.samplerate, MIN_PROMPT_SR))
    duration = info.frames / float(info.samplerate)
    if duration > MAX_PROMPT_SEC:
        raise ValueError('{}时长 {:.1f}s 超过 {}s 上限'.format(label, duration, MAX_PROMPT_SEC))
    return duration


def fix_prompt_text(text, auto):
    """零样本复刻的 prompt 文本：CosyVoice3 必须带 'You are a helpful assistant.<|endofprompt|>' 前缀"""
    text = (text or '').strip()
    if auto and MODEL_KIND == 'v3' and END_TAG not in text:
        text = V3_PREFIX + text
    return text


def fix_cross_text(text, auto):
    """跨语种复刻的合成文本：CosyVoice3 同样需要前缀"""
    text = (text or '').strip()
    if auto and MODEL_KIND == 'v3' and END_TAG not in text:
        text = V3_PREFIX + text
    return text


def fix_instruct_text(text, auto):
    """instruct 文本：以 <|endofprompt|> 结尾；CosyVoice3 还需要 'You are a helpful assistant. ' 开头"""
    text = (text or '').strip()
    if not auto or END_TAG in text:
        return text
    if MODEL_KIND == 'v3':
        return 'You are a helpful assistant. {}{}'.format(text, END_TAG)
    return text + END_TAG


# ----------------------------------------------------------------------------
# 音色库（持久化到 LIB_DIR，不改动模型目录里的 spk2info.pt）
#   <name>.pt   说话人信息（tensor 字典，与 add_zero_shot_spk 产物一致）
#   <name>.wav  prompt 音频副本（试听；自然语言控制模式也要用它）
#   <name>.json 元信息（prompt 文本、创建时间、时长）
# ----------------------------------------------------------------------------
def _voice_paths(name):
    base = os.path.join(LIB_DIR, name)
    return base + '.pt', base + '.wav', base + '.json'


def read_meta(name):
    try:
        with open(_voice_paths(name)[2], 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def voice_wav_path(name):
    p = _voice_paths(name)[1]
    return p if os.path.exists(p) else None


def load_library():
    """把音色库目录里的音色载入 frontend.spk2info（可重复调用，已载入的会跳过）"""
    loaded = 0
    for f in sorted(os.listdir(LIB_DIR)):
        if not f.endswith('.pt'):
            continue
        name = f[:-3]
        if name in LIB_IDS and name in cosyvoice.frontend.spk2info:
            continue
        if name in cosyvoice.frontend.spk2info and name not in LIB_IDS:
            logging.warning('音色库中的 {} 与模型自带音色重名，已跳过'.format(name))
            continue
        try:
            spk = torch.load(os.path.join(LIB_DIR, f), map_location=cosyvoice.frontend.device, weights_only=True)
            if not is_zero_shot_spk(spk):
                raise ValueError('不是有效的音色文件')
            cosyvoice.frontend.spk2info[name] = spk
            LIB_IDS.add(name)
            loaded += 1
        except Exception as e:
            logging.warning('载入音色 {} 失败: {}'.format(f, e))
    return loaded


def validate_voice_name(name):
    name = (name or '').strip()
    if not re.fullmatch(r'[\w\-]{1,40}', name):
        raise gr.Error('音色名称只能包含中文、字母、数字、下划线和短横线，长度 1~40')
    return name


def do_save_voice(name, wav_up, wav_rec, prompt_text, overwrite, auto_prefix):
    name = validate_voice_name(name)
    wav = wav_up or wav_rec
    try:
        duration = check_audio(wav, 'prompt音频')
    except ValueError as e:
        raise gr.Error(str(e))
    if not (prompt_text or '').strip():
        raise gr.Error('prompt文本为空，您是否忘记输入prompt文本？')
    text = fix_prompt_text(prompt_text, auto_prefix)
    with LIB_LOCK:
        existing = cosyvoice.frontend.spk2info.get(name)
        if existing is not None and name not in LIB_IDS:
            raise gr.Error('名称 {} 与模型自带音色重名，请换一个名称'.format(name))
        if existing is not None and not overwrite:
            raise gr.Error('音色 {} 已存在，勾选“覆盖同名音色”后重试'.format(name))
        try:
            with INFER_LOCK:
                cosyvoice.add_zero_shot_spk(text, wav, name)
        except Exception as e:
            logging.exception('保存音色失败')
            raise gr.Error('提取音色失败：{}'.format(e))
        spk = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
               for k, v in cosyvoice.frontend.spk2info[name].items()}
        pt, wav_copy, js = _voice_paths(name)
        torch.save(spk, pt)
        data, sr = sf.read(wav, dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=1)
        sf.write(wav_copy, data, sr, subtype='PCM_16')
        meta = {'name': name, 'prompt_text': text,
                'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'duration': round(duration, 2),
                'model': os.path.basename(os.path.normpath(MODEL_DIR))}
        with open(js, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        LIB_IDS.add(name)
    return name


def do_delete_voice(name):
    if not name:
        raise gr.Error('请先选择要删除的音色')
    with LIB_LOCK:
        with INFER_LOCK:
            if name not in LIB_IDS:
                raise gr.Error('只能删除音色库里保存的音色（模型自带或 spk2info.pt 里的音色不可删除）')
            cosyvoice.frontend.spk2info.pop(name, None)
            for p in _voice_paths(name):
                if os.path.exists(p):
                    os.remove(p)
            LIB_IDS.discard(name)


def voice_rows():
    rows = []
    for name in saved_voices():
        in_lib = name in LIB_IDS
        meta = read_meta(name) if in_lib else {}
        rows.append([name, '音色库' if in_lib else '模型目录 spk2info.pt',
                     meta.get('created', ''), meta.get('duration', ''),
                     (meta.get('prompt_text', '') or '')[:60]])
    return rows


VOICE_HEADERS = ['名称', '来源', '创建时间', '时长(秒)', 'prompt 文本']


# ----------------------------------------------------------------------------
# 输出保存与历史记录
# ----------------------------------------------------------------------------
def save_output(audio, mode, voice, text, seed):
    now = datetime.now()
    day_dir = os.path.join(OUT_DIR, now.strftime('%Y%m%d'))
    os.makedirs(day_dir, exist_ok=True)
    safe = re.sub(r'[^\w\-]+', '_', text.replace(V3_PREFIX, ''))[:20].strip('_') or 'audio'
    base = os.path.join(day_dir, '{}_{}'.format(now.strftime('%H%M%S'), safe))
    path, n = base + '.wav', 1
    while os.path.exists(path):
        n += 1
        path = '{}_{}.wav'.format(base, n)
    sf.write(path, np.clip(audio, -1.0, 1.0), SAMPLE_RATE, subtype='PCM_16')
    meta = {'time': now.strftime('%Y-%m-%d %H:%M:%S'), 'mode': mode, 'voice': voice,
            'seed': seed, 'duration': round(len(audio) / float(SAMPLE_RATE), 2), 'text': text}
    with open(path[:-4] + '.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return path


def history_choices(limit=50):
    files = []
    for root, _, names in os.walk(OUT_DIR):
        for n in names:
            if n.endswith('.wav'):
                p = os.path.join(root, n)
                files.append((os.path.getmtime(p), p))
    files.sort(reverse=True)
    choices = []
    for _, p in files[:limit]:
        try:
            with open(p[:-4] + '.json', 'r', encoding='utf-8') as f:
                m = json.load(f)
        except Exception:
            m = {}
        label = '{} | {} | {} | {}'.format(m.get('time', os.path.basename(p)), m.get('mode', ''),
                                           m.get('voice', ''), (m.get('text', '') or '').replace('\n', ' ')[:24])
        choices.append((label, p))
    return choices


# ----------------------------------------------------------------------------
# 合成
# ----------------------------------------------------------------------------
def available_modes():
    modes = []
    if MODEL_KIND == 'v1' and sft_speakers():
        modes.append(MODE_SFT)
    modes += [MODE_ZS, MODE_CL]
    if MODEL_KIND != 'v1' or sft_speakers():
        modes.append(MODE_INS)
    return modes


def instruct_for(mode):
    if mode == MODE_SFT:
        return '1. 选择预训练音色\n2. 点击生成音频按钮'
    if mode == MODE_ZS:
        return ('1. 选择已保存音色；或选择/录入 prompt 音频（不超过30s，两者都有时优先上传文件）\n'
                '2. 使用 prompt 音频时需输入与音频内容一致的 prompt 文本（CosyVoice3 会自动补全前缀）\n'
                '3. 点击生成音频按钮')
    if mode == MODE_CL:
        return '1. 选择已保存音色，或选择/录入 prompt 音频（不超过30s）\n2. 合成文本与 prompt 使用不同语言\n3. 点击生成音频按钮'
    if MODEL_KIND == 'v1':
        return '1. 选择预训练音色\n2. 输入instruct文本\n3. 点击生成音频按钮'
    return ('1. 选择已保存音色，或选择/录入 prompt 音频（不超过30s）\n'
            '2. 输入instruct文本，例如：请用广东话表达\n3. 点击生成音频按钮')


def mode_flags(mode):
    return {
        'sft': mode == MODE_SFT or (mode == MODE_INS and MODEL_KIND == 'v1'),
        'prompt': mode in (MODE_ZS, MODE_CL) or (mode == MODE_INS and MODEL_KIND != 'v1'),
        'prompt_text': mode == MODE_ZS,
        'instruct': mode == MODE_INS,
    }


def on_mode_change(mode):
    fl = mode_flags(mode)
    return (gr.Dropdown(visible=fl['sft']),
            gr.Column(visible=fl['prompt']),
            gr.Textbox(visible=fl['prompt_text']),
            gr.Textbox(visible=fl['instruct']),
            gr.Textbox(value=instruct_for(mode)))


def build_iterator(mode, tts_text, sft_spk, saved_voice, prompt_wav, prompt_text, instruct_text,
                   stream, speed, text_frontend, auto_prefix):
    """按模式构造推理迭代器，参数不合法时抛 ValueError。返回 (iterator, 音色标签)"""
    kw = dict(stream=stream, speed=speed, text_frontend=text_frontend)

    if mode == MODE_SFT:
        if not sft_spk:
            raise ValueError('没有可用的预训练音色！')
        return cosyvoice.inference_sft(tts_text, sft_spk, **kw), sft_spk

    if mode == MODE_INS and MODEL_KIND == 'v1':
        if not sft_spk:
            raise ValueError('没有可用的预训练音色！')
        if not (instruct_text or '').strip():
            raise ValueError('您正在使用自然语言控制模式, 请输入instruct文本')
        return cosyvoice.inference_instruct(tts_text, sft_spk, fix_instruct_text(instruct_text, auto_prefix), **kw), sft_spk

    # 以下模式都需要 prompt：已保存音色 或 临时 prompt 音频
    use_saved = bool(saved_voice)
    if use_saved and saved_voice not in cosyvoice.frontend.spk2info:
        raise ValueError('音色 {} 不存在，请点击“刷新音色”'.format(saved_voice))

    if mode == MODE_ZS:
        if use_saved:
            return cosyvoice.inference_zero_shot(tts_text, '', '', zero_shot_spk_id=saved_voice, **kw), saved_voice
        check_audio(prompt_wav)
        if not (prompt_text or '').strip():
            raise ValueError('prompt文本为空，您是否忘记输入prompt文本？')
        return cosyvoice.inference_zero_shot(tts_text, fix_prompt_text(prompt_text, auto_prefix), prompt_wav, **kw), '临时prompt'

    if mode == MODE_CL:
        text = fix_cross_text(tts_text, auto_prefix)
        if use_saved:
            return cosyvoice.inference_cross_lingual(text, '', zero_shot_spk_id=saved_voice, **kw), saved_voice
        check_audio(prompt_wav)
        return cosyvoice.inference_cross_lingual(text, prompt_wav, **kw), '临时prompt'

    # MODE_INS（CosyVoice2 / CosyVoice3 的 instruct2）
    if not (instruct_text or '').strip():
        raise ValueError('您正在使用自然语言控制模式, 请输入instruct文本')
    if use_saved:
        # 用缓存的 spk id 时 frontend 会忽略 instruct 文本，所以这里改用音色库里保存的 prompt 音频
        prompt_wav = voice_wav_path(saved_voice)
        if not prompt_wav:
            raise ValueError('音色 {} 没有保存 prompt 音频，自然语言控制模式无法使用，请在音色库重新保存'.format(saved_voice))
        label = saved_voice
    else:
        label = '临时prompt'
    check_audio(prompt_wav)
    return cosyvoice.inference_instruct2(tts_text, fix_instruct_text(instruct_text, auto_prefix), prompt_wav, **kw), label


def generate_audio(tts_text, mode, sft_spk, saved_voice, wav_up, wav_rec, prompt_text, instruct_text,
                   seed, random_seed, stream, speed, text_frontend, auto_prefix):
    tts_text = (tts_text or '').strip()
    if not tts_text:
        raise gr.Error('请输入需要合成的文本')
    prompt_wav = wav_up or wav_rec or None
    speed = float(speed or 1.0)
    if stream and speed != 1.0:
        gr.Info('流式推理不支持调速，已按 1.0 倍速生成')
        speed = 1.0
    seed = random.randint(1, 100000000) if random_seed else int(seed or 0)

    chunks = []
    try:
        with INFER_LOCK:
            set_all_random_seed(seed)
            iterator, voice_label = build_iterator(mode, tts_text, sft_spk, saved_voice, prompt_wav, prompt_text,
                                                   instruct_text, stream, speed, text_frontend, auto_prefix)
            for out in iterator:
                wav = out['tts_speech'].detach().cpu().float().numpy().flatten()
                chunks.append(wav)
                if stream:
                    yield (SAMPLE_RATE, wav), _skip(), seed, _skip()
    except ValueError as e:
        raise gr.Error(str(e))
    except gr.Error:
        raise
    except Exception as e:
        logging.exception('合成失败')
        raise gr.Error('合成失败：{}'.format(e))

    if not chunks:
        raise gr.Error('没有生成任何音频，请检查输入文本')
    full = np.concatenate(chunks)
    path = save_output(full, mode, voice_label, tts_text, seed)
    history = gr.Dropdown(choices=history_choices())
    if stream:
        yield _skip(), path, seed, history
    else:
        yield (SAMPLE_RATE, full), path, seed, history


def convert_voice(src_up, src_rec, prm_up, prm_rec, saved_voice):
    """语音转换：把源音频转成 prompt 音色（CosyVoice2/3 为实验性支持）"""
    src = src_up or src_rec
    try:
        check_audio(src, '源音频')
        if saved_voice:
            prm = voice_wav_path(saved_voice)
            if not prm:
                raise ValueError('音色 {} 没有保存 prompt 音频，请改用上传/录制 prompt'.format(saved_voice))
            voice_label = saved_voice
        else:
            prm = prm_up or prm_rec
            voice_label = '临时prompt'
        check_audio(prm, 'prompt音频')
    except ValueError as e:
        raise gr.Error(str(e))
    try:
        with INFER_LOCK:
            set_all_random_seed(random.randint(1, 100000000))
            chunks = [o['tts_speech'].detach().cpu().float().numpy().flatten()
                      for o in cosyvoice.inference_vc(src, prm, stream=False, speed=1.0)]
    except Exception as e:
        logging.exception('语音转换失败')
        raise gr.Error('语音转换失败（当前模型可能不支持）：{}'.format(e))
    if not chunks:
        raise gr.Error('没有生成任何音频')
    full = np.concatenate(chunks)
    path = save_output(full, '语音转换', voice_label, os.path.basename(src), 0)
    return (SAMPLE_RATE, full), path, gr.Dropdown(choices=history_choices())


# ----------------------------------------------------------------------------
# 界面
# ----------------------------------------------------------------------------
def build_ui():
    modes = available_modes()
    sft_list = sft_speakers()
    fl = mode_flags(modes[0])

    with gr.Blocks(title='CosyVoice WebUI') as demo:
        gr.Markdown('### CosyVoice WebUI　[代码库](https://github.com/pxysea/CosyVoice)')
        gr.Markdown('模型：`{}`（{}）｜采样率 {} Hz｜音色库：`{}`｜输出目录：`{}`'.format(
            MODEL_DIR, type(cosyvoice).__name__, SAMPLE_RATE, LIB_DIR, OUT_DIR))

        with gr.Tabs():
            # ---------------- 语音合成 ----------------
            with gr.Tab('语音合成'):
                tts_text = gr.Textbox(label='输入合成文本', lines=3, value=DEFAULT_TEXT)
                with gr.Row():
                    mode = gr.Radio(choices=modes, label='选择推理模式', value=modes[0])
                    instruction_text = gr.Textbox(label='操作步骤', value=instruct_for(modes[0]), lines=3,
                                                  interactive=False)
                sft_dd = gr.Dropdown(choices=sft_list, label='选择预训练音色',
                                     value=sft_list[0] if sft_list else None, visible=fl['sft'])
                with gr.Column(visible=fl['prompt']) as prompt_col:
                    with gr.Row():
                        saved_dd = gr.Dropdown(choices=voice_choices(), value='', allow_custom_value=True,
                                               label='选择已保存音色（选择后忽略下方 prompt 音频/文本）')
                        refresh_btn = gr.Button('刷新音色', scale=0, min_width=100)
                    with gr.Row():
                        wav_up = gr.Audio(sources=['upload'], type='filepath',
                                          label='选择prompt音频文件，注意采样率不低于16khz，时长不超过30s')
                        wav_rec = gr.Audio(sources=['microphone'], type='filepath', label='录制prompt音频文件')
                prompt_text = gr.Textbox(label='输入prompt文本', lines=1, visible=fl['prompt_text'],
                                         placeholder='请输入prompt文本，需与prompt音频内容一致，暂时不支持自动识别...')
                instruct_text = gr.Textbox(label='输入instruct文本', lines=1, visible=fl['instruct'],
                                           placeholder='例如：请用广东话表达（CosyVoice3 会自动补全前缀和结束符）')

                with gr.Accordion('高级选项', open=False):
                    with gr.Row():
                        stream = gr.Checkbox(label='流式推理', value=False)
                        text_frontend = gr.Checkbox(label='文本正则化', value=True)
                        auto_prefix = gr.Checkbox(label='自动补全前缀/结束符（CosyVoice3 必需）', value=True)
                    with gr.Row():
                        speed = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0,
                                          label='速度调节（仅支持非流式推理）')
                        seed = gr.Number(value=0, label='随机推理种子', precision=0)
                        random_seed = gr.Checkbox(label='每次随机种子', value=True)

                generate_btn = gr.Button('生成音频', variant='primary')
                audio_output = gr.Audio(label='合成音频', autoplay=True, streaming=True, interactive=False)
                file_output = gr.File(label='下载（已自动保存到输出目录）', interactive=False)

                with gr.Accordion('把当前 prompt 保存为音色', open=False):
                    gr.Markdown('使用上方的 prompt 音频和 prompt 文本。保存后可在“选择已保存音色”中直接使用，无需每次重新上传。')
                    with gr.Row():
                        quick_name = gr.Textbox(label='音色名称', placeholder='中文、字母、数字、下划线、短横线')
                        quick_overwrite = gr.Checkbox(label='覆盖同名音色', value=False)
                        quick_btn = gr.Button('保存为音色', scale=0, min_width=120)
                    quick_status = gr.Markdown()

            # ---------------- 音色库 ----------------
            with gr.Tab('音色库'):
                gr.Markdown('音色保存在 `{}`，每个音色包含 `.pt`（说话人信息）、`.wav`（prompt 音频）、`.json`（元信息）。'
                            '音色与模型绑定，更换模型后需重新保存。'.format(LIB_DIR))
                voice_table = gr.Dataframe(value=voice_rows(), headers=VOICE_HEADERS, interactive=False, wrap=True)
                with gr.Row():
                    lib_dd = gr.Dropdown(choices=saved_voices(), value=None, label='选择音色（试听 / 删除）')
                    lib_refresh = gr.Button('刷新列表', scale=0, min_width=100)
                    lib_delete = gr.Button('删除所选音色', variant='stop', scale=0, min_width=140)
                lib_preview = gr.Audio(label='prompt 音频试听', type='filepath', interactive=False)
                lib_prompt_text = gr.Textbox(label='prompt 文本', interactive=False)
                lib_status = gr.Markdown()
                with gr.Accordion('新建音色', open=False):
                    lib_name = gr.Textbox(label='音色名称', placeholder='中文、字母、数字、下划线、短横线')
                    with gr.Row():
                        lib_wav_up = gr.Audio(sources=['upload'], type='filepath',
                                              label='选择prompt音频文件（采样率不低于16khz，不超过30s）')
                        lib_wav_rec = gr.Audio(sources=['microphone'], type='filepath', label='录制prompt音频文件')
                    lib_new_text = gr.Textbox(label='prompt文本（需与音频内容一致）', lines=1)
                    with gr.Row():
                        lib_overwrite = gr.Checkbox(label='覆盖同名音色', value=False)
                        lib_save_btn = gr.Button('保存音色', variant='primary')

            # ---------------- 语音转换 ----------------
            with gr.Tab('语音转换'):
                gr.Markdown('把源音频的内容转换成目标音色。CosyVoice2 / CosyVoice3 上为实验性支持。')
                with gr.Row():
                    vc_src_up = gr.Audio(sources=['upload'], type='filepath', label='源音频（不超过30s）')
                    vc_src_rec = gr.Audio(sources=['microphone'], type='filepath', label='录制源音频')
                vc_saved = gr.Dropdown(choices=voice_choices(), value='', allow_custom_value=True,
                                       label='目标音色（选择已保存音色，或使用下方 prompt 音频）')
                with gr.Row():
                    vc_prm_up = gr.Audio(sources=['upload'], type='filepath', label='目标 prompt 音频')
                    vc_prm_rec = gr.Audio(sources=['microphone'], type='filepath', label='录制目标 prompt 音频')
                vc_btn = gr.Button('开始转换', variant='primary')
                vc_out = gr.Audio(label='转换结果', interactive=False)
                vc_file = gr.File(label='下载', interactive=False)

            # ---------------- 历史记录 ----------------
            with gr.Tab('历史记录'):
                gr.Markdown('每次合成都会保存到 `{}`（按日期分目录，同名 `.json` 记录文本、音色、种子）。'.format(OUT_DIR))
                with gr.Row():
                    history_dd = gr.Dropdown(choices=history_choices(), value=None, label='最近 50 条')
                    history_refresh = gr.Button('刷新', scale=0, min_width=100)
                history_audio = gr.Audio(label='回放', type='filepath', interactive=False)

        # ---------------- 事件 ----------------
        mode.change(on_mode_change, inputs=[mode],
                    outputs=[sft_dd, prompt_col, prompt_text, instruct_text, instruction_text], api_name=False)

        generate_btn.click(
            generate_audio,
            inputs=[tts_text, mode, sft_dd, saved_dd, wav_up, wav_rec, prompt_text, instruct_text,
                    seed, random_seed, stream, speed, text_frontend, auto_prefix],
            outputs=[audio_output, file_output, seed, history_dd], api_name='generate_audio')

        def refresh_voices(cur_saved, cur_lib, cur_vc):
            load_library()
            names = saved_voices()
            return (gr.Dropdown(choices=voice_choices(), value=cur_saved if cur_saved in names else ''),
                    gr.Dropdown(choices=names, value=cur_lib if cur_lib in names else None),
                    gr.Dropdown(choices=voice_choices(), value=cur_vc if cur_vc in names else ''),
                    voice_rows())

        refresh_inputs = [saved_dd, lib_dd, vc_saved]
        refresh_outputs = [saved_dd, lib_dd, vc_saved, voice_table]
        refresh_btn.click(refresh_voices, refresh_inputs, refresh_outputs, api_name=False)
        lib_refresh.click(refresh_voices, refresh_inputs, refresh_outputs, api_name=False)

        def after_save(name):
            names = saved_voices()
            return ('已保存音色：**{}**'.format(name),
                    gr.Dropdown(choices=voice_choices(), value=name),
                    gr.Dropdown(choices=names, value=name),
                    gr.Dropdown(choices=voice_choices(), value=name),
                    voice_rows())

        def quick_save(name, up, rec, text, overwrite, auto):
            return after_save(do_save_voice(name, up, rec, text, overwrite, auto))

        def lib_save(name, up, rec, text, overwrite, auto):
            return after_save(do_save_voice(name, up, rec, text, overwrite, auto))

        quick_btn.click(quick_save,
                        [quick_name, wav_up, wav_rec, prompt_text, quick_overwrite, auto_prefix],
                        [quick_status, saved_dd, lib_dd, vc_saved, voice_table], api_name=False)
        lib_save_btn.click(lib_save,
                           [lib_name, lib_wav_up, lib_wav_rec, lib_new_text, lib_overwrite, auto_prefix],
                           [lib_status, saved_dd, lib_dd, vc_saved, voice_table], api_name=False)

        def show_voice(name):
            if not name:
                return None, ''
            return voice_wav_path(name), read_meta(name).get('prompt_text', '')

        lib_dd.change(show_voice, [lib_dd], [lib_preview, lib_prompt_text], api_name=False)

        def delete_voice(name, cur_saved, cur_vc):
            do_delete_voice(name)
            names = saved_voices()
            return ('已删除音色：**{}**'.format(name),
                    gr.Dropdown(choices=voice_choices(), value=cur_saved if cur_saved in names else ''),
                    gr.Dropdown(choices=names, value=None),
                    gr.Dropdown(choices=voice_choices(), value=cur_vc if cur_vc in names else ''),
                    voice_rows())

        lib_delete.click(delete_voice, [lib_dd, saved_dd, vc_saved],
                         [lib_status, saved_dd, lib_dd, vc_saved, voice_table], api_name=False)

        vc_btn.click(convert_voice, [vc_src_up, vc_src_rec, vc_prm_up, vc_prm_rec, vc_saved],
                     [vc_out, vc_file, history_dd], api_name='convert_voice')

        history_refresh.click(lambda: gr.Dropdown(choices=history_choices()), None, history_dd, api_name=False)
        history_dd.change(lambda p: p, [history_dd], [history_audio], api_name=False)

        # 页面加载/刷新时重新同步音色库和历史记录
        demo.load(refresh_voices, refresh_inputs, refresh_outputs, api_name=False)
        demo.load(lambda: gr.Dropdown(choices=history_choices()), None, history_dd, api_name=False)

    demo.queue(max_size=8, default_concurrency_limit=2)
    return demo


# ----------------------------------------------------------------------------
# 启动
# ----------------------------------------------------------------------------
MODEL_DIR = ''


def init_runtime(args):
    global cosyvoice, MODEL_KIND, SAMPLE_RATE, LIB_DIR, OUT_DIR, MODEL_DIR
    MODEL_DIR = args.model_dir
    cosyvoice = AutoModel(model_dir=args.model_dir, fp16=args.fp16)
    cls = type(cosyvoice).__name__
    MODEL_KIND = {'CosyVoice3': 'v3', 'CosyVoice2': 'v2'}.get(cls, 'v1')
    SAMPLE_RATE = cosyvoice.sample_rate
    # 音色与模型绑定：按模型目录名分开存放
    LIB_DIR = os.path.join(os.path.abspath(args.library_dir), os.path.basename(os.path.normpath(args.model_dir)))
    OUT_DIR = os.path.abspath(args.output_dir)
    os.makedirs(LIB_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    n = load_library()
    logging.info('model={} kind={} sample_rate={} 载入音色库 {} 个'.format(cls, MODEL_KIND, SAMPLE_RATE, n))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8007)
    parser.add_argument('--model_dir', type=str, default='pretrained_models/Fun-CosyVoice3-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--fp16', action='store_true', help='使用 fp16 推理（请先确认 fp32 下合成正常）')
    parser.add_argument('--library_dir', type=str, default=os.path.join(ROOT_DIR, 'spk_library'), help='音色库目录')
    parser.add_argument('--output_dir', type=str, default=os.path.join(ROOT_DIR, 'outputs'), help='合成音频输出目录')
    parser.add_argument('--share', action='store_true', help='生成 gradio 公网分享链接')
    args = parser.parse_args()

    init_runtime(args)
    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share,
                allowed_paths=[LIB_DIR, OUT_DIR])