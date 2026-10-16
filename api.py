# api.py
# 使用方法: python api.py --port 9933 --model_dir pretrained_models/CosyVoice2-0.5B

import os
import sys
import argparse
import io
import logging

import torch
import torchaudio
import numpy as np
from flask import Flask, request, send_file, jsonify

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{ROOT_DIR}/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
cosyvoice = None
default_sft_spk = None


def tensor_to_wav_bytes(tts_speech, sample_rate):
    """将 torch tensor 转为 wav 字节流"""
    buf = io.BytesIO()
    torchaudio.save(buf, tts_speech, sample_rate, format='wav')
    buf.seek(0)
    return buf


@app.route('/tts', methods=['POST'])
def tts():
    """
    使用预训练音色合成
    参数: text (必须), role (可选, 默认为第一个可用音色)
    返回: wav 音频文件
    """
    data = request.form
    text = data.get('text', '').strip()
    role = data.get('role', '').strip() or default_sft_spk

    if not text:
        return jsonify({'error': 'text is required'}), 400
    if not role:
        return jsonify({'error': 'no available speaker'}), 400

    try:
        set_all_random_seed(42)
        result = next(cosyvoice.inference_sft(text, role, stream=False, speed=1.0))
        buf = tensor_to_wav_bytes(result['tts_speech'], cosyvoice.sample_rate)
        return send_file(buf, mimetype='audio/wav')
    except Exception as e:
        logger.error(f'TTS error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/clone_eq', methods=['POST'])
def clone_eq():
    """
    同语言音色克隆 (3s极速复刻)
    参数: text (必须), reference_audio (必须, 文件路径或上传文件),
          reference_text (必须, 参考音频对应的文字)
    返回: wav 音频文件
    """
    text = request.form.get('text', '').strip()
    reference_text = request.form.get('reference_text', '').strip()
    reference_audio_path = request.form.get('reference_audio', '').strip()

    # 支持文件上传
    if 'reference_audio_file' in request.files:
        file = request.files['reference_audio_file']
        reference_audio_path = f'/tmp/cosyvoice_prompt_{os.getpid()}.wav'
        file.save(reference_audio_path)

    if not text:
        return jsonify({'error': 'text is required'}), 400
    if not reference_audio_path or not os.path.exists(reference_audio_path):
        return jsonify({'error': 'reference_audio not found'}), 400
    if not reference_text:
        return jsonify({'error': 'reference_text is required'}), 400

    try:
        set_all_random_seed(42)
        result = next(cosyvoice.inference_zero_shot(text, reference_text, reference_audio_path,
                                                     stream=False, speed=1.0))
        buf = tensor_to_wav_bytes(result['tts_speech'], cosyvoice.sample_rate)
        return send_file(buf, mimetype='audio/wav')
    except Exception as e:
        logger.error(f'Clone eq error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/clone_cross', methods=['POST'])
def clone_cross():
    """
    跨语言音色克隆
    参数: text (必须), reference_audio (必须, 文件路径或上传文件)
    返回: wav 音频文件
    """
    text = request.form.get('text', '').strip()
    reference_audio_path = request.form.get('reference_audio', '').strip()

    if 'reference_audio_file' in request.files:
        file = request.files['reference_audio_file']
        reference_audio_path = f'/tmp/cosyvoice_prompt_{os.getpid()}.wav'
        file.save(reference_audio_path)

    if not text:
        return jsonify({'error': 'text is required'}), 400
    if not reference_audio_path or not os.path.exists(reference_audio_path):
        return jsonify({'error': 'reference_audio not found'}), 400

    try:
        set_all_random_seed(42)
        result = next(cosyvoice.inference_cross_lingual(text, reference_audio_path,
                                                         stream=False, speed=1.0))
        buf = tensor_to_wav_bytes(result['tts_speech'], cosyvoice.sample_rate)
        return send_file(buf, mimetype='audio/wav')
    except Exception as e:
        logger.error(f'Clone cross error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/instruct', methods=['POST'])
def instruct():
    """
    自然语言控制 (需要 Instruct 模型)
    参数: text (必须), role (必须), instruct_text (必须)
    返回: wav 音频文件
    """
    text = request.form.get('text', '').strip()
    role = request.form.get('role', '').strip()
    instruct_text = request.form.get('instruct_text', '').strip()

    if not text or not role or not instruct_text:
        return jsonify({'error': 'text, role, and instruct_text are required'}), 400

    try:
        set_all_random_seed(42)
        result = next(cosyvoice.inference_instruct(text, role, instruct_text,
                                                    stream=False, speed=1.0))
        buf = tensor_to_wav_bytes(result['tts_speech'], cosyvoice.sample_rate)
        return send_file(buf, mimetype='audio/wav')
    except Exception as e:
        logger.error(f'Instruct error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/v1/audio/speech', methods=['POST'])
def openai_compatible():
    """
    兼容 OpenAI TTS API
    JSON 参数: model, input, voice, response_format, speed
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON'}), 400

    text = data.get('input', '').strip()
    voice = data.get('voice', '').strip()

    if not text:
        return jsonify({'error': 'input is required'}), 400

    try:
        set_all_random_seed(42)
        # 如果 voice 是文件路径，走克隆模式
        if voice and os.path.exists(voice):
            result = next(cosyvoice.inference_cross_lingual(text, voice, stream=False, speed=1.0))
        else:
            # 否则当作预训练音色名
            role = voice if voice else default_sft_spk
            result = next(cosyvoice.inference_sft(text, role, stream=False, speed=1.0))

        buf = tensor_to_wav_bytes(result['tts_speech'], cosyvoice.sample_rate)
        return send_file(buf, mimetype='audio/wav')
    except Exception as e:
        logger.error(f'OpenAI TTS error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/speakers', methods=['GET'])
def list_speakers():
    """列出可用的预训练音色"""
    return jsonify({'speakers': cosyvoice.list_available_spks()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9933)
    parser.add_argument('--model_dir', type=str, default='pretrained_models/Fun-CosyVoice3-0.5B')
    args = parser.parse_args()

    logger.info(f'Loading model from {args.model_dir}')
    cosyvoice = AutoModel(model_dir=args.model_dir)

    spks = cosyvoice.list_available_spks()
    default_sft_spk = spks[0] if spks else None
    logger.info(f'Available speakers: {spks}')

    app.run(host='0.0.0.0', port=args.port, threaded=False)