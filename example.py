import sys
from typing import Counter

import torch

from cosyvoice.llm import llm
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli import cosyvoice
from cosyvoice.cli.cosyvoice import AutoModel
import numpy as np
import soundfile as sf
import torch
from scipy.io.wavfile import write as wav_write


def cosyvoice_example():
    """ CosyVoice Usage, check https://fun-audio-llm.github.io/ for more details
    """
    cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M-SFT')
    # sft usage
    print(cosyvoice.list_available_spks())
    # change stream=True for chunk stream inference
    for i, j in enumerate(cosyvoice.inference_sft('你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？', '中文女', stream=False)):
        save_wav('sft_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M')
    # zero_shot usage
    for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav')):
        save_wav('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    # cross_lingual usage, <|zh|><|en|><|ja|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
    for i, j in enumerate(cosyvoice.inference_cross_lingual('<|en|>And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that\'s coming into the family is a reason why sometimes we don\'t buy the whole thing.',
                                                            './asset/cross_lingual_prompt.wav')):
        save_wav('cross_lingual_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    # vc usage
    for i, j in enumerate(cosyvoice.inference_vc('./asset/cross_lingual_prompt.wav', './asset/zero_shot_prompt.wav')):
        save_wav('vc_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M-Instruct')
    # instruct usage, support <laughter></laughter><strong></strong>[laughter][breath]
    for i, j in enumerate(cosyvoice.inference_instruct('在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。', '中文男',
                                                       'Theo \'Crimson\', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.<|endofprompt|>')):
        save_wav('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)


def cosyvoice2_example():
    """ CosyVoice2 Usage, check https://funaudiollm.github.io/cosyvoice2/ for more details
    """
    cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B')

    # 强制统一精度为 Float32
    if not torch.cuda.is_available() or True:  # 临时方案：直接强制
        # CosyVoice 的模型结构通常包含 llm, flow, hift 等组件
        if hasattr(cosyvoice.model, 'llm'):
            cosyvoice.model.llm.float()
        if hasattr(cosyvoice.model, 'flow'):
            cosyvoice.model.flow.float()
        if hasattr(cosyvoice.model, 'hift'):
            cosyvoice.model.hift.float()

    # NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
    # zero_shot usage
    # for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav')):
    #     save_wav('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # # save zero_shot spk for future usage
    # assert cosyvoice.add_zero_shot_spk('希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav', 'my_zero_shot_spk') is True
    # for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '', '', zero_shot_spk_id='my_zero_shot_spk')):
    #     save_wav('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    # cosyvoice.save_spkinfo()

    # # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
    # for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', './asset/zero_shot_prompt.wav')):
    #     save_wav('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # # instruct usage
    # for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', './asset/zero_shot_prompt.wav')):
    #     save_wav('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # # bistream usage, you can use generator as input, this is useful when using text llm model as input
    # # NOTE you should still have some basic sentence split logic because llm can not handle arbitrary sentence length
    # def text_generator():
    #     yield '收到好友从远方寄来的生日礼物，'
    #     yield '那份意外的惊喜与深深的祝福'
    #     yield '让我心中充满了甜蜜的快乐，'
    #     yield '笑容如花儿般绽放。'
    # for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav', stream=False)):
    #     save_wav('zero_shot_bistream_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

def save_wav(path, speech, sample_rate):
    if isinstance(speech, torch.Tensor):
        speech = speech.detach().float().cpu().numpy()

    audio = np.asarray(speech).squeeze()

    if audio.ndim != 1:
        raise ValueError(
            f"预期单声道音频，实际形状: {audio.shape}"
        )

    if not np.isfinite(audio).all():
        raise ValueError("音频包含 NaN 或 Inf")

    if audio.size == 0:
        raise ValueError("生成的音频为空")

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    duration = len(audio) / sample_rate

    print(
        f"{path}: duration={duration:.2f}s, "
        f"samples={len(audio)}, "
        f"peak={peak:.5f}, rms={rms:.5f}"
    )

    sf.write(
        path,
        audio,
        sample_rate,
        subtype="PCM_16"
    )

    print(f"音频已保存: {path}, 采样率: {sample_rate} Hz")

def cosyvoice3_example():
    """ CosyVoice3 Usage, check https://github.com/QwenAudio/CosyVoice/ for more details
    """
    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')    

    # cosyvoice.model.llm.float()

    # llm = cosyvoice.model.llm

    # print("LLM dtype:", dict(
    #     Counter(str(p.dtype) for p in llm.parameters())
    # ))

    # print("LLM device:", next(llm.parameters()).device)
    # print("model.fp16:", getattr(cosyvoice.model, "fp16", None))
    # print("LLM training mode:", llm.training)

    # for name in ("flow", "hift"):
    #     module = getattr(cosyvoice.model, name, None)
    #     if module is not None:
    #         print(
    #             name,
    #             dict(Counter(str(p.dtype) for p in module.parameters()))
    #         )
            
    # zero_shot usage
    for i, j in enumerate(cosyvoice.inference_zero_shot('八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。', 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
                                                        './asset/zero_shot_prompt.wav', stream=False)):
        save_wav('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L280
    for i, j in enumerate(cosyvoice.inference_cross_lingual('You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]',
                                                            './asset/zero_shot_prompt.wav', stream=False)):
        save_wav('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # instruct usage, for supported control, check cosyvoice/utils/common.py#L28
    for i, j in enumerate(cosyvoice.inference_instruct2('好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。', 'You are a helpful assistant. 请用广东话表达。<|endofprompt|>',
                                                        './asset/zero_shot_prompt.wav', stream=False)):
        save_wav('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', 'You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>',
                                                        './asset/zero_shot_prompt.wav', stream=False)):
        save_wav('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # hotfix usage
    for i, j in enumerate(cosyvoice.inference_zero_shot('高管也通过电话、短信、微信等方式对报道[j][ǐ]予好评。', 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
                                                        './asset/zero_shot_prompt.wav', stream=False)):
        save_wav('hotfix_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # NOTE for Japanese usage, you must translate it to katakana.
    # 歴史的世界においては、過去は単に過ぎ去ったものではない、プラトンのいう如く非有が有である。 -> レキシ テキ セカイ ニ オイ テ ワ、カコ ワ タンニ スギサッ タ モノ デ ワ ナイ、プラトン ノ イウ ゴトク ヒ ユー ガ ユー デ アル。
    for i, j in enumerate(cosyvoice.inference_cross_lingual('You are a helpful assistant.<|endofprompt|>レキシ テキ セカイ ニ オイ テ ワ、カコ ワ タンニ スギサッ タ モノ デ ワ ナイ、プラトン ノ イウ ゴトク ヒ ユー ガ ユー デ アル。',
                                                            './asset/zero_shot_prompt.wav', stream=False)):
        save_wav('japanese_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)


def main():
    # cosyvoice_example()
    # cosyvoice2_example()
    cosyvoice3_example()


if __name__ == '__main__':
    main()
