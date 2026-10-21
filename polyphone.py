"""多音字 / 指定读音（CosyVoice3 的 Pronunciation Inpainting）

CosyVoice3 的分词器里有拼音特殊 token：声母 [y] [zh] ... + 带声调的韵母 [uè] [ǐ] ...，
用 "[声母][韵母]" 替换掉汉字即可强制读音，例如 不亦[y][uè]乎（见 example.py 的 [j][ǐ]）。
手写这种 token 容易出错，这里提供更友好的写法，合成前自动转换：

  1. 行内标注：字{拼音}，拼音可用声调符号或数字，ü 可写成 v
       学而时习之，不亦说{yuè}乎。   不亦说{yue4}乎   绿{lv4}
  2. 多音字词典 polyphone_dict.txt：每行一个带标注的词/句，合成文本里出现去掉标注后的原文时自动套用。
     修改后无需重启（按文件修改时间自动重载）。

只有 CosyVoice3 支持；其它模型调用 strip_marks() 去掉标注，只保留汉字。
"""
import os
import re
import threading

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DICT_PATH = os.path.join(ROOT_DIR, 'polyphone_dict.txt')

INITIALS = ('zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h',
            'j', 'q', 'x', 'r', 'z', 'c', 's', 'y', 'w')
TONE_MARKS = {'a': 'āáǎà', 'e': 'ēéěè', 'i': 'īíǐì', 'o': 'ōóǒò', 'u': 'ūúǔù', 'ü': 'ǖǘǚǜ'}
UNMARK = {m: (v, i + 1) for v, ms in TONE_MARKS.items() for i, m in enumerate(ms)}

# 字{拼音}：一个汉字后紧跟花括号拼音
MARK_RE = re.compile(r'([㐀-鿿]){\s*([a-zA-Zà-ǜ]+[0-5]?)\s*}')

_dict_lock = threading.Lock()
_dict_cache = {'mtime': None, 'items': []}


def _parse_syllable(py):
    """'yuè' / 'yue4' / 'lv4' -> (无调拼音, 声调 0~4，0 为轻声)"""
    py = py.strip().lower().replace('v', 'ü')
    tone = 0
    if py and py[-1].isdigit():
        tone, py = int(py[-1]) % 5, py[:-1]
    base = []
    for ch in py:
        if ch in UNMARK:
            ch, tone = UNMARK[ch]
        base.append(ch)
    return ''.join(base), tone


def _mark_tone(final, tone):
    """给韵母标声调：有 a 标 a，有 e 标 e，ou 标 o，否则标最后一个元音"""
    if tone == 0:
        return final
    if 'a' in final:
        i = final.index('a')
    elif 'e' in final:
        i = final.index('e')
    elif 'ou' in final:
        i = final.index('o')
    else:
        i = max(final.rfind(v) for v in 'iouü')
        if i < 0:
            return final
    return final[:i] + TONE_MARKS[final[i]][tone - 1] + final[i + 1:]


def pinyin_to_tokens(py, vocab):
    """拼音 -> '[声母][韵母]'，vocab 为分词器里的特殊 token 集合；不支持的拼音抛 ValueError"""
    base, tone = _parse_syllable(py)
    if not re.fullmatch(r'[a-zü]+', base):
        raise ValueError('拼音 {{{}}} 格式不对，应为 yuè / yue4 这样的单个音节'.format(py))
    initial = next((s for s in INITIALS if base.startswith(s) and len(base) > len(s)), '')
    final = base[len(initial):]
    if initial in ('j', 'q', 'x', 'y') and final.startswith('ü'):
        final = 'u' + final[1:]          # ju/qu/xu/yu 的 ü 写作 u
    marked = _mark_tone(final, tone)
    candidates = [marked, marked.replace('ü', 'v')]   # 词表里 lüè 写作 [vè]
    final_tok = next(('[{}]'.format(c) for c in candidates if '[{}]'.format(c) in vocab), None)
    init_tok = '[{}]'.format(initial) if initial else ''
    if final_tok is None or (init_tok and init_tok not in vocab):
        raise ValueError('拼音 {{{}}} 不在 CosyVoice3 的拼音词表里（韵母 [{}] 不支持），请换个写法或去掉标注'.format(py, marked))
    return init_tok + final_tok


def strip_marks(text):
    """去掉 字{拼音} 标注，只保留汉字（非 CosyVoice3 模型用）"""
    return MARK_RE.sub(r'\1', text)


def load_dict(path=DICT_PATH):
    """读取多音字词典，返回 [(原文, 带标注文本)]，按原文长度降序（长词优先）。文件变动自动重载"""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return []
    with _dict_lock:
        if _dict_cache['mtime'] != mtime:
            items = {}
            with open(path, encoding='utf-8') as f:
                for line in f:
                    line = line.split('#', 1)[0].strip()
                    if line and MARK_RE.search(line):
                        items[strip_marks(line)] = line
            _dict_cache.update(mtime=mtime, items=sorted(items.items(), key=lambda kv: -len(kv[0])))
        return _dict_cache['items']


def apply_dict(text, items=None):
    """把文本里出现的词典原文替换成带标注的写法（已替换的片段不会被更短的词条再次匹配）"""
    items = load_dict() if items is None else items
    if not items:
        return text
    pattern = re.compile('|'.join(re.escape(k) for k, _ in items))
    table = dict(items)
    return pattern.sub(lambda m: table[m.group(0)], text)


def apply_pinyin(text, vocab, use_dict=True):
    """词典 + 行内标注 -> CosyVoice3 拼音 token。例：不亦说{yuè}乎 -> 不亦[y][uè]乎"""
    if use_dict:
        text = apply_dict(text)
    return MARK_RE.sub(lambda m: pinyin_to_tokens(m.group(2), vocab), text)


def pinyin_vocab(tokenizer):
    """从 CosyVoice3 分词器取出拼音相关的特殊 token 集合（非 v3 返回空集）"""
    tk = getattr(tokenizer, 'tokenizer', tokenizer)
    try:
        return {t for t in tk.get_added_vocab() if t.startswith('[') and t.endswith(']')}
    except AttributeError:
        return set()
