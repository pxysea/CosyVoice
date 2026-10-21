"""CosyVoice 统一服务：同一进程同时提供 REST API (/api) 与原 Gradio WebUI (/ui)。
共用模型、音色库(spk_library/<模型>/)、推理锁 —— 直接复用改进版 webui.py 的函数，不重复造轮子。
放在 CosyVoice 仓库根目录(与 webui.py 同级)运行：
    python cosyvoice_api.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B --port 8007

音色模型(可扩展)：音色 = webui 音色库里的 <name>.pt/.wav/.json；.json 里除 webui 自带字段外，
额外记录 source(clone|design|…) / description / tags / base_voice / instruct / seed / sample_text。
新增音色来源只需在保存时写不同的 source 与元信息，无需改存储。"""
import argparse
import io
import json
import logging
import os
import random
import sys
import tempfile
from typing import Literal

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, RedirectResponse, Response  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import polyphone  # noqa: E402
import webui as W  # noqa: E402

router = APIRouter(prefix="/api")
EXTRA_META = ("source", "description", "tags", "base_voice", "instruct", "seed", "sample_text")


def _msg(e: Exception) -> str:
    return getattr(e, "message", None) or str(e)


def _write_meta(name: str, **kw) -> dict:
    m = W.read_meta(name)
    m.update({k: v for k, v in kw.items() if v is not None and k in EXTRA_META})
    with open(W._voice_paths(name)[2], "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    return m


def _parse_tags(s) -> list[str] | None:
    if s is None:
        return None
    if isinstance(s, list):
        return [str(x).strip() for x in s if str(x).strip()]
    return [x.strip() for x in str(s).replace("，", ",").split(",") if x.strip()]


def voice_info(name: str) -> dict:
    in_lib = name in W.LIB_IDS
    m = W.read_meta(name) if in_lib else {}
    return {
        "name": name, "source": m.get("source", "library" if in_lib else "model"),
        "in_library": in_lib, "deletable": in_lib, "created": m.get("created", ""),
        "duration": m.get("duration"), "prompt_text": (m.get("prompt_text") or "").replace(W.V3_PREFIX, ""),
        "description": m.get("description", ""), "tags": m.get("tags", []),
        "base_voice": m.get("base_voice"), "instruct": m.get("instruct"), "seed": m.get("seed"),
        "has_audio": W.voice_wav_path(name) is not None,
    }


# ---------------- 健康 / 音色库 ----------------
@router.get("/health")
def health():
    return {"ok": True, "model": type(W.cosyvoice).__name__, "kind": W.MODEL_KIND,
            "sample_rate": W.SAMPLE_RATE, "voices": len(W.saved_voices()), "library": W.LIB_DIR}


@router.get("/voices")
def list_voices():
    W.load_library()  # 顺手载入手动放进音色库目录的新文件
    return [voice_info(n) for n in W.saved_voices()]


@router.get("/voices/{name}/audio")
def voice_audio(name: str):
    p = W.voice_wav_path(name) if name in W.saved_voices() else None
    if not p:
        raise HTTPException(404, f"音色 {name} 不存在或没有保存 prompt 音频")
    return FileResponse(p, media_type="audio/wav")


@router.post("/voices")
def clone_voice(name: str = Form(...), prompt_text: str = Form(...), file: UploadFile = File(...),
                overwrite: bool = Form(False), description: str = Form(""), tags: str = Form(""),
                meta: str = Form("")):
    """复刻：上传 3~30 秒 prompt 音频 + 对应文字 -> 保存为命名音色。meta 为可选 JSON(source/base_voice/instruct/...)。"""
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as t:
        t.write(file.file.read())
    try:
        W.do_save_voice(name, t.name, None, prompt_text, overwrite, True)
    except gr.Error as e:
        raise HTTPException(400, _msg(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"保存音色失败: {e}")
    finally:
        os.unlink(t.name)
    extra = {}
    if meta:
        try:
            extra = json.loads(meta)
        except ValueError:
            raise HTTPException(400, "meta 不是合法 JSON")
    extra.setdefault("source", "clone")
    extra["description"] = description or extra.get("description", "")
    extra["tags"] = _parse_tags(tags) or _parse_tags(extra.get("tags")) or []
    _write_meta(name, **extra)
    return voice_info(name)


class PatchReq(BaseModel):
    description: str | None = None
    tags: list[str] | str | None = None


@router.patch("/voices/{name}")
def patch_voice(name: str, req: PatchReq):
    if name not in W.LIB_IDS:
        raise HTTPException(404, "只能修改音色库里的音色")
    _write_meta(name, description=req.description, tags=_parse_tags(req.tags))
    return voice_info(name)


@router.delete("/voices/{name}")
def delete_voice(name: str):
    try:
        W.do_delete_voice(name)
    except gr.Error as e:
        raise HTTPException(400, _msg(e))
    return {"deleted": name}


# ---------------- 合成 ----------------
class TTSReq(BaseModel):
    text: str
    voice: str | None = None
    mode: Literal["zero_shot", "cross_lingual", "instruct"] = "zero_shot"
    instruct: str | None = None      # mode=instruct 时必填，如 "请用温柔舒缓的语气，慢速说话"
    speed: float = 1.0
    seed: int | None = None
    text_frontend: bool = True
    save: bool = False               # True: 同时写入 webui 的 outputs/ 历史记录


@router.post("/tts")
def tts(r: TTSReq):
    """返回 audio/wav。zero_shot=复刻音色朗读；instruct=在该音色基础上按自然语言指令改变语气/语速/方言；
    cross_lingual=跨语种。"""
    text = r.text.strip()
    if not text:
        raise HTTPException(400, "text 为空")
    voice = r.voice or ""
    known = set(W.saved_voices()) | set(W.sft_speakers())
    if voice and voice not in known:
        raise HTTPException(404, f"音色 {voice} 不存在。可用: {sorted(known)}")
    sft = voice if voice in W.sft_speakers() else None
    mode = {"zero_shot": W.MODE_ZS, "cross_lingual": W.MODE_CL, "instruct": W.MODE_INS}[r.mode]
    if sft and W.MODEL_KIND == "v1" and r.mode != "instruct":
        mode = W.MODE_SFT
    if not (voice or sft):
        raise HTTPException(400, "需要指定 voice（先在音色库里复刻/生成一个音色）")
    if r.mode == "instruct" and not (r.instruct or "").strip():
        raise HTTPException(400, "mode=instruct 需要 instruct")
    seed = r.seed if r.seed is not None else random.randint(1, 100000000)
    try:
        with W.INFER_LOCK:
            W.set_all_random_seed(seed)
            it, label = W.build_iterator(mode, text, sft, "" if sft else voice, None, "", r.instruct or "",
                                         False, r.speed, r.text_frontend, True)
            chunks = [o["tts_speech"].detach().cpu().float().numpy().flatten() for o in it]
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"合成失败: {e}")
    if not chunks:
        raise HTTPException(500, "没有生成任何音频")
    full = np.clip(np.concatenate(chunks), -1.0, 1.0)
    if r.save:
        W.save_output(full, f"API-{r.mode}", label, text, seed)
    buf = io.BytesIO()
    sf.write(buf, full, W.SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav",
                    headers={"X-Seed": str(seed), "X-Duration": f"{len(full) / W.SAMPLE_RATE:.2f}",
                             "X-Sample-Rate": str(W.SAMPLE_RATE)})


# ---------------- 多音字 ----------------
class PinyinReq(BaseModel):
    text: str
    use_dict: bool = True


@router.post("/pinyin")
def check_pinyin(r: PinyinReq):
    """多音字检查（不合成）：返回套用词典后的文本、送入模型的文本，以及每个多音字的读音和来源
    (inline=手动标注 / dict=词典 / guess=pypinyin 推测，可能有误)。"""
    res = polyphone.analyze(r.text, use_dict=r.use_dict)
    try:
        res["model_text"] = W.fix_pronunciation(res["text"])
    except ValueError as e:
        raise HTTPException(400, str(e))
    res["effective"] = W.MODEL_KIND == "v3"
    return res


# ---------------- 启动 ----------------
def create_app(args) -> FastAPI:
    W.init_runtime(args)
    app = FastAPI(title="CosyVoice API")
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def _root():
        return RedirectResponse("/ui/")

    return gr.mount_gradio_app(app, W.build_ui(), path="/ui")


class _QuietHealth(logging.Filter):
    """访问日志里不记成功的健康检查：工坊后端每隔几秒探测一次 /api/health，会刷满日志。失败的照常记录。"""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args if isinstance(record.args, tuple) else ()
        # uvicorn 访问日志的参数：(客户端, 方法, 路径, HTTP 版本, 状态码)
        if len(args) >= 5 and str(args[2]).startswith("/api/health"):
            try:
                return int(args[4]) >= 400
            except (TypeError, ValueError):
                return True
        return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8007)
    ap.add_argument("--model_dir", default="pretrained_models/Fun-CosyVoice3-0.5B")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--llm", choices=["base", "rl"], default="base", help="CosyVoice3 LLM 权重：base=llm.pt, rl=llm.rl.pt")
    ap.add_argument("--library_dir", default=os.path.join(ROOT, "spk_library"))
    ap.add_argument("--output_dir", default=os.path.join(ROOT, "outputs"))
    a = ap.parse_args()
    logging.getLogger("uvicorn.access").addFilter(_QuietHealth())
    uvicorn.run(create_app(a), host=a.host, port=a.port)
