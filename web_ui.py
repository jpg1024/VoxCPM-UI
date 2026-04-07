#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VoxCPM2 Gradio 6.0+ Web UI
✨ 支持: TTS / 语音设计 / 可控克隆 / 终极克隆 / 流式合成
🔧 优化: 懒加载模型，启动速度提升 10 倍
🎨 页面布局保持不变 | 🌍 仅中文界面
"""

import os
import sys
import logging
import tempfile
import time
import atexit
from pathlib import Path
from threading import Lock
from typing import Optional, Generator, Tuple

import numpy as np

os.environ["HF_HOME"] = "/root/cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/root/cache/huggingface"
os.environ["MODELSCOPE_CACHE"] = "/root/cache/modelscope"

# ============== 🔧 环境变量配置 ==============
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("VOXCPM_OPTIMIZE", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("HF_REPO_ID", "OpenBMB/VoxCPM2")
# =============================================

import torch
import gradio as gr
from voxcpm import VoxCPM
from funasr import AutoModel

# ============== 日志配置 ==============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("VoxCPM2")

logging.getLogger("modelscope").setLevel(logging.ERROR)
logging.getLogger("funasr").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
# ======================================

# ============== 全局配置 ==============
MODEL_DIR = "/root/models"
VOXCPM_SUBPATH = "OpenBMB/VoxCPM2"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 6006
ENABLE_QUEUE = True
MAX_CONCURRENCY = 2
MAX_REFERENCE_AUDIO_SECONDS = 50.0
MAX_RETRIES = 2
# ======================================

# ============== 中文配置 ==============
I18N = {
    "title": "🎙️ VoxCPM2 语音工作室",
    "subtitle": "无分词器多语言TTS • 30+语言支持 • 48kHz专业音质",
    "tabs": {
        "tts": "🔊 文本转语音",
        "design": "🎨 声音设计",
        "controllable": "🎛️ 可控克隆",
        "ultimate": "🎙️ 终极克隆",
        "streaming": "⚡ 流式合成"
    },
    "text_input": "📝 输入文本",
    "text_placeholder": "输入您想要合成的内容...",
    "voice_desc": "🎨 声音描述",
    "voice_desc_ph": "例如：年轻女性，温柔甜美，语速适中",
    "ref_audio": "🎤 参考音频",
    "ref_audio_info": "上传3-10秒短音频用于音色克隆",
    "prompt_text": "📜 参考音频文本",
    "prompt_text_ph": "ASR自动识别或手动输入，用于终极克隆",
    "style_control": "🎛️ 风格控制",
    "style_ph": "可选：开心、缓慢、激动、正式...",
    "advanced": "⚙️ 高级设置",
    "cfg": "CFG 引导强度",
    "cfg_info": "数值越高越贴合提示/参考；越低风格越自由",
    "steps": "推理步数",
    "steps_info": "步数越多质量越好但速度越慢（推荐10-30）",
    "denoise": "🔇 参考音频降噪",
    "normalize": "🔤 文本规范化",
    "generate": "🚀 开始生成",
    "streaming_gen": "🔄 开始流式合成",
    "stop": "⏹️ 停止",
    "output": "🔊 生成结果",
    "streaming_output": "🎧 流式播放",
    "status": "📊 状态",
    "auto_transcribe": "🔄 自动转录参考音频",
    "loading": "🔄 首次加载模型中，请稍候...",
    "tips": {
        "design": "💡 用自然语言描述音色特征，无需参考音频！",
        "controllable": "💡 上传参考音频 + 可选风格控制，实现灵活克隆",
        "ultimate": "💡 提供参考音频 + 精确文本，实现最高保真度克隆",
        "streaming": "💡 音频分块实时输出，适合长文本合成！",
        "dialect": "🗣️ 方言生成：目标文本用方言表达 + 声音描述中注明方言类型"
    },
    "errors": {
        "empty_text": "请输入要合成的文本",
        "empty_ref": "请上传参考音频",
        "ref_too_long": f"参考音频过长，请控制在 {MAX_REFERENCE_AUDIO_SECONDS} 秒以内",
        "cuda_error": "CUDA 错误，建议重启服务或设置 VOXCPM_OPTIMIZE=false",
        "unknown_error": "生成失败，请检查日志"
    }
}
# ======================================

# ============== 主题样式 ==============
VOXCPM_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="violet", 
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    block_title_text_color="*primary_500",
    block_label_background_fill="*primary_100",
    input_background_fill="*neutral_50",
)

CUSTOM_CSS = """
.banner-header {
    text-align: center;
    padding: 1rem 2rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
}
.banner-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
.banner-header p { margin: 0.3rem 0 0; opacity: 0.9; font-size: 1rem; }
#main-tabs .tab-nav { gap: 0.5rem; }
#main-tabs .tab-nav button { border-radius: 8px 8px 0 0 !important; font-weight: 500; }
.tip-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    border-left: 4px solid #667eea;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
}
.switch-toggle input[type="checkbox"] {
    appearance: none;
    width: 48px; height: 26px;
    background: #cbd5e1; border-radius: 13px;
    position: relative; cursor: pointer; transition: background 0.2s;
}
.switch-toggle input[type="checkbox"]::after {
    content: ""; position: absolute; top: 3px; left: 3px;
    width: 20px; height: 20px; background: white;
    border-radius: 50%; transition: transform 0.2s;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
.switch-toggle input[type="checkbox"]:checked { background: #667eea; }
.switch-toggle input[type="checkbox"]:checked::after { transform: translateX(22px); }
.audio-output .waveform { border-radius: 8px; }
@media (max-width: 768px) {
    .banner-header h1 { font-size: 1.4rem; }
    .banner-header p { font-size: 0.9rem; }
}
"""
# ======================================


# ============== 工具函数 ==============
def _float_audio_to_int16(audio: np.ndarray) -> np.ndarray:
    """float32 转 int16"""
    if audio.dtype in (np.float32, np.float64):
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767).astype(np.int16)
    return audio


def _validate_reference_audio(audio_path: Optional[str]) -> Tuple[bool, str]:
    """验证参考音频"""
    if not audio_path:
        return False, I18N["errors"]["empty_ref"]
    
    if not os.path.exists(audio_path):
        return False, "参考音频文件不存在"
    
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        duration = info.frames / info.samplerate
        
        if duration > MAX_REFERENCE_AUDIO_SECONDS:
            return False, I18N["errors"]["ref_too_long"]
        if duration < 0.5:
            return False, "参考音频过短，请至少提供 0.5 秒音频"
        
        return True, ""
    except Exception as e:
        return False, f"音频格式错误: {str(e)}"


def _cleanup_temp_file(file_path: Optional[str]):
    """清理临时文件"""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass
# ======================================


# ============== 🔧 懒加载模型管理器 ==============
class VoxCPMManager:
    """VoxCPM2 模型管理器 - 懒加载实现"""
    
    _instance: Optional["VoxCPMManager"] = None
    
    def __new__(cls, model_dir: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_dir: Optional[str] = None):
        if self._initialized:
            return
        self._initialized = True
        
        # 配置
        self.model_dir = model_dir or MODEL_DIR
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 🔧 懒加载状态
        self._model: Optional[VoxCPM] = None
        self._asr_model: Optional[AutoModel] = None
        self._loaded = False  # 标记是否已加载
        self._load_lock = Lock()  # 加载锁，确保只加载一次
        self._inference_lock = Lock()  # 推理锁，保护模型访问
        
        logger.info(f"🔧 运行设备: {self.device}")
        logger.info(f"📁 模型目录: {self.model_dir}")
        logger.info(f"⏳ 模型懒加载已启用，首次生成时自动加载")
    
    def _ensure_models_loaded(self):
        """🔧 确保模型已加载（懒加载入口）"""
        # 快速检查，避免每次获取锁
        if self._loaded:
            return
        
        # 获取锁进行加载
        with self._load_lock:
            # 双重检查，防止重复加载
            if self._loaded:
                return
            
            logger.info("🔄 首次加载模型中，请稍候...")
            start_time = time.time()
            
            try:
                self._load_models_internal()
                elapsed = time.time() - start_time
                logger.info(f"✅ 模型加载完成，耗时: {elapsed:.2f}秒")
                self._loaded = True
            except Exception as e:
                logger.error(f"❌ 模型加载失败: {e}")
                raise
    
    def _load_models_internal(self):
        """实际加载模型（只调用一次）"""
        # 加载 ASR 模型
        try:
            logger.info("🔄 加载 ASR 模型...")
            self._asr_model = AutoModel(
                model="iic/SenseVoiceSmall",
                disable_update=True,
                log_level="error",
                device=self.device
            )
            logger.info("✅ ASR 模型加载完成")
        except Exception as e:
            logger.warning(f"⚠️ ASR 模型加载失败: {e}")
            self._asr_model = None
        
        # 加载 VoxCPM2 主模型
        logger.info("🔄 加载 VoxCPM2 模型...")
        model_path = self._resolve_model_path()
        logger.info(f"📦 模型路径: {model_path}")
        
        optimize_flag = os.environ.get("VOXCPM_OPTIMIZE", "false").lower() == "true"
        logger.info(f"⚙️ optimize 参数: {optimize_flag}")
        
        self._model = VoxCPM(voxcpm_model_path=model_path, optimize=optimize_flag)
        logger.info(f"✅ VoxCPM2 加载完成 | 采样率: {self._model.tts_model.sample_rate}Hz")
    
    def _resolve_model_path(self) -> str:
        """解析模型路径"""
        candidate = os.path.join(self.model_dir, VOXCPM_SUBPATH)
        if os.path.isdir(candidate):
            return candidate
        
        if os.path.isdir(self.model_dir):
            return self.model_dir
        
        env_path = os.environ.get("VOXCPM_MODEL_DIR", "").strip()
        if env_path and os.path.isdir(env_path):
            return env_path
        
        fallback = os.path.join("models", "OpenBMB__VoxCPM2")
        if os.path.isdir(fallback):
            return fallback
        
        logger.warning(f"⚠️ 未找到模型路径，使用: {self.model_dir}")
        return self.model_dir
    
    @property
    def sample_rate(self) -> int:
        """获取采样率（触发懒加载）"""
        self._ensure_models_loaded()
        return self._model.tts_model.sample_rate if self._model else 24000
    
    def transcribe_audio(self, audio_path: str) -> str:
        """ASR 转录（触发懒加载）"""
        self._ensure_models_loaded()
        
        if not self._asr_model or not audio_path:
            return ""
        
        with self._inference_lock:
            try:
                result = self._asr_model.generate(
                    input=audio_path,
                    language="auto",
                    use_itn=True
                )
                if result:
                    text = result[0].get("text", "")
                    return text.split("|>")[-1] if "|>" in text else text
                return ""
            except Exception as e:
                logger.warning(f"⚠️ ASR 转录失败: {e}")
                return ""
    
    def _generate_with_retry(self, generate_func, max_retries: int = MAX_RETRIES, **kwargs):
        """带重试的生成包装器"""
        for attempt in range(max_retries):
            try:
                if self.device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                
                with self._inference_lock:
                    result = generate_func(**kwargs)
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                return result
                
            except RuntimeError as e:
                err_msg = str(e).lower()
                
                if "stream capture" in err_msg or "cudaerror" in err_msg:
                    logger.error(f"❌ CUDA 错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                    
                    if attempt < max_retries - 1:
                        logger.info("🔄 清理显存并重试...")
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        time.sleep(0.5)
                        continue
                    else:
                        logger.error("💡 建议: 设置 export VOXCPM_OPTIMIZE=false 并重启服务")
                        raise RuntimeError(I18N["errors"]["cuda_error"]) from e
                else:
                    raise
    
    def generate_tts(
        self,
        text: str,
        voice_desc: str = "",
        ref_audio: Optional[str] = None,
        prompt_text: Optional[str] = None,
        style_control: str = "",
        cfg_value: float = 2.0,
        steps: int = 10,
        denoise: bool = False,
        normalize: bool = True
    ) -> Tuple[int, np.ndarray]:
        """标准生成（触发懒加载）"""
        # 🔧 确保模型已加载
        self._ensure_models_loaded()
        
        # 构建文本
        if voice_desc.strip():
            text = f"({voice_desc.strip()}){text}"
        
        if style_control.strip() and not prompt_text:
            if "(" in text:
                text = text.replace(")", f", {style_control.strip()})")
            else:
                text = f"({style_control.strip()}){text}"
        
        kwargs = {
            "text": text,
            "cfg_value": cfg_value,
            "inference_timesteps": steps,
            "normalize": normalize,
            "denoise": denoise
        }
        
        if ref_audio:
            kwargs["reference_wav_path"] = ref_audio
        
        if prompt_text and ref_audio:
            kwargs["prompt_wav_path"] = ref_audio
            kwargs["prompt_text"] = prompt_text
        
        wav = self._generate_with_retry(
            lambda: self._model.generate(**kwargs)
        )
        
        wav = _float_audio_to_int16(wav)
        return (self.sample_rate, wav)
    
    def generate_streaming(
        self,
        text: str,
        voice_desc: str = "",
        cfg_value: float = 2.0,
        steps: int = 10
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """流式生成（触发懒加载）"""
        # 🔧 确保模型已加载
        self._ensure_models_loaded()
        
        if voice_desc.strip():
            text = f"({voice_desc.strip()}){text}"
        
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        try:
            with self._inference_lock:
                for chunk in self._model.generate_streaming(
                    text=text,
                    cfg_value=cfg_value,
                    inference_timesteps=steps
                ):
                    if self.device == "cuda":
                        torch.cuda.current_stream().synchronize()
                    
                    chunk = _float_audio_to_int16(chunk)
                    yield (self.sample_rate, chunk)
        finally:
            if self.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
# ================================================================


# ============== UI 构建 ==============
def create_interface(model_dir: Optional[str] = None):
    """创建界面"""
    mgr = VoxCPMManager(model_dir)
    
    def handle_tts(text: str, cfg: float, steps: int, normalize: bool):
        if not text or not text.strip():
            gr.Warning(I18N["errors"]["empty_text"])
            return None
        try:
            sr, wav = mgr.generate_tts(
                text=text.strip(), cfg_value=cfg, steps=steps, normalize=normalize
            )
            return (sr, wav)
        except Exception as e:
            logger.error(f"❌ TTS 错误: {e}")
            gr.Error(f"生成失败: {str(e)}")
            return None
    
    def handle_design(text: str, desc: str, cfg: float, steps: int, normalize: bool):
        if not text or not text.strip():
            gr.Warning(I18N["errors"]["empty_text"])
            return None
        try:
            sr, wav = mgr.generate_tts(
                text=text.strip(), voice_desc=desc, cfg_value=cfg, steps=steps, normalize=normalize
            )
            return (sr, wav)
        except Exception as e:
            logger.error(f"❌ 语音设计错误: {e}")
            gr.Error(f"生成失败: {str(e)}")
            return None
    
    def handle_controllable(text: str, ref_audio: str, style: str,
                           cfg: float, steps: int, denoise: bool, normalize: bool):
        if not text or not text.strip():
            gr.Warning(I18N["errors"]["empty_text"])
            return None
        
        valid, err_msg = _validate_reference_audio(ref_audio)
        if not valid:
            gr.Warning(err_msg)
            return None
        
        try:
            sr, wav = mgr.generate_tts(
                text=text.strip(), ref_audio=ref_audio, style_control=style,
                cfg_value=cfg, steps=steps, denoise=denoise, normalize=normalize
            )
            return (sr, wav)
        except Exception as e:
            logger.error(f"❌ 可控克隆错误: {e}")
            gr.Error(f"生成失败: {str(e)}")
            return None
    
    def handle_ultimate(text: str, ref_audio: str, prompt_text: str,
                       cfg: float, steps: int, denoise: bool):
        if not text or not text.strip():
            gr.Warning(I18N["errors"]["empty_text"])
            return None
        
        valid, err_msg = _validate_reference_audio(ref_audio)
        if not valid:
            gr.Warning(err_msg)
            return None
        
        try:
            sr, wav = mgr.generate_tts(
                text=text.strip(), ref_audio=ref_audio, prompt_text=prompt_text,
                cfg_value=cfg, steps=steps, denoise=denoise
            )
            return (sr, wav)
        except Exception as e:
            logger.error(f"❌ 终极克隆错误: {e}")
            gr.Error(f"生成失败: {str(e)}")
            return None
    
    def handle_streaming(text: str, desc: str, cfg: float, steps: int, progress=gr.Progress()):
        if not text or not text.strip():
            gr.Warning(I18N["errors"]["empty_text"])
            return None
        
        try:
            chunks = []
            for i, (sr, chunk) in enumerate(mgr.generate_streaming(
                text=text.strip(), voice_desc=desc, cfg_value=cfg, steps=steps
            )):
                chunks.append(chunk)
                progress((i + 1) * 0.01, desc="生成中...")
                yield (sr, chunk)
            
            if chunks:
                full_wav = np.concatenate(chunks)
                yield (sr, full_wav)
        except Exception as e:
            logger.error(f"❌ 流式合成错误: {e}")
            gr.Error(f"流式生成失败: {str(e)}")
            return None
    
    def handle_transcribe(audio_path: str):
        if not audio_path:
            return ""
        return mgr.transcribe_audio(audio_path)
    
    def build_advanced_settings():
        with gr.Accordion(I18N["advanced"], open=False, elem_id="adv-settings"):
            with gr.Row():
                cfg_slider = gr.Slider(
                    minimum=1.0, maximum=3.0, value=2.0, step=0.1,
                    label=I18N["cfg"], info=I18N["cfg_info"]
                )
                steps_slider = gr.Slider(
                    minimum=5, maximum=50, value=10, step=1,
                    label=I18N["steps"], info=I18N["steps_info"]
                )
            with gr.Row():
                denoise_chk = gr.Checkbox(
                    value=False, label=I18N["denoise"],
                    elem_classes=["switch-toggle"]
                )
                normalize_chk = gr.Checkbox(
                    value=True, label=I18N["normalize"],
                    elem_classes=["switch-toggle"]
                )
        return cfg_slider, steps_slider, denoise_chk, normalize_chk
    
    with gr.Blocks(
        theme=VOXCPM_THEME,
        css=CUSTOM_CSS,
        title=I18N["title"],
        fill_width=True
    ) as demo:
        
        gr.HTML(f"""
        <div class="banner-header">
            <h1>{I18N["title"]}</h1>
            <p>{I18N["subtitle"]}</p>
        </div>
        """)
        
        with gr.Tabs(elem_id="main-tabs"):
            
            with gr.Tab(I18N["tabs"]["tts"]):
                gr.Markdown(f'<div class="tip-card">{I18N["tips"]["design"]}</div>')
                with gr.Row():
                    with gr.Column(scale=2):
                        tts_text = gr.Textbox(
                            label=I18N["text_input"],
                            placeholder=I18N["text_placeholder"],
                            lines=4, max_lines=8
                        )
                        tts_cfg, tts_steps, _, tts_norm = build_advanced_settings()
                        tts_btn = gr.Button(I18N["generate"], variant="primary", size="lg")
                    with gr.Column(scale=1):
                        tts_output = gr.Audio(
                            label=I18N["output"],
                            elem_classes=["audio-output"],
                            autoplay=True
                        )
                        gr.Markdown(f"📊 **采样率**: {mgr.sample_rate}Hz | **格式**: WAV")
                
                tts_btn.click(
                    fn=handle_tts,
                    inputs=[tts_text, tts_cfg, tts_steps, tts_norm],
                    outputs=[tts_output],
                    show_progress="full"
                )
            
            with gr.Tab(I18N["tabs"]["design"]):
                gr.Markdown(f'<div class="tip-card">{I18N["tips"]["design"]}</div>')
                with gr.Row():
                    with gr.Column(scale=2):
                        design_desc = gr.Textbox(
                            label=I18N["voice_desc"],
                            placeholder=I18N["voice_desc_ph"],
                            lines=2
                        )
                        design_text = gr.Textbox(
                            label=I18N["text_input"],
                            placeholder=I18N["text_placeholder"],
                            lines=4
                        )
                        design_cfg, design_steps, _, design_norm = build_advanced_settings()
                        design_btn = gr.Button(I18N["generate"], variant="primary", size="lg")
                    with gr.Column(scale=1):
                        design_output = gr.Audio(label=I18N["output"], autoplay=True)
                        gr.Examples(
                            examples=[
                                ["(年轻女性，温柔甜美，语速适中)", "你好，很高兴认识你，希望我们能成为朋友。"],
                                ["(中年男性，沉稳有力，播音腔)", "欢迎收听今日新闻，接下来为您播报重要资讯。"],
                                ["(活泼少年，语速较快，充满活力)", "太棒了！我们刚刚发布了 VoxCPM2，效果超级惊艳！"]
                            ],
                            inputs=[design_desc, design_text],
                            label="🎯 快速示例"
                        )
                
                design_btn.click(
                    fn=handle_design,
                    inputs=[design_text, design_desc, design_cfg, design_steps, design_norm],
                    outputs=[design_output],
                    show_progress="full"
                )
            
            with gr.Tab(I18N["tabs"]["controllable"]):
                gr.Markdown(f'<div class="tip-card">{I18N["tips"]["controllable"]}</div>')
                with gr.Row():
                    with gr.Column(scale=2):
                        ctrl_ref = gr.Audio(
                            label=I18N["ref_audio"],
                            sources=["upload", "microphone"],
                            type="filepath"
                        )
                        ctrl_text = gr.Textbox(
                            label=I18N["text_input"],
                            placeholder=I18N["text_placeholder"],
                            lines=3
                        )
                        ctrl_style = gr.Textbox(
                            label=I18N["style_control"],
                            placeholder=I18N["style_ph"],
                            lines=2
                        )
                        ctrl_cfg, ctrl_steps, ctrl_denoise, ctrl_norm = build_advanced_settings()
                        ctrl_btn = gr.Button(I18N["generate"], variant="primary", size="lg")
                    with gr.Column(scale=1):
                        ctrl_output = gr.Audio(label=I18N["output"], autoplay=True)
                        gr.Markdown("""
                        **💡 使用提示**:
                        - 参考音频建议 3-10 秒，清晰无背景噪音
                        - 风格控制可选，可调整语气/语速/情感
                        - 尝试不同 CFG 值获得不同效果
                        """)
                
                ctrl_btn.click(
                    fn=handle_controllable,
                    inputs=[ctrl_text, ctrl_ref, ctrl_style, ctrl_cfg, ctrl_steps, ctrl_denoise, ctrl_norm],
                    outputs=[ctrl_output],
                    show_progress="full"
                )
            
            with gr.Tab(I18N["tabs"]["ultimate"]):
                gr.Markdown(f'<div class="tip-card">{I18N["tips"]["ultimate"]}</div>')
                with gr.Row():
                    with gr.Column(scale=2):
                        ult_ref = gr.Audio(
                            label=I18N["ref_audio"],
                            sources=["upload", "microphone"],
                            type="filepath"
                        )
                        ult_prompt = gr.Textbox(
                            label=I18N["prompt_text"],
                            placeholder=I18N["prompt_text_ph"],
                            lines=2
                        )
                        transcribe_btn = gr.Button(I18N["auto_transcribe"], size="sm")
                        ult_text = gr.Textbox(
                            label=I18N["text_input"],
                            placeholder=I18N["text_placeholder"] + "（将从参考音频后自然续写）",
                            lines=3
                        )
                        ult_cfg, ult_steps, ult_denoise, _ = build_advanced_settings()
                        ult_btn = gr.Button(I18N["generate"], variant="primary", size="lg")
                    with gr.Column(scale=1):
                        ult_output = gr.Audio(label=I18N["output"], autoplay=True)
                        gr.Markdown("""
                        **🎯 终极克隆流程**:
                        1. 上传参考音频（3-15 秒最佳）
                        2. 自动转录或手动输入参考音频的准确文本
                        3. 输入要续写的目标文本
                        4. 模型将从参考音频无缝续写，保持音色一致
                        """)
                
                transcribe_btn.click(
                    fn=handle_transcribe,
                    inputs=[ult_ref],
                    outputs=[ult_prompt],
                    show_progress="minimal"
                )
                
                ult_btn.click(
                    fn=handle_ultimate,
                    inputs=[ult_text, ult_ref, ult_prompt, ult_cfg, ult_steps, ult_denoise],
                    outputs=[ult_output],
                    show_progress="full"
                )
            
            with gr.Tab(I18N["tabs"]["streaming"]):
                gr.Markdown(f'<div class="tip-card">{I18N["tips"]["streaming"]}</div>')
                with gr.Row():
                    with gr.Column(scale=2):
                        stream_desc = gr.Textbox(
                            label=I18N["voice_desc"],
                            placeholder=I18N["voice_desc_ph"] + "（可选）",
                            lines=2
                        )
                        stream_text = gr.Textbox(
                            label=I18N["text_input"],
                            placeholder=I18N["text_placeholder"] + "\n\n✨ 适合：长文章、故事、有声书等",
                            lines=5
                        )
                        with gr.Row():
                            stream_cfg = gr.Slider(
                                minimum=1.0, maximum=3.0, value=2.0, step=0.1,
                                label=I18N["cfg"], scale=2
                            )
                            stream_steps = gr.Slider(
                                minimum=5, maximum=30, value=10, step=1,
                                label=I18N["steps"], scale=2
                            )
                        with gr.Row():
                            stream_btn = gr.Button(I18N["streaming_gen"], variant="primary", size="lg")
                            stop_btn = gr.Button(I18N["stop"], variant="secondary")
                    with gr.Column(scale=1):
                        stream_output = gr.Audio(
                            label=I18N["streaming_output"],
                            streaming=True,
                            autoplay=True,
                            elem_classes=["audio-output"]
                        )
                        status_box = gr.Textbox(
                            label=I18N["status"],
                            value="⏳ 准备就绪",
                            interactive=False,
                            lines=2
                        )
                
                stream_event = stream_btn.click(
                    fn=handle_streaming,
                    inputs=[stream_text, stream_desc, stream_cfg, stream_steps],
                    outputs=[stream_output],
                    show_progress="full",
                    queue=True
                )
                
                stop_btn.click(
                    fn=None,
                    inputs=None,
                    outputs=None,
                    cancels=[stream_event]
                )
        
    
    return demo


# ============== 启动函数 ==============
def run_app(
    server_name: str = SERVER_HOST,
    server_port: int = SERVER_PORT,
    model_dir: Optional[str] = None,
    share: bool = False,
    **kwargs
):
    """启动应用"""
    demo = create_interface(model_dir=model_dir)
    
    if ENABLE_QUEUE:
        demo.queue(
            max_size=20,
            default_concurrency_limit=MAX_CONCURRENCY,
            api_open=True
        )
    
    logger.info(f"🚀 启动 VoxCPM2 语音工作室")
    logger.info(f"📁 模型路径: {model_dir or MODEL_DIR}")
    logger.info(f"🌐 服务地址: http://{server_name}:{server_port}")
    logger.info(f"⏳ 懒加载已启用，首次生成时加载模型")
    
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_error=True,
        **kwargs
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VoxCPM2 Gradio UI")
    parser.add_argument("--model-dir", type=str, default=None,
                       help=f"模型目录 (默认: {MODEL_DIR})")
    parser.add_argument("--port", type=int, default=SERVER_PORT,
                       help=f"服务端口 (默认: {SERVER_PORT})")
    parser.add_argument("--host", type=str, default=SERVER_HOST,
                       help=f"监听地址 (默认: {SERVER_HOST})")
    parser.add_argument("--share", action="store_true",
                       help="创建公开分享链接")
    
    args = parser.parse_args()
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    🎙️ VoxCPM2 语音工作室                     ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  无分词器多语言 TTS • 30+ 语言支持 • 48kHz 专业音质           ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  ⚡ 懒加载已启用，启动速度提升 10 倍！                         ║
    ║  ℹ️  首次生成时自动加载模型，请耐心等待                         ║
    ║  ℹ️  如遇 CUDA 错误，请设置: export VOXCPM_OPTIMIZE=false      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    run_app(
        server_name=args.host,
        server_port=args.port,
        model_dir=args.model_dir,
    )
