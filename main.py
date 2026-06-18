# main.py
import asyncio
import edge_tts
import pygame
import speech_recognition as sr
from openai import AsyncOpenAI
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.templating import Jinja2Templates
import uvicorn
import os
from dotenv import load_dotenv
import tempfile
import speech_recognition as sr
import dashscope
from dashscope.audio.asr import Recognition
import pyaudio
import numpy as np
import time
from openwakeword.model import Model


# ================= 1. 配置区域 =================
load_dotenv()
# 填入你申请的阿里云百炼 API Key
dashscope.api_key = os.getenv("ALI_KEY");

# 初始化录音识别器
recognizer = sr.Recognizer()

def calibrate_noise():
    print("\n[系统提示] 🎤 正在校准环境基准噪音，请保持安静 1 秒钟...")
    try:
        with sr.Microphone(sample_rate=16000) as global_source:
            recognizer.adjust_for_ambient_noise(global_source, duration=1)
        print("[系统提示] ✅ 噪声校准完成！设备已准备就绪。")
    except Exception as e:
        print(f"[错误] 麦克风初始化失败: {e}")

def wait_for_wake_word():
    print("\n[系统状态] 正在加载唤醒模型，请稍候...")
    
    # 动态获取当前 main.py 所在的文件夹路径，并拼上模型文件名
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "hey_jarvis.onnx")
    
    # 直接加载我们打包好的本地模型文件！
    oww_model = Model(wakeword_models=[model_path], inference_framework="onnx")
    
    print("[系统状态] 模型加载完成，准备接管麦克风...")
    time.sleep(1) 
    
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1280
    
    pa = pyaudio.PyAudio()
    
    try:
        mic_stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
    except Exception as e:
        print(f"❌ [错误] 麦克风被占用，无法打开: {e}")
        return False

    print(f"\n💤 [休眠中] 等待唤醒词 (请喊 'Hey Jarvis')...")
    
    try:
        while True:
            pcm = mic_stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(pcm, dtype=np.int16)
            
            prediction = oww_model.predict(audio_data)
            
            # 因为指定了具体路径，prediction 的 key 就是绝对路径的名字
            for mdl_name, score in prediction.items():
                if score > 0.2:
                    print(f"\n🔔 [唤醒] 检测到唤醒词！(置信度: {score:.2f})")
                    return True
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        pa.terminate()
        
def record_audio():
    with sr.Microphone(sample_rate=16000) as source:
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            return None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(audio.get_wav_data())
            return tf.name

def call_ali_asr(wav_file_path):
    asr_instance = Recognition(
        model='paraformer-realtime-v2',
        format='wav',
        sample_rate=16000,
        callback=None
    )
    return asr_instance.call(os.path.abspath(wav_file_path))

# 初始化音频播放模块
pygame.mixer.init()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 写死系统提示词
SYSTEM_PROMPT = "你是一个贴心的家庭智能音响助手，请用简短、口语化、温柔的中文回答用户的问题，每次回答不超过50个字。"

# 全局状态：当前选中的模型
app_state = {
    "model": "deepseek-chat" # 默认使用 DeepSeek
}

# ================= 2. WebSocket 管理 =================
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # 连接时发送当前模型状态给前端
        await websocket.send_text(json.dumps({"type": "config_update", "model": app_state["model"]}))

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

# ================= 3. 路由与接口 =================
@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # 监听前端发来的消息（例如切换模型）
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "change_model":
                app_state["model"] = message.get("model")
                print(f"前端已将模型切换为: {app_state['model']}")
                # 这里不需要广播，因为是前端主动切换的，只需后端记住即可
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ================= 4. 核心语音循环 =================
async def smart_speaker_loop():
    await asyncio.sleep(2)
    
    # 启动时自动预生成“我在”的本地音频，保证唤醒零延迟
    current_dir = os.path.dirname(os.path.abspath(__file__))
    wake_audio_path = os.path.join(current_dir, "wozai.mp3")
    if not os.path.exists(wake_audio_path):
        print("\n[系统状态] 正在预生成唤醒提示音...")
        tts = edge_tts.Communicate("我在", "zh-CN-XiaoxiaoNeural")
        await tts.save(wake_audio_path)

    # 提前校准环境基准噪音
    calibrate_noise()

    while True:
        # --- 休眠等待唤醒阶段 ---
        await manager.broadcast(json.dumps({"type": "status", "text": "💤 休眠中 (喊Hey Jarvis 唤醒)"}))
        # 阻塞在此，直到听到唤醒词才会往下走
        await asyncio.to_thread(wait_for_wake_word) 
    
        # --- 唤醒成功，播放“我在” ---
        await manager.broadcast(json.dumps({"type": "status", "text": "✨ 我在！请说话..."}))
        pygame.mixer.music.load(wake_audio_path)
        pygame.mixer.music.play()
        # 等待提示音播放完再开始录音，防止麦克风把自己说的“我在”录进去
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)
        
        # 异步执行真实录音
        wav_file = await asyncio.to_thread(record_audio)
        
        if wav_file is None:
            continue
            
        await manager.broadcast(json.dumps({"type": "status", "text": "语音识别中..."}))
        
        # 调用阿里 ASR 将录音转文字
        user_text = ""
        try:
            response = await asyncio.to_thread(call_ali_asr, wav_file)
            
            if response.status_code == 200:
                if hasattr(response, "get_sentence"):
                    sentences = response.get_sentence()
                else:
                    sentences = response.output.get("sentences", [])
                
                if isinstance(sentences, list):
                    user_text = "".join([s.get("text", "") for s in sentences]).strip()
                else:
                    user_text = str(sentences)
            else:
                print(f"❌ [错误] 阿里接口返回异常 -> 状态码: {response.status_code}")
                
        except Exception as e:
            print(f"❌ [错误] 语音识别请求失败: {e}")
        finally:
            # 清理临时音频文件
            if wav_file and os.path.exists(wav_file):
                os.remove(wav_file)

        # 如果没有提取到文字，就重新回到开头继续等声音
        if not user_text:
            continue
            
        # 将识别到的文字推送到网页前端
        await manager.broadcast(json.dumps({"type": "user_msg", "text": user_text}))
        
        # --- 调用国内大模型 (LLM) ---
        # 【重点】这里展示如何根据前端选择的模型，请求不同的国内API
        current_model = app_state["model"]
        await asyncio.sleep(1) # 模拟网络请求耗时
        if "deepseek" in current_model:
            client = AsyncOpenAI(
                api_key=os.getenv("DEEPSEEK_KEY"), 
                base_url="https://api.deepseek.com"
            )
            
            try:
                # 尝试请求 DeepSeek 接口
                response = await client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_text}
                    ],
                    max_tokens=100
                )
                ai_reply = response.choices[0].message.content
            except Exception as e:
                # 如果发生错误（如余额不足、断网），将错误打印到控制台，并给前端发送友好提示
                print(f"❌ 大模型调用失败: {e}")
                ai_reply = "抱歉主人，大模型接口调用失败了，请检查网络或API余额。"

        elif "qwen" in current_model:
            # TODO: 使用 dashscope 库 或 openai库 + 阿里百炼 API Key
            ai_reply = f"（来自通义千问）很高兴为您服务，今天阳光明媚。"
        elif "glm" in current_model:
            # TODO: 使用 zhipuai 库 + 智谱 API Key
            ai_reply = f"（来自智谱GLM）今天天气不错，适合出门哦。"
        else:
            ai_reply = "我不知道我在用什么模型..."

        # --- 文字转语音 (TTS) 与播放 ---
        await manager.broadcast(json.dumps({"type": "status", "text": "正在说话..."}))
        await manager.broadcast(json.dumps({"type": "ai_msg", "text": ai_reply}))
        
        try:
            # 1. 配置 edge-tts (zh-CN-XiaoxiaoNeural 是微软很经典的温柔女声)
            voice = "zh-CN-XiaoxiaoNeural"
            tts = edge_tts.Communicate(ai_reply, voice)
            audio_file = "reply.mp3"
            
            # 2. 异步保存为 mp3 文件
            await tts.save(audio_file)
            
            # 3. 使用 pygame 播放音频
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # 4. 异步等待音频播放完毕，期间不会卡死网页和 WebSocket
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
                
        except Exception as e:
            print(f"❌ 语音合成或播放失败: {e}")
            await asyncio.sleep(2) # 失败了就稍微等2秒，避免循环跑得太快

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(smart_speaker_loop())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
