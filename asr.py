import os

from dashscope.audio.asr import Recognition


def transcribe(wav_path: str) -> str:
    """调用阿里 ASR，返回识别文本；失败返回空字符串。"""
    asr_instance = Recognition(
        model="paraformer-realtime-v2",
        format="wav",
        sample_rate=16000,
        callback=None,
    )

    try:
        response = asr_instance.call(os.path.abspath(wav_path))

        if response.status_code != 200:
            print(f"❌ [错误] 阿里接口返回异常 -> 状态码: {response.status_code}")
            return ""

        if hasattr(response, "get_sentence"):
            sentences = response.get_sentence()
        else:
            sentences = response.output.get("sentences", [])

        if isinstance(sentences, list):
            return "".join(s.get("text", "") for s in sentences).strip()
        return str(sentences).strip()
    except Exception as e:
        print(f"❌ [错误] 语音识别请求失败: {e}")
        return ""
