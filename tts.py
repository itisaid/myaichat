import edge_tts


async def synthesize(
    text: str,
    output_path: str,
    voice: str = "zh-CN-XiaoxiaoNeural",
) -> None:
    tts = edge_tts.Communicate(text, voice)
    await tts.save(output_path)
