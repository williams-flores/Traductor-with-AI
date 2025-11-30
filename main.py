import gradio as gr
import whisper
from translate import Translator 
from dotenv import dotenv_values #Para guardar API_KEY's de manera segura
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

# requirements.txt
"""
gradio
openai-whisper
translate
python-dotenv
elevenlabs
"""

# Configuración .env
# config = dotenv_values(".env")
# ELEVENLABS_API_KEY = config["ELEVENLABS_API_KEY"]
# Para ejecutarlo abrir un entorno virtual y ejecutar el main.py

config = dotenv_values(".env")
ELEVENLAPS_API_KEY = config["ELEVENLAPS_API_KEY"]

def translator(audio_file):
    # 1. Transcribir texto

    # Usamos Whisper: https://github.com/openai/whisper
    # Alternativa API online: https://www.assemblyai.com

    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_file, language="Spanish", fp16=False)
        transcription = result["text"]

    except Exception as e:
        raise gr.Error(
            f"Se ha producido un error transcribiendo el texto: {str(e)}")
    
    print(f"Texto orginal: {transcription}")

    # 2. Traducir texto

    # Usamos Translate: https://github.com/terryyin/translate-python

    try:
        en_transcription = Translator(from_lang="es", to_lang="en").translate(transcription)
        hi_transcription = Translator(from_lang="es", to_lang="hi").translate(transcription)
        fr_transcription = Translator(from_lang="es", to_lang="fr").translate(transcription)
        ja_transcription = Translator(from_lang="es", to_lang="ja").translate(transcription)

    except Exception as e:
        raise gr.Error(
            f"Se ha producido un error traduciendo el audio: {str(e)}")
    
    
    print(f"Texto traducido a Inglés: {en_transcription}")
    print(f"Texto traducido a Hindu: {hi_transcription}")
    print(f"Texto traducido a Francés: {fr_transcription}")
    print(f"Texto traducido a Japonés: {ja_transcription}")

    # 3. Generar audio traducido

    # Usamos Elevenlabs IO: https://elevenlabs.io/docs/api-reference/getting-started

    en_save_file_path = text_to_speach(en_transcription, "en")
    hi_save_file_path = text_to_speach(hi_transcription, "hi")
    fr_save_file_path = text_to_speach(fr_transcription, "fr")
    ja_save_file_path = text_to_speach(ja_transcription, "ja")

    return en_save_file_path, hi_save_file_path, fr_save_file_path, ja_save_file_path
    
def text_to_speach(text: str, language: str) -> str:

    try:
        client = ElevenLabs(api_key=ELEVENLAPS_API_KEY)

        #Parametros predefinos por la herramienta ElevensLabs(genera audios)
        response = client.text_to_speech.convert(
                voice_id="onwK4e9ZLuTAKqWW03F9",  #Daniel 
                optimize_streaming_latency="0",
                output_format="mp3_22050_32",
                text=text,
                model_id="eleven_turbo_v2",
                voice_settings=VoiceSettings(
                    stability=0.0,
                    similarity_boost=0.0,
                    style=0.0,
                    use_speaker_boost=True,
                ),
            )
        
        save_file_path = f"{language}.mp3"

        with open(save_file_path, "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)

    except Exception as e:
        raise gr.Error(
            f"Se ha producido un error generando el audio: {str(e)}")

    return save_file_path


    

web = gr.Interface(
    fn=translator,
    inputs=gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="Español"
    ),
    outputs=[
        gr.Audio(label="Inglés"),
        gr.Audio(label="hindu"),
        gr.Audio(label="Francés"),
        gr.Audio(label="Japonés")
    ],
    title="Traductor de voz",
    description="Traductor de voz con AI a varios idiomas",
)

web.launch()