# Gradio app
# A chatbot that supports Audio inputs(user can upload an audio file.)

# from transformers import AutoModel, AutoTokenizer

import gradio as gr
from transformers import AutoModel



if gr.NO_RELOAD:
    model = AutoModel.from_pretrained("DeSTA-ntu/DeSTA2-8B-beta", trust_remote_code=True)
    model.to("cuda")
    model.eval()

def reset_chat(history, chatbot):
    history = [{"role": "system", "content": "Focus on the input audio. You are a helpful voice assistant."}]
    # history.clear()
    return (history, None, gr.update(interactive=False), gr.update(interactive=True))

def upload_audio(history, speech, text_box, chatbot, chat_button, upload_button):
    # {"role": "audio", "content": "assets/audios/DialogueEmotionClassification_DailyTalk_0196_7_1_d756.wav"},
    print(speech)
    if speech is None:
        gr.Warning("⚠️ Please upload an audio file first!", duration=5)
        return (history, speech, text_box, chatbot, chat_button, upload_button)
    history.append({"role": "audio", "content": speech})
    chatbot.append([f"Speech: \n\n{speech}", None])

    return (
        history,
        gr.update(interactive=True), # speech box
        gr.update(interactive=True, placeholder="Start chatting!"), # text_box,
        chatbot,
        gr.update(interactive=True), # chat_button,
        gr.update(interactive=False) # upload_button
    )

def user_send_message(history, speech, text_box, chatbot):
    history.append({"role": "user", "content": text_box})
    chatbot.append([f"{text_box}", None])

    return (
        history,
        speech,
        gr.update(interactive=True, placeholder="Start chatting!", value=""), # text_box,
        chatbot,
    )

def model_response(history, speech, text_box, chatbot):
    
    print(history)

    messages = history
    generated_ids = model.chat(messages, max_new_tokens=128, do_sample=False, temperature=1.0, top_p=1.0)
    response = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    history.append({"role": "assistant", "content": response})
    chatbot[-1][1] = response
    return (
        history,
        speech,
        gr.update(interactive=True, placeholder="Start chatting!"), # text_box,
        chatbot,
    )


with gr.Blocks() as demo:
    gr.Markdown("# DeSTA2 demo page")
    message_box = gr.Markdown(value="have fun!", label="Message")


    history = gr.State([{ "role": "system", "content": "Focus on the input audio. You are a helpful voice assistant." }])
    # history = gr.State([])
    with gr.Row():
        chatbot = gr.Chatbot(label="DeSTA2", height="100%", min_height="400px")
    
    with gr.Row():
        with gr.Column():
            speech = gr.Audio(label="Audio", type="filepath", sources=["microphone", "upload"])
            upload_button = gr.Button("Upload")
        with gr.Column():
            text_box = gr.Textbox(label="User", interactive=False, placeholder="Upload an audio first!")
            chat_button = gr.Button("Send", interactive=False)

    with gr.Row():
        # top_p = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, label="Top P")
        # temperature = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, label="Temperature")
        gr.Button("Reset chat").click(reset_chat, 
                                      inputs=[history, chatbot], 
                                      outputs=[history, chatbot, chat_button, upload_button])

    upload_button.click(upload_audio, 
                        inputs=[history, speech, text_box, chatbot, chat_button, upload_button], 
                        outputs=[history, speech, text_box, chatbot, chat_button, upload_button]
                        )
    chat_button.click(user_send_message, 
                      inputs=[history, speech, text_box, chatbot], 
                      outputs=[history, speech, text_box, chatbot]).then(
                            model_response, 
                            inputs=[history, speech, text_box, chatbot], 
                            outputs=[history, speech, text_box, chatbot]
                      )
    
    with gr.Row():
        examples_prompt = gr.Examples(
            examples = [
                "Transcribe the speech accurately.",
                "What is the primary emotion conveyed by the speaker?",
                "Describe the content and tone of the audio in detail.",
                "Provide a summary of the audio content.",
                "Identify the language spoken in the recording.",
                "What does the background noise in the audio indicate?",
                "Identify if the speaker has a specific accent and describe it.",
                "What is the gender and approximate age of the speaker?",
                "Summarize the conversation happening in this audio.",
                "Classify the type of audio: speech, music, noise, or mixed.",
                "Assess the clarity and intelligibility of the speech.",
                "What is the emotional state of the speaker, and why do you think so?",
                "Provide a timestamped breakdown of key events in the audio."
                "將這段語音轉成文字，請確保準確的時間點。",
                "你能辨認出這段語音的情感是什麼嗎？",
                "這段聲音中的說話者有什麼情緒？",
                "從這段聲音中提取關鍵詞。",
                "請翻譯這段語音的內容。",
                "從這段聲音中找出說話者的性別和口音。",
            ],
            inputs=[text_box],
            label="Example prompts"
        )
    with gr.Row():
        examples = gr.Examples(
            examples = [
                ["assets/audios/0_000307.wav"],
                ["assets/audios/4_0_d47.wav"],
                ["assets/audios/7_1_d7.wav"],
                ["assets/audios/AccentClassification_AccentdbExtended_0193_british_s01_176.wav"],
                ["assets/audios/DialogueEmotionClassification_DailyTalk_0196_7_1_d756.wav"],
                ["assets/audios/EmotionRecognition_MultimodalEmotionlinesDataset_0026_dia382_utt0.wav"],
                ["assets/audios/LanguageIdentification_VoxForge_0000_de143-43.flac"],
                ["assets/audios/MUL0608_120.98_148.92.wav"],
                ["assets/audios/NoiseDetection_LJSpeech_MUSAN-Music_0199_music_LJSpeech-1.1_16k_LJ050-0033.wav"],
                ["assets/audios/Ses01F_script03_1_F029.wav"],
                ["assets/audios/Ses01M_script01_1_F014.wav"],
                ["assets/audios/Ses04F_impro02_M004.wav"],
                ["assets/audios/SpeakerVerification_LibriSpeech-TestClean_0046_3575-170457-0038.flac"],
                ["assets/audios/SpeechTextMatching_LJSpeech_0001_LJ001-0107.wav"],
                ["assets/audios/common_voice_en_34980360.mp3"],
                ["assets/audios/p284_159.wav"],
                ["assets/audios/p287_162.wav"]
            ],
            inputs=[speech],
            label="Example audios"
        )

if __name__ == "__main__":
    demo.launch(share=True)