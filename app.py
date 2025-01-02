# Gradio app
# A chatbot that supports Audio inputs(user can upload an audio file.)

# from transformers import AutoModel, AutoTokenizer

import gradio as gr
from transformers import AutoModel

message_box = None

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
    if speech is None:
        gr.Warning("⚠️ Please upload an audio file first!", duration=5)
        return (history, speech, text_box, chatbot, chat_button, upload_button)
    history.append({"role": "audio", "content": speech})
    chatbot.append([f"Speech: \n\n{speech}", None])

    return (
        history,
        gr.update(interactive=False), # speech box
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
            speech = gr.Audio(label="Audio", type="filepath")
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
            examples=[
                "Transcribe the speech.",
                "What is the emotion of the speaker?",
                "What does the audio sound like?",
                "將這段語音轉成文字。",
                "你聽到了什麼？"
            ],
            inputs=[text_box],
            label="Example prompts"
        )
    with gr.Row():
        examples = gr.Examples(
            examples=[
                ["assets/audios/0_000307.wav"],
                ["assets/audios/4_0_d47.wav"],
                ["assets/audios/7_1_d7.wav"],
                ["assets/audios/AccentClassification_AccentdbExtended_0193_british_s01_176.wav"],
                ["assets/audios/common_voice_en_34980360.mp3"],
                ["assets/audios/DialogueEmotionClassification_DailyTalk_0196_7_1_d756.wav"],
                ["assets/audios/Ses01M_script01_1_F014.wav"],
            ],
            inputs=[speech],
            label="Example audios"
        )

if __name__ == "__main__":
    demo.launch(share=True)