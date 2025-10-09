from huggingface_hub import hf_hub_download
import gradio as gr
from image_inference import RTDETR_ONNX
import os # Import os for logging
import inspect # Import inspect for type checking

# --- Setup ---
model_path = hf_hub_download(
    repo_id="hasnatz/v-safe-rf-detr",
    filename="inference_model.onnx"
)

model = RTDETR_ONNX(model_path)

# Ready-made example images (local files or URLs)
examples = [
    ["examples/121113-F-LV838-027.jpg"],
    ["examples/goggles_bing_construction_goggles_000109.jpg"],
    ["examples/image-shows-busy-construction-site-where-concrete-mixer-truck-works-alongside-laborers-safety-gear-focus-teamwork-347908285 (Small).jpeg"],
    ["examples/istockphoto-1324894706-612x612.jpg"],
    ["examples/shutterstock_174689291.jpg"],
    ["examples/worker_bing_construction_building_worker_000043 (8).jpg"],
    ["examples/worker_bing_construction_building_worker_000066 (1).jpg"],
    ["examples/worker_bing_construction_building_worker_000091 (2).jpg"]
]


# Building Gradio UI
custom_theme = gr.themes.Base().set(
    body_background_fill="#0f0f11",
    block_background_fill="#0f0f11",
    block_border_color="#0f0f11",
    background_fill_primary="#0f0f11"
)


with gr.Blocks(theme=custom_theme) as demo:

    gr.HTML(
    """
    <div style="text-align: center;">
    <img 
        src='/gradio_api/file=Logo.png' 
        alt='My Image' 
        style='height: 100px; width: auto; display: block; margin: 0 auto;'
    >
    <h2>Vision-Based Construction Safety Detection</h2>
    <br>
</div>
    """
    )

    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="pil", label="Upload an Image", height=450)
                confidence_img = gr.Slider(0.0, 1.0, value=0.25, step=0.05, label="Confidence Threshold")
                run_img_btn = gr.Button("Run Inference")

            with gr.Column():
                output_img = gr.Image(type="pil", label="Annotated Result", height=450)

        def predict_image(image, conf):
            annotated = model.run_inference(image, confidence_threshold=conf)
            return annotated

        run_img_btn.click(
            fn=predict_image,
            inputs=[input_img, confidence_img],
            outputs=output_img
        )

        gr.Examples(
        examples=examples,              
        inputs=[input_img],             
        outputs=[output_img],           
        fn=predict_image,               
        cache_examples=False)

    with gr.Tab("Video"):
        with gr.Row():
            with gr.Column():
                input_video = gr.Video(label="Upload a Video (.mp4)")
                confidence_vid = gr.Slider(0.0, 1.0, value=0.25, step=0.05, label="Confidence Threshold")
                run_vid_btn = gr.Button("Run Video Inference")

            with gr.Column():
                output_video = gr.Video(label="Annotated Video Result")

        def predict_video(video, conf):
            """
            Processes the uploaded video, annotates frames, writes to a temporary MP4 file,
            and returns the file path for Gradio Video playback.
            """
            
            # --- START FIX (Simplified extraction logic) ---
            video_path_str = video
            
            if not isinstance(video, str) and hasattr(video, 'name'):
                video_path_str = video.name
            # --- END FIX ---
        
        
            video_path = model.process_video_to_file(
                video_path_str,  # Pass the guaranteed string path
                max_duration=5,
                target_fps=5,
                max_height=640,
                confidence_threshold=conf
            )
            return video_path
        
        # --- Change 2: Update the click listener inputs ---
        run_vid_btn.click(
            fn=predict_video,
            # CHANGED: input_video corresponds to the 'video' argument now
            inputs=[input_video, confidence_vid], 
            outputs=output_video
        )


# Launch the app
if __name__ == "__main__":
    # Ensure all required folders are allowed
    demo.launch(allowed_paths=["Logo.png"], share=True)