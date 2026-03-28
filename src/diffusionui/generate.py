import logging
import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk

import huggingface_hub as hf_hub
import openvino_genai as ov_genai
from PIL import Image, ImageTk

MODEL_ID = "OpenVINO/stable-diffusion-v1-5-int8-ov"
DEVICE = "CPU"
COLLECTION_ID = "OpenVINO/image-generation"


def get_available_models() -> list[str]:
    """Fetch available models from the OpenVINO/image-generation collection."""
    try:
        collection = hf_hub.get_collection(COLLECTION_ID)
        models = [item.item_id for item in collection.items if hasattr(item, 'item_id')]
        # Strip OpenVINO/ prefix for display
        stripped_models = [m.replace("OpenVINO/", "") for m in models]
        return sorted(stripped_models) if stripped_models else [MODEL_ID.replace("OpenVINO/", "")]
    except hf_hub.HfHubHTTPError as e:
        logging.warning(f"Failed to fetch models from HuggingFace: {e}")
        return [MODEL_ID.replace("OpenVINO/", "")]


def get_full_model_id(
        model_name: str
        ) -> str:
    """Convert display model name to full model ID with OpenVINO/ prefix."""
    if model_name.startswith("OpenVINO/"):
        return model_name
    return f"OpenVINO/{model_name}"


class ImageGenerator:
    def __init__(
            self,
            model_id: str = MODEL_ID,
            device: str = DEVICE
            ) -> None:
        self.model_id = model_id
        self.device = device
        self.pipeline: ov_genai.Text2ImagePipeline | None = None

    def _ensure_pipeline(
            self,
            status_callback: Callable[[str], None]
            ) -> None:
        if self.pipeline is not None:
            return
        status_callback("Downloading model…")
        model_path = hf_hub.snapshot_download(self.model_id)
        status_callback("Loading pipeline…")
        self.pipeline = ov_genai.Text2ImagePipeline(model_path, self.device)
        status_callback("")

    def generate_image(
            self,
            prompt: str,
            status_callback: Callable[[str], None],
            **kwargs
    ) -> list[Image.Image]:
        self._ensure_pipeline(status_callback=status_callback)
        assert self.pipeline is not None
        status_callback("Generating image…")
        # Build generation parameters - use default steps if not provided
        gen_params = kwargs.copy()
        image_tensor = self.pipeline.generate(prompt, **gen_params)
        print("Generation done")
        status_callback("")

        # Convert tensor data to PIL images and return as list
        return [Image.fromarray(img) for img in image_tensor.data]


class DiffusionUI(tk.Tk):
    def __init__(
            self
            ) -> None:
        super().__init__()
        self.title("Diffusion UI")
        self.geometry("1000x1100")

        self.available_models = get_available_models()
        self.generator = ImageGenerator()
        self.preview_photo: ImageTk.PhotoImage | None = None

        self._build_layout()

    def _build_layout(
            self
            ) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        # Main controls frame
        controls = ttk.Frame(root)
        controls.pack(fill="x", pady=(0, 8))

        ttk.Label(controls, text="Prompt:").grid(row=0, column=0, sticky="nw", padx=(0, 8))

        self.prompt_var = tk.StringVar(value="sailing ship in storm by Rembrandt")
        self.prompt_text = tk.Text(controls, height=4, width=60, wrap="word")
        self.prompt_text.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self.prompt_text.insert("1.0", self.prompt_var.get())

        device_frame = ttk.Frame(controls)
        device_frame.grid(row=0, column=2, sticky="n", padx=(0, 8))

        ttk.Label(device_frame, text="Device:").pack(side="left", padx=(0, 8))
        self.device_var = tk.StringVar(value="CPU")
        self.device_combo = ttk.Combobox(device_frame, textvariable=self.device_var,
                                         values=["CPU", "GPU"], state="readonly", width=8)
        self.device_combo.pack(side="left")

        model_frame = ttk.Frame(controls)
        model_frame.grid(row=0, column=3, sticky="ne", padx=(0, 8))

        ttk.Label(model_frame, text="Model:").pack(side="left", padx=(0, 8))
        self.model_var = tk.StringVar(value=MODEL_ID.replace("OpenVINO/", ""))
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                        values=self.available_models, state="readonly", width=28)
        self.model_combo.pack(side="left")

        self.generate_button = ttk.Button(controls, text="Generate", command=self._start_generation)
        self.generate_button.grid(row=0, column=4, sticky="n", padx=(8, 0))

        controls.columnconfigure(1, weight=1)

        # Optional parameters frame
        params_frame = ttk.LabelFrame(root, text="Optional Parameters", padding=8)
        params_frame.pack(fill="x", pady=(10, 0))

        ttk.Label(params_frame, text="Steps:").grid(row=0, column=0, sticky="w")
        self.steps_var = tk.StringVar(value="")
        self.steps_spin = ttk.Spinbox(params_frame, from_=0, to=100, textvariable=self.steps_var,
                                      width=8)
        self.steps_spin.grid(row=0, column=1, sticky="w", padx=(8, 8))

        ttk.Label(params_frame, text="Negative Prompt:").grid(row=0, column=2, sticky="nw",
                                                              padx=(0, 8))
        self.negative_prompt_var = tk.StringVar(value="")
        self.negative_prompt_text = tk.Text(params_frame, height=2, width=40, wrap="word")
        self.negative_prompt_text.grid(row=0, column=3, sticky="ew", padx=(0, 8))

        ttk.Label(params_frame, text="Guidance Scale:").grid(row=0, column=4, sticky="e")
        self.guidance_scale_var = tk.StringVar(value="")
        self.guidance_scale_spin = ttk.Spinbox(params_frame, from_=0.0, to=20.0,
                                               textvariable=self.guidance_scale_var, width=8)
        self.guidance_scale_spin.grid(row=0, column=5, padx=(8, 8))

        ttk.Label(params_frame, text="Prompt 2:").grid(row=1, column=0, sticky="nw", padx=(0, 8))
        self.prompt_2_var = tk.StringVar(value="")
        self.prompt_2_text = tk.Text(params_frame, height=2, width=40, wrap="word")
        self.prompt_2_text.grid(row=1, column=1, sticky="ew", padx=(0, 8))

        ttk.Label(params_frame, text="Negative Prompt 2:").grid(row=1, column=2, sticky="nw",
                                                                padx=(0, 8))
        self.negative_prompt_2_var = tk.StringVar(value="")
        self.negative_prompt_2_text = tk.Text(params_frame, height=2, width=40, wrap="word")
        self.negative_prompt_2_text.grid(row=1, column=3, sticky="ew", padx=(0, 8))

        ttk.Label(params_frame, text="Height:").grid(row=1, column=4, sticky="e")
        self.height_var = tk.StringVar(value="")
        self.height_spin = ttk.Spinbox(params_frame, from_=0, to=2048, textvariable=self.height_var,
                                       width=8)
        self.height_spin.grid(row=1, column=5, sticky="w", padx=(8, 8))

        ttk.Label(params_frame, text="Prompt 3:").grid(row=2, column=0, sticky="nw", padx=(0, 8))
        self.prompt_3_var = tk.StringVar(value="")
        self.prompt_3_text = tk.Text(params_frame, height=2, width=40, wrap="word")
        self.prompt_3_text.grid(row=2, column=1, sticky="ew", padx=(0, 8))

        ttk.Label(params_frame, text="Negative Prompt 3:").grid(row=2, column=2, sticky="nw",
                                                                padx=(0, 8))
        self.negative_prompt_3_var = tk.StringVar(value="")
        self.negative_prompt_3_text = tk.Text(params_frame, height=2, width=40, wrap="word")
        self.negative_prompt_3_text.grid(row=2, column=3, sticky="ew", padx=(0, 8))

        ttk.Label(params_frame, text="Width:").grid(row=2, column=4, sticky="e")
        self.width_var = tk.StringVar(value="")
        self.width_spin = ttk.Spinbox(params_frame, from_=0, to=2048, textvariable=self.width_var,
                                      width=8)
        self.width_spin.grid(row=2, column=5, padx=(8, 8))

        ttk.Label(params_frame, text="Images per Prompt:").grid(row=3, column=0, sticky="w")
        self.num_images_var = tk.StringVar(value="")
        self.num_images_spin = ttk.Spinbox(params_frame, from_=0, to=10,
                                           textvariable=self.num_images_var, width=8)
        self.num_images_spin.grid(row=3, column=1, sticky="w", padx=(8, 8))

        ttk.Label(params_frame, text="Seed (0 = random):").grid(row=3, column=2, sticky="e")
        self.seed_var = tk.StringVar(value="")
        self.seed_spin = ttk.Spinbox(params_frame, from_=0, to=2147483647,
                                     textvariable=self.seed_var, width=12)
        self.seed_spin.grid(row=3, column=3, padx=(8, 8))

        ttk.Label(params_frame, text="Strength (0-1):").grid(row=3, column=4, sticky="e")
        self.strength_var = tk.StringVar(value="")
        self.strength_spin = ttk.Spinbox(params_frame, from_=0.0, to=1.0,
                                         textvariable=self.strength_var, width=8)
        self.strength_spin.grid(row=3, column=5, sticky="w", padx=(8, 8))

        ttk.Label(params_frame, text="Max Sequence Length:").grid(row=4, column=0, sticky="w")
        self.max_seq_length_var = tk.StringVar(value="")
        self.max_seq_length_spin = ttk.Spinbox(params_frame, from_=0, to=512,
                                               textvariable=self.max_seq_length_var, width=12)
        self.max_seq_length_spin.grid(row=4, column=1, padx=(8, 8))

        params_frame.columnconfigure(1, weight=1)
        params_frame.columnconfigure(3, weight=1)
        params_frame.columnconfigure(5, weight=1)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(root, textvariable=self.status_var).pack(anchor="w", pady=(10, 8))

        self.preview_frame = ttk.LabelFrame(root, text="Images", padding=8)
        self.preview_frame.pack(fill="both", expand=True)

        self.preview_photos: list[ImageTk.PhotoImage] = []

        self.prompt_text.focus_set()
        self.bind("<Control-Return>", lambda
            _event: self._start_generation())

    def _start_generation(
            self
            ) -> None:
        prompt = self.prompt_text.get("1.0", "end-1c").strip()
        if not prompt:
            messagebox.showwarning("Missing prompt", "Please enter a prompt.")
            return

        self.generate_button.config(state=tk.DISABLED)

        # Check if device or model has changed and recreate generator if needed
        selected_device = self.device_var.get()
        selected_model = get_full_model_id(self.model_var.get())
        if self.generator.device != selected_device or self.generator.model_id != selected_model:
            self.generator = ImageGenerator(model_id=selected_model, device=selected_device)

        # Collect optional parameters - only include if explicitly set by user
        kwargs = {}
        if self.steps_var.get().strip():
            kwargs["num_inference_steps"] = int(self.steps_var.get())

        negative_prompt = self.negative_prompt_text.get("1.0", "end-1c").strip()
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        
        if self.guidance_scale_var.get().strip():
            kwargs["guidance_scale"] = float(self.guidance_scale_var.get())

        prompt_2 = self.prompt_2_text.get("1.0", "end-1c").strip()
        if prompt_2:
            kwargs["prompt_2"] = prompt_2

        prompt_3 = self.prompt_3_text.get("1.0", "end-1c").strip()
        if prompt_3:
            kwargs["prompt_3"] = prompt_3

        negative_prompt_2 = self.negative_prompt_2_text.get("1.0", "end-1c").strip()
        if negative_prompt_2:
            kwargs["negative_prompt_2"] = negative_prompt_2

        negative_prompt_3 = self.negative_prompt_3_text.get("1.0", "end-1c").strip()
        if negative_prompt_3:
            kwargs["negative_prompt_3"] = negative_prompt_3
        
        if self.height_var.get().strip():
            kwargs["height"] = int(self.height_var.get())
        if self.width_var.get().strip():
            kwargs["width"] = int(self.width_var.get())
        if self.num_images_var.get().strip():
            kwargs["num_images_per_prompt"] = int(self.num_images_var.get())
        if self.seed_var.get().strip():
            kwargs["rng_seed"] = int(self.seed_var.get())
        if self.strength_var.get().strip():
            kwargs["strength"] = float(self.strength_var.get())
        if self.max_seq_length_var.get().strip():
            kwargs["max_sequence_length"] = int(self.max_seq_length_var.get())

        # Run model loading/inference off the UI thread to keep the window responsive.
        worker = threading.Thread(target=self._generate_in_background, args=(prompt,),
                                  kwargs=kwargs, daemon=True, )
        worker.start()

    def _generate_in_background(
            self,
            prompt: str,
            **kwargs
            ) -> None:
        try:
            images = self.generator.generate_image(prompt,
                                                   status_callback=self._set_status_from_worker,
                                                   **kwargs)
            self.after(0, self._on_generation_success, images)
        except Exception as error:  # noqa: BLE001 - surface any generation failure in the UI
            self.after(0, self._on_generation_error, error)

    def _set_status_from_worker(
            self,
            text: str
            ) -> None:
        self.after(0, self.status_var.set, text)

    def _on_generation_success(
            self,
            images: list[Image.Image]
            ) -> None:
        if not images:
            self.status_var.set("No images generated")
            self.generate_button.config(state=tk.NORMAL)
            return

        # Clear existing preview content
        for widget in self.preview_frame.winfo_children():
            widget.destroy()

        # Calculate grid layout (prefer square-ish grids)
        num_images = len(images)
        cols = int(num_images ** 0.5)
        rows = (num_images + cols - 1) // cols

        # Store PhotoImage references to prevent garbage collection
        self.preview_photos = []

        # Create a grid of image labels
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols

            # Convert PIL image to PhotoImage
            photo = ImageTk.PhotoImage(img)
            self.preview_photos.append(photo)

            # Create label with the image
            label = ttk.Label(self.preview_frame, image=photo)
            label.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")

        # Configure grid weights for even distribution
        for i in range(rows):
            self.preview_frame.grid_rowconfigure(i, weight=1)
        for i in range(cols):
            self.preview_frame.grid_columnconfigure(i, weight=1)

        self.status_var.set("Done")
        self.generate_button.config(state=tk.NORMAL)

    def _on_generation_error(
            self,
            error: Exception
            ) -> None:
        self.status_var.set("Generation failed")
        self.generate_button.config(state=tk.NORMAL)
        messagebox.showerror("Generation failed", str(error))
        logging.exception(error)


def main() -> None:
    app = DiffusionUI()
    app.mainloop()


if __name__ == "__main__":
    main()
