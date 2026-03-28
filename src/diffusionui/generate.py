import gc
import logging
import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk

import huggingface_hub as hf_hub
import openvino_genai as ov_genai
from PIL import Image, ImageTk

MODEL_ID = "OpenVINO/stable-diffusion-v1-5-int8-ov"
DEVICE = "GPU"
COLLECTION_ID = "OpenVINO/image-generation"


def get_available_models() -> list[str]:
    """Fetch available models from the OpenVINO/image-generation collection."""
    try:
        collection = hf_hub.get_collection(COLLECTION_ID)
        models = [item.item_id for item in collection.items if hasattr(item, "item_id")]
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
        gc.collect()
        self.pipeline = ov_genai.Text2ImagePipeline(model_path, self.device)
        status_callback("")

    def generate_image(
            self,
            prompt: str,
            status_callback: Callable[[str], None],
            progress_callback: Callable[[int, int], None] | None = None,
            **kwargs
    ) -> list[Image.Image]:
        self._ensure_pipeline(status_callback=status_callback)
        assert self.pipeline is not None
        status_callback("Generating image…")
        # Build generation parameters - use default steps if not provided
        gen_params = kwargs.copy()

        # Create a callback that updates progress
        gen_params["callback"] = lambda step, num_steps, latent: progress_callback(step, num_steps)

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

        # Initialize dynamically created attributes to None so IDE recognizes them
        self.steps_var: tk.StringVar = tk.StringVar(value="")
        self.negative_prompt_text: tk.Text = tk.Text(self)  # type: ignore
        self.guidance_scale_var: tk.StringVar = tk.StringVar(value="")
        self.prompt_2_text: tk.Text = tk.Text(self)  # type: ignore
        self.negative_prompt_2_text: tk.Text = tk.Text(self)  # type: ignore
        self.height_var: tk.StringVar = tk.StringVar(value="")
        self.width_var: tk.StringVar = tk.StringVar(value="")
        self.prompt_3_text: tk.Text = tk.Text(self)  # type: ignore
        self.negative_prompt_3_text: tk.Text = tk.Text(self)  # type: ignore
        self.num_images_var: tk.StringVar = tk.StringVar(value="")
        self.seed_var: tk.StringVar = tk.StringVar(value="")
        self.strength_var: tk.StringVar = tk.StringVar(value="")
        self.max_seq_length_var: tk.StringVar = tk.StringVar(value="")
        self.prompt_text: tk.Text = tk.Text(self)  # type: ignore
        self.device_var: tk.StringVar = tk.StringVar(value="")
        self.device_combo: ttk.Combobox
        self.model_var: tk.StringVar = tk.StringVar(value="")
        self.model_combo: ttk.Combobox
        self.generate_button: ttk.Button
        self.status_var: tk.StringVar = tk.StringVar(value="")
        self.progress_var: tk.DoubleVar = tk.DoubleVar(value=0)
        self.progress_bar: ttk.Progressbar
        self.preview_frame: ttk.LabelFrame
        self.preview_photos: list[ImageTk.PhotoImage] = []

        self._build_layout()

    @staticmethod
    def _create_spinbox_field(
            frame,
            label_text: str,
            var,
            from_val: float,
            to_val: float,
            width: int,
            row: int,
            col_label: int,
            col_widget: int,
            sticky: str = "w"
    ) -> None:
        """Helper to create a label + spinbox field."""
        ttk.Label(frame, text=label_text).grid(row=row, column=col_label, sticky="e")
        spinbox = ttk.Spinbox(frame, from_=from_val, to=to_val, textvariable=var, width=width)
        spinbox.grid(row=row, column=col_widget, sticky=sticky, padx=(8, 8))

    @staticmethod
    def _create_text_field(
            frame,
            label_text: str,
            height: int,
            width: int,
            row: int,
            col_label: int,
            col_widget: int
    ) -> tk.Text:
        """Helper to create a label + text field."""
        ttk.Label(frame, text=label_text).grid(row=row, column=col_label, sticky="nw", padx=(0, 8))
        text_widget = tk.Text(frame, height=height, width=width, wrap="word")
        text_widget.grid(row=row, column=col_widget, sticky="ew", padx=(0, 8))
        return text_widget

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
        self.device_var = tk.StringVar(value="GPU")
        self.device_combo = ttk.Combobox(device_frame, textvariable=self.device_var,
            values=["CPU", "GPU"], state="readonly", width=8, )
        self.device_combo.pack(side="left")

        model_frame = ttk.Frame(controls)
        model_frame.grid(row=0, column=3, sticky="ne", padx=(0, 8))

        ttk.Label(model_frame, text="Model:").pack(side="left", padx=(0, 8))
        self.model_var = tk.StringVar(value=MODEL_ID.replace("OpenVINO/", ""))
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
            values=self.available_models, state="readonly", width=28, )
        self.model_combo.pack(side="left")

        self.generate_button = ttk.Button(controls, text="Generate", command=self._start_generation)
        self.generate_button.grid(row=0, column=4, sticky="n", padx=(8, 0))

        controls.columnconfigure(1, weight=1)

        # Optional parameters frame
        params_frame = ttk.LabelFrame(root, text="Optional Parameters", padding=8)
        params_frame.pack(fill="x", pady=(10, 0))

        # Define parameters with their properties: (attr_name, label, type, row, col_label,
        # col_widget, from, to, width, height)
        spinbox_params = [("steps_var", "Steps:", int, 0, 0, 1, 1, 100, 8),
            ("guidance_scale_var", "Guidance Scale:", float, 0, 4, 5, 0.0, 20.0, 8),
            ("height_var", "Height:", int, 1, 4, 5, 0, 2048, 8),
            ("width_var", "Width:", int, 2, 4, 5, 0, 2048, 8),
            ("num_images_var", "Images per Prompt:", int, 3, 0, 1, 0, 10, 8),
                          ("seed_var", "Seed:", int, 3, 2, 3, 0, 2147483647, 12),
            ("strength_var", "Strength (0-1):", float, 3, 4, 5, 0.0, 1.0, 8),
            ("max_seq_length_var", "Max Sequence Length:", int, 4, 0, 1, 0, 512, 12), ]

        text_params = [("negative_prompt_text", "Negative Prompt:", 0, 2, 3),
            ("prompt_2_text", "Prompt 2:", 1, 0, 1),
            ("negative_prompt_2_text", "Negative Prompt 2:", 1, 2, 3),
            ("prompt_3_text", "Prompt 3:", 2, 0, 1),
            ("negative_prompt_3_text", "Negative Prompt 3:", 2, 2, 3), ]

        # Create spinbox parameters
        for attr_name, label, converter, row, col_label, col_widget, from_val, to_val, width in (
                spinbox_params):
            setattr(self, attr_name.replace("_var", "_var"), tk.StringVar(value=""))
            self._create_spinbox_field(params_frame, label, getattr(self, attr_name), from_val,
                                       to_val, width, row, col_label, col_widget)

        # Create text parameters
        for attr_name, label, row, col_label, col_widget in text_params:
            text_widget = self._create_text_field(params_frame, label, 2, 40, row, col_label,
                                                  col_widget)
            setattr(self, attr_name, text_widget)

        params_frame.columnconfigure(1, weight=1)
        params_frame.columnconfigure(3, weight=1)
        params_frame.columnconfigure(5, weight=1)

        # Set minimum widths to prevent columns from squishing
        params_frame.columnconfigure(0, minsize=120)
        params_frame.columnconfigure(1, minsize=200)
        params_frame.columnconfigure(2, minsize=140)
        params_frame.columnconfigure(3, minsize=200)
        params_frame.columnconfigure(4, minsize=140)
        params_frame.columnconfigure(5, minsize=100)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(root, textvariable=self.status_var).pack(anchor="w", pady=(10, 8))

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(root, variable=self.progress_var, maximum=100,
                                             mode="determinate")
        self.progress_bar.pack(fill="x", pady=(0, 8))

        self.preview_frame = ttk.LabelFrame(root, text="Images", padding=8)
        self.preview_frame.pack(fill="both", expand=True)

        self.preview_photos: list[ImageTk.PhotoImage] = []

        self.prompt_text.focus_set()
        self.bind("<Control-Return>", lambda
            _event: self._start_generation())

    def _collect_optional_parameters(
            self
            ) -> dict:
        """Collect optional parameters from UI and return as kwargs dict."""
        kwargs = {}

        # Helper to add numeric parameter
        def add_if_set(
                var,
                key: str,
                converter
                ) -> None:
            value = var.get().strip() if hasattr(var, 'get') else ""
            if value:
                kwargs[key] = converter(value)

        # Helper to add text parameter
        def add_text_if_set(
                widget,
                key: str
                ) -> None:
            value = widget.get("1.0", "end-1c").strip()
            if value:
                kwargs[key] = value

        add_if_set(self.steps_var, "num_inference_steps", int)
        add_text_if_set(self.negative_prompt_text, "negative_prompt")
        add_if_set(self.guidance_scale_var, "guidance_scale", float)
        add_text_if_set(self.prompt_2_text, "prompt_2")
        add_text_if_set(self.negative_prompt_2_text, "negative_prompt_2")
        add_text_if_set(self.prompt_3_text, "prompt_3")
        add_text_if_set(self.negative_prompt_3_text, "negative_prompt_3")
        add_if_set(self.height_var, "height", int)
        add_if_set(self.width_var, "width", int)
        add_if_set(self.num_images_var, "num_images_per_prompt", int)
        add_if_set(self.seed_var, "rng_seed", int)
        add_if_set(self.strength_var, "strength", float)
        add_if_set(self.max_seq_length_var, "max_sequence_length", int)

        return kwargs

    def _start_generation(
            self
            ) -> None:
        prompt = self.prompt_text.get("1.0", "end-1c").strip()
        if not prompt:
            messagebox.showwarning("Missing prompt", "Please enter a prompt.")
            return

        self.generate_button.config(state=tk.DISABLED)
        self.progress_var.set(0)

        # Check if device or model has changed and recreate generator if needed
        selected_device = self.device_var.get()
        selected_model = get_full_model_id(self.model_var.get())
        if self.generator.device != selected_device or self.generator.model_id != selected_model:
            self.generator = ImageGenerator(model_id=selected_model, device=selected_device)

        # Collect optional parameters
        kwargs = self._collect_optional_parameters()

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
                                                   progress_callback=self._update_progress_from_worker,
                                                   **kwargs)
            self.after(0, self._on_generation_success, images)
        except Exception as error:  # noqa: BLE001 - surface any generation failure in the UI
            self.after(0, self._on_generation_error, error)

    def _set_status_from_worker(
            self,
            text: str
            ) -> None:
        self.after(0, self.status_var.set, text)

    def _update_progress_from_worker(
            self,
            step: int,
            num_steps: int
            ) -> None:
        # Convert step/num_steps to percentage
        gc.collect()
        percentage = (step / num_steps) * 100 if num_steps > 0 else 0
        self.after(0, self.progress_var.set, percentage)

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

        # Calculate grid layout (prefer wider layouts)
        num_images = len(images)
        cols = int(num_images ** 0.5)
        rows = (num_images + cols - 1) // cols

        # Try incrementing columns to see if we can reduce rows (prefer wider)
        cols_plus_1 = cols + 1
        rows_if_plus_1 = (num_images + cols_plus_1 - 1) // cols_plus_1
        if rows_if_plus_1 < rows:
            cols = cols_plus_1
            rows = rows_if_plus_1

        # Update the preview frame to get available space
        self.preview_frame.update_idletasks()
        available_width = self.preview_frame.winfo_width() - 40  # Account for padding
        available_height = self.preview_frame.winfo_height() - 40

        # Calculate size for each image in the grid
        max_img_width = max(available_width // cols, 100)  # Minimum 100px
        max_img_height = max(available_height // rows, 100)  # Minimum 100px

        # Store PhotoImage references to prevent garbage collection
        self.preview_photos = []

        # Create a grid of image labels
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols

            # Resize image to fit available space while maintaining aspect ratio
            img_copy = img.copy()
            img_copy.thumbnail((max_img_width, max_img_height), Image.Resampling.LANCZOS)

            # Convert PIL image to PhotoImage
            photo = ImageTk.PhotoImage(img_copy)
            self.preview_photos.append(photo)

            # Create label with the image
            label = ttk.Label(self.preview_frame, image=photo)
            label.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")

        # Configure grid weights for even distribution
        for i in range(rows):
            self.preview_frame.grid_rowconfigure(i, weight=1)
        for i in range(cols):
            self.preview_frame.grid_columnconfigure(i, weight=1)

        self.progress_var.set(0)
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
