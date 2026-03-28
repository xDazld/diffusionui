import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk

import huggingface_hub as hf_hub
import openvino_genai as ov_genai
from PIL import Image, ImageTk

MODEL_ID = "OpenVINO/stable-diffusion-v1-5-int8-ov"
DEVICE = "CPU"


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
    ) -> Image.Image:
        self._ensure_pipeline(status_callback=status_callback)
        assert self.pipeline is not None
        status_callback("Generating image…")
        # Build generation parameters - use default steps if not provided
        gen_params = kwargs.copy()
        image_tensor = self.pipeline.generate(prompt, **gen_params)
        print("Generation done")
        status_callback("")
        return Image.fromarray(image_tensor.data[0])


class DiffusionUI(tk.Tk):
    def __init__(
            self
            ) -> None:
        super().__init__()
        self.title("Diffusion UI")
        self.geometry("900x900")

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
        controls.pack(fill="x")

        ttk.Label(controls, text="Prompt:").grid(row=0, column=0, sticky="w")

        self.prompt_var = tk.StringVar(value="sailing ship in storm by Rembrandt")
        self.prompt_entry = ttk.Entry(controls, textvariable=self.prompt_var)
        self.prompt_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8))

        self.generate_button = ttk.Button(controls, text="Generate", command=self._start_generation)
        self.generate_button.grid(row=0, column=2, padx=(8, 0))

        controls.columnconfigure(1, weight=1)

        # Optional parameters frame
        params_frame = ttk.LabelFrame(root, text="Optional Parameters", padding=8)
        params_frame.pack(fill="x", pady=(10, 0))

        ttk.Label(params_frame, text="Steps:").grid(row=0, column=0, sticky="w")
        self.steps_var = tk.StringVar(value="")
        self.steps_spin = ttk.Spinbox(params_frame, from_=0, to=100, textvariable=self.steps_var,
                                      width=8)
        self.steps_spin.grid(row=0, column=1, sticky="w", padx=(8, 8))

        ttk.Label(params_frame, text="Negative Prompt:").grid(row=0, column=2, sticky="e")
        self.negative_prompt_var = tk.StringVar(value="")
        self.negative_prompt_entry = ttk.Entry(params_frame, textvariable=self.negative_prompt_var)
        self.negative_prompt_entry.grid(row=0, column=3, sticky="ew", padx=(8, 8))

        ttk.Label(params_frame, text="Guidance Scale:").grid(row=0, column=4, sticky="e")
        self.guidance_scale_var = tk.StringVar(value="")
        self.guidance_scale_spin = ttk.Spinbox(params_frame, from_=0.0, to=20.0,
                                               textvariable=self.guidance_scale_var, width=8)
        self.guidance_scale_spin.grid(row=0, column=5, padx=(8, 8))

        ttk.Label(params_frame, text="Prompt 2:").grid(row=1, column=0, sticky="w")
        self.prompt_2_var = tk.StringVar(value="")
        self.prompt_2_entry = ttk.Entry(params_frame, textvariable=self.prompt_2_var)
        self.prompt_2_entry.grid(row=1, column=1, sticky="ew", padx=(8, 8))

        ttk.Label(params_frame, text="Negative Prompt 2:").grid(row=1, column=2, sticky="e")
        self.negative_prompt_2_var = tk.StringVar(value="")
        self.negative_prompt_2_entry = ttk.Entry(params_frame,
                                                 textvariable=self.negative_prompt_2_var)
        self.negative_prompt_2_entry.grid(row=1, column=3, sticky="ew", padx=(8, 8))

        ttk.Label(params_frame, text="Height:").grid(row=1, column=4, sticky="e")
        self.height_var = tk.StringVar(value="")
        self.height_spin = ttk.Spinbox(params_frame, from_=0, to=2048, textvariable=self.height_var,
                                       width=8)
        self.height_spin.grid(row=1, column=5, sticky="w", padx=(8, 8))

        ttk.Label(params_frame, text="Prompt 3:").grid(row=2, column=0, sticky="w")
        self.prompt_3_var = tk.StringVar(value="")
        self.prompt_3_entry = ttk.Entry(params_frame, textvariable=self.prompt_3_var)
        self.prompt_3_entry.grid(row=2, column=1, sticky="ew", padx=(8, 8))

        ttk.Label(params_frame, text="Negative Prompt 3:").grid(row=2, column=2, sticky="e")
        self.negative_prompt_3_var = tk.StringVar(value="")
        self.negative_prompt_3_entry = ttk.Entry(params_frame,
                                                 textvariable=self.negative_prompt_3_var)
        self.negative_prompt_3_entry.grid(row=2, column=3, sticky="ew", padx=(8, 8))

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

        preview_frame = ttk.LabelFrame(root, text="Image", padding=8)
        preview_frame.pack(fill="both", expand=True)

        self.preview_label = ttk.Label(preview_frame, anchor="center")
        self.preview_label.pack(fill="both", expand=True)

        self.prompt_entry.focus_set()
        self.bind("<Return>", lambda
            _event: self._start_generation())

    def _start_generation(
            self
            ) -> None:
        prompt = self.prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("Missing prompt", "Please enter a prompt.")
            return

        self.generate_button.config(state=tk.DISABLED)

        # Collect optional parameters - only include if explicitly set by user
        kwargs = {}
        if self.steps_var.get().strip():
            kwargs["num_inference_steps"] = int(self.steps_var.get())
        if self.negative_prompt_var.get().strip():
            kwargs["negative_prompt"] = self.negative_prompt_var.get().strip()
        if self.guidance_scale_var.get().strip():
            kwargs["guidance_scale"] = float(self.guidance_scale_var.get())
        if self.prompt_2_var.get().strip():
            kwargs["prompt_2"] = self.prompt_2_var.get().strip()
        if self.prompt_3_var.get().strip():
            kwargs["prompt_3"] = self.prompt_3_var.get().strip()
        if self.negative_prompt_2_var.get().strip():
            kwargs["negative_prompt_2"] = self.negative_prompt_2_var.get().strip()
        if self.negative_prompt_3_var.get().strip():
            kwargs["negative_prompt_3"] = self.negative_prompt_3_var.get().strip()
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
            pil_image = self.generator.generate_image(prompt,
                                                      status_callback=self._set_status_from_worker,
                                                      **kwargs)
            self.after(0, self._on_generation_success, pil_image)
        except Exception as error:  # noqa: BLE001 - surface any generation failure in the UI
            self.after(0, self._on_generation_error, error)

    def _set_status_from_worker(
            self,
            text: str
            ) -> None:
        self.after(0, self.status_var.set, text)

    def _on_generation_success(
            self,
            image: Image.Image
            ) -> None:
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview_label.config(image=self.preview_photo)

        self.status_var.set("Done")
        self.generate_button.config(state=tk.NORMAL)

    def _on_generation_error(
            self,
            error: Exception
            ) -> None:
        self.status_var.set("Generation failed")
        self.generate_button.config(state=tk.NORMAL)
        messagebox.showerror("Generation failed", str(error))


def main() -> None:
    app = DiffusionUI()
    app.mainloop()


if __name__ == "__main__":
    main()
