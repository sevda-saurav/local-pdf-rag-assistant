import os
from threading import Thread

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    TextStreamer,
)

from pdf_reader import PdfReader
from local_embedding import LocalEmbedding


# Load .env variables
load_dotenv()


class AiModel:

    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        """
        Initializes the tokenizer and local LLM.
        """

        self.model_name = model_name

        print("Running checks to make sure everything is good...")

        self.hugging_face_auth()
        self.hardware_check()

        # Your machine does not have NVIDIA CUDA,
        # so we explicitly use CPU.
        self.device = torch.device("cpu")

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        print("Loading model on CPU...")
        print("This may take some time. Please wait...")

        # IMPORTANT:
        # Do NOT use device_map="auto" on this CPU-only setup.
        #
        # Loading without device_map avoids CPU/disk/meta-device
        # offloading that was causing the Tensor.item() meta error.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            trust_remote_code=True,
        )

        # Explicitly move the complete model to CPU.
        self.model = self.model.to(self.device)

        # Evaluation mode because we are doing inference only.
        self.model.eval()

        print("Model loaded successfully on CPU!")

    # ---------------------------------------------------------
    # HARDWARE
    # ---------------------------------------------------------

    def hardware_check(self):
        """
        Checks whether CUDA is available.
        """

        if torch.cuda.is_available():
            print(
                f"GPU detected: "
                f"{torch.cuda.get_device_name(0)}"
            )
        else:
            print("WARNING: No NVIDIA CUDA GPU detected.")
            print("Running the LLM on CPU.")

    # ---------------------------------------------------------
    # HUGGING FACE AUTHENTICATION
    # ---------------------------------------------------------

    def hugging_face_auth(self):
        """
        Logs into Hugging Face using the HF_TOKEN
        stored in the .env file.
        """

        hf_token = os.environ.get("HF_TOKEN")

        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN was not found in the .env file."
            )

        print("Attempting Hugging Face login...")

        login(token=hf_token)

        print("Hugging Face login successful!")

    # ---------------------------------------------------------
    # INPUT PREPARATION
    # ---------------------------------------------------------

    def _prepare_inputs(self, prompt):
        """
        Converts the user prompt into the chat format
        expected by Qwen and moves tensors to CPU.
        """

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
        )

        # Explicitly move every tensor to CPU.
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        return inputs

    # ---------------------------------------------------------
    # SIMPLE QUESTION
    # ---------------------------------------------------------

    def ask_a_question(self, prompt="Hello there!"):
        """
        Ask the local LLM a normal question.

        This version prints the generated response
        directly to the terminal.
        """

        inputs = self._prepare_inputs(prompt)

        streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        with torch.inference_mode():

            self.model.generate(
                **inputs,
                max_new_tokens=300,
                streamer=streamer,
                pad_token_id=self.tokenizer.eos_token_id,
            )

    # ---------------------------------------------------------
    # PDF RAG - NON STREAMING
    # ---------------------------------------------------------

    def ask_a_question_from_pdf(
        self,
        pdf_path,
        prompt="Tell me what this PDF is about",
    ):
        """
        Performs a complete RAG operation:

        PDF
        ↓
        Text extraction
        ↓
        Embeddings
        ↓
        Vector search
        ↓
        Relevant context
        ↓
        Qwen
        """

        # Read PDF
        pdf_reader = PdfReader(pdf_path)

        pdf_paragraphs = pdf_reader.get_paragraphs()

        # Create embedding index
        local_embedding = LocalEmbedding()

        local_embedding.build_index(pdf_paragraphs)

        # Retrieve relevant sections
        relevant_sections = local_embedding.get_context(
            prompt,
            k=10,
        )

        # Create RAG prompt
        full_prompt = self.full_prompt_for_rag(
            relevant_sections=relevant_sections,
            question_prompt=prompt,
        )

        # Prepare model input
        inputs = self._prepare_inputs(full_prompt)

        # Generate
        streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        with torch.inference_mode():

            self.model.generate(
                **inputs,
                max_new_tokens=300,
                streamer=streamer,
                pad_token_id=self.tokenizer.eos_token_id,
            )

    # ---------------------------------------------------------
    # PDF RAG - STREAMING
    # ---------------------------------------------------------

    def ask_a_question_from_pdf_stream(
        self,
        pdf_path: str,
        prompt: str = "Tell me what this PDF is about",
        local_embedding=None,
    ):
        """
        Streaming RAG function.

        Used by Streamlit's st.write_stream().

        If an embedding index is already provided,
        it is reused instead of rebuilding it.
        """

        # -----------------------------------------------------
        # Build index only if one doesn't already exist
        # -----------------------------------------------------

        if local_embedding is None:

            pdf_reader = PdfReader(pdf_path)

            pdf_paragraphs = pdf_reader.get_paragraphs()

            local_embedding = LocalEmbedding()

            local_embedding.build_index(pdf_paragraphs)

        # -----------------------------------------------------
        # Retrieve relevant PDF content
        # -----------------------------------------------------

        relevant_sections = local_embedding.get_context(
            prompt,
            k=10,
        )

        # -----------------------------------------------------
        # Build RAG prompt
        # -----------------------------------------------------

        full_prompt = self.full_prompt_for_rag(
            relevant_sections=relevant_sections,
            question_prompt=prompt,
        )

        # -----------------------------------------------------
        # Prepare model input
        # -----------------------------------------------------

        inputs = self._prepare_inputs(full_prompt)

        # -----------------------------------------------------
        # Streaming
        # -----------------------------------------------------

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=300.0,
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": 300,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # -----------------------------------------------------
        # Run generation in background thread
        # -----------------------------------------------------

        thread = Thread(
            target=self._generate,
            kwargs=generation_kwargs,
            daemon=True,
        )

        thread.start()

        # -----------------------------------------------------
        # Send generated text to Streamlit
        # -----------------------------------------------------

        for chunk in streamer:
            yield chunk

        thread.join()

    # ---------------------------------------------------------
    # SAFE GENERATION
    # ---------------------------------------------------------

    def _generate(self, **kwargs):
        """
        Runs model.generate() safely inside the
        background thread.
        """

        try:

            with torch.inference_mode():

                self.model.generate(**kwargs)

        except Exception as error:

            print("\nLLM generation failed:")
            print(error)

            raise

    # ---------------------------------------------------------
    # RAG PROMPT
    # ---------------------------------------------------------

    def full_prompt_for_rag(
        self,
        relevant_sections,
        question_prompt,
    ):
        """
        Creates the prompt used for Retrieval-Augmented Generation.
        """

        return f"""
You are an AI assistant answering questions about a document.

Use ONLY the information provided in the document context below.

If the answer cannot be found in the document context, say:

"The document does not contain information on this topic."

Do not use outside knowledge.

DOCUMENT CONTEXT:
----------------
{relevant_sections}
----------------

USER QUESTION:
{question_prompt}

Answer clearly and concisely.
"""