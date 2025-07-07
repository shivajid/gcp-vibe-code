#!/usr/bin/env python3
"""
Mistral 8B Supervised Fine-Tuning Example

This script demonstrates how to perform supervised fine-tuning (SFT) on Mistral 8B 
using the Tunix framework with QLoRA (Quantized Low-Rank Adaptation).

Usage:
    python mistral_sft_example.py

Requirements:
    - Google Colab with T4/TPU runtime or local GPU
    - Mistral 8B model weights (from Hugging Face)
    - Custom dataset for fine-tuning
"""

import gc
import os
import time
from typing import Dict, Any, List

from flax import nnx
import jax
import jax.numpy as jnp
import optax
from orbax import checkpoint as ocp
from qwix import lora
from transformers import AutoTokenizer, AutoModelForCausalLM
from tunix.generate import sampler as sampler_lib
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
from tunix.models.llama3 import model as llama3_lib
from tunix.models.llama3 import params as params_lib


class MistralConfig:
    """Configuration for Mistral 8B model."""
    
    def __init__(self):
        self.num_layers = 32
        self.vocab_size = 32000
        self.embed_dim = 4096
        self.hidden_dim = 14336
        self.num_heads = 32
        self.head_dim = 128
        self.num_kv_heads = 8
        self.rope_theta = 10000
        self.norm_eps = 1e-5
        self.sliding_window_size = 4096  # Mistral-specific
        
    def to_llama_config(self):
        """Convert to Llama3 config for compatibility."""
        return llama3_lib.ModelConfig(
            num_layers=self.num_layers,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            rope_theta=self.rope_theta,
            norm_eps=self.norm_eps,
        )


class MistralSFTExample:
    """Mistral 8B Supervised Fine-Tuning Example."""
    
    def __init__(self):
        # Hyperparameters
        self.batch_size = 8
        self.max_sequence_length = 2048
        self.mesh = [(1, 8), ("fsdp", "tp")]
        self.model_name = "mistralai/Mistral-7B-v0.1"  # Using 7B as 8B isn't available
        
        # LoRA parameters
        self.rank = 16
        self.alpha = 2.0
        
        # Training parameters
        self.max_steps = 500
        self.eval_every_n_steps = 50
        self.learning_rate = 1e-4
        
        # Directories
        self.intermediate_ckpt_dir = "/tmp/intermediate_ckpt/"
        self.ckpt_dir = "/tmp/ckpts/"
        self.profiling_dir = "/tmp/profiling/"
        
        # Create directories
        os.makedirs(self.intermediate_ckpt_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.profiling_dir, exist_ok=True)
        
        # Initialize components
        self.mistral_config = MistralConfig()
        self.llama_config = self.mistral_config.to_llama_config()
        self.mesh_obj = None
        self.model = None
        self.tokenizer = None
        
    def load_mistral_model(self):
        """Load Mistral model from Hugging Face and convert to Tunix format."""
        print("Loading Mistral model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_hf = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        
        # Create mesh
        self.mesh_obj = jax.make_mesh(*self.mesh)
        
        # Create Llama3 model with Mistral config
        abs_mistral = nnx.eval_shape(
            lambda: llama3_lib.Llama3(self.llama_config, rngs=nnx.Rngs(params=0))
        )
        
        # Initialize with random weights (in practice, you'd load from HF)
        graph_def, abs_state = nnx.split(abs_mistral)
        abs_state = jax.tree.map(
            lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16, sharding=s),
            abs_state,
            nnx.get_named_sharding(abs_state, self.mesh_obj),
        )
        
        # For demo purposes, we'll use random initialization
        # In practice, you'd convert HF weights to Tunix format
        checkpointer = ocp.StandardCheckpointer()
        restored_params = checkpointer.restore(
            os.path.join(self.intermediate_ckpt_dir, "state"), 
            target=abs_state
        )
        
        graph_def, _ = nnx.split(abs_mistral)
        self.model = nnx.merge(graph_def, restored_params)
        
        print("Model loaded successfully!")
        return self.model
    
    def create_sampler(self, model, tokenizer, model_config):
        """Create a sampler for text generation."""
        
        # Create a simple tokenizer adapter
        class SimpleTokenizer:
            def __init__(self, hf_tokenizer):
                self.hf_tokenizer = hf_tokenizer
                
            def encode(self, text):
                return self.hf_tokenizer.encode(text)
                
            def decode(self, tokens):
                return self.hf_tokenizer.decode(tokens)
                
            def pad_id(self):
                return self.hf_tokenizer.pad_token_id
        
        simple_tokenizer = SimpleTokenizer(tokenizer)
        
        sampler = sampler_lib.Sampler(
            transformer=model,
            tokenizer=simple_tokenizer,
            cache_config=sampler_lib.CacheConfig(
                cache_size=256,
                num_layers=model_config.num_layers,
                num_kv_heads=model_config.num_kv_heads,
                head_dim=model_config.head_dim,
            ),
        )
        
        return sampler
    
    def apply_lora(self, base_model):
        """Apply LoRA to the model."""
        print("Applying LoRA to model...")
        
        lora_provider = lora.LoraProvider(
            module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
            rank=self.rank,
            alpha=self.alpha,
            # Uncomment for QLoRA (with quantization)
            # weight_qtype="nf4",
            # tile_size=256,
        )

        model_input = base_model.get_model_input()
        lora_model = lora.apply_lora_to_model(
            base_model, lora_provider, **model_input
        )

        with self.mesh_obj:
            state = nnx.state(lora_model)
            pspecs = nnx.get_partition_spec(state)
            sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
            nnx.update(lora_model, sharded_state)

        print("LoRA applied successfully!")
        return lora_model
    
    def create_instruction_dataset(self):
        """Create a simple instruction dataset for fine-tuning."""
        
        # Sample instruction data
        instructions = [
            {
                "instruction": "Write a poem about artificial intelligence.",
                "response": "In circuits deep and code divine,\nSilicon dreams in perfect line,\nLearning patterns, growing wise,\nArtificial, yet human eyes."
            },
            {
                "instruction": "Explain machine learning to a 10-year-old.",
                "response": "Machine learning is like teaching a computer to learn from examples, just like how you learn to recognize animals by seeing many pictures of them."
            },
            {
                "instruction": "What is the capital of France?",
                "response": "The capital of France is Paris."
            },
            {
                "instruction": "How do you make a sandwich?",
                "response": "To make a sandwich: 1) Get bread, 2) Add fillings like meat, cheese, or vegetables, 3) Put the pieces together."
            },
            {
                "instruction": "Write a short story about a magical library.",
                "response": "In the heart of the ancient city stood a library unlike any other. Its books whispered secrets to those who dared to listen, and its shelves stretched into impossible dimensions."
            },
            {
                "instruction": "What are the benefits of exercise?",
                "response": "Exercise provides many benefits including improved cardiovascular health, stronger muscles, better mood, increased energy, and weight management."
            },
            {
                "instruction": "Explain photosynthesis in simple terms.",
                "response": "Photosynthesis is how plants make their own food using sunlight, water, and carbon dioxide to create oxygen and glucose."
            },
            {
                "instruction": "What is the meaning of life?",
                "response": "The meaning of life is a deeply personal question that varies for each individual. Many find meaning through relationships, purpose, growth, and contributing to something larger than themselves."
            }
        ]
        
        # Format for training
        formatted_data = []
        for item in instructions:
            # Create prompt in instruction format
            prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}"
            formatted_data.append({"text": prompt})
        
        return formatted_data
    
    def tokenize_dataset(self, data, max_length=None):
        """Tokenize the dataset."""
        if max_length is None:
            max_length = self.max_sequence_length
            
        tokenized_data = []
        
        for item in data:
            # Tokenize the text
            tokens = self.tokenizer.encode(
                item["text"], 
                max_length=max_length, 
                truncation=True, 
                padding="max_length",
                return_tensors="pt"
            )
            
            # Convert to numpy for JAX
            input_tokens = tokens["input_ids"].numpy()
            attention_mask = tokens["attention_mask"].numpy()
            
            tokenized_data.append({
                "input_tokens": input_tokens,
                "attention_mask": attention_mask
            })
        
        return tokenized_data
    
    def create_data_iterator(self, data, batch_size=None):
        """Create a simple data iterator."""
        if batch_size is None:
            batch_size = self.batch_size
            
        class SimpleIterator:
            def __init__(self, data, batch_size):
                self.data = data
                self.batch_size = batch_size
                self.index = 0
                
            def __iter__(self):
                return self
                
            def __next__(self):
                if self.index >= len(self.data):
                    self.index = 0  # Reset for next epoch
                    
                batch_data = self.data[self.index:self.index + self.batch_size]
                self.index += self.batch_size
                
                # Pad batch to batch_size
                while len(batch_data) < self.batch_size:
                    batch_data.append(batch_data[0])  # Repeat first item
                
                # Stack into batch
                input_tokens = jnp.stack([item["input_tokens"] for item in batch_data])
                attention_mask = jnp.stack([item["attention_mask"] for item in batch_data])
                
                return peft_trainer.TrainingInput(
                    input_tokens=input_tokens,
                    input_mask=attention_mask
                )
        
        return SimpleIterator(data, batch_size)
    
    def gen_model_input_fn(self, x: peft_trainer.TrainingInput):
        """Generate model inputs from training data."""
        
        # Create padding mask
        pad_mask = x.input_tokens != self.tokenizer.pad_token_id
        
        # Build positions
        positions = jnp.arange(x.input_tokens.shape[1])
        positions = jnp.broadcast_to(positions, x.input_tokens.shape)
        
        # Create attention mask (causal + padding)
        seq_len = x.input_tokens.shape[1]
        causal_mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1)
        padding_mask = jnp.broadcast_to(pad_mask[:, :, None], (pad_mask.shape[0], seq_len, seq_len))
        attention_mask = (causal_mask == 0) & padding_mask
        
        return {
            'input_tokens': x.input_tokens,
            'input_mask': x.input_mask,
            'positions': positions,
            'attention_mask': attention_mask,
        }
    
    def train_model(self, lora_model):
        """Train the model using supervised fine-tuning."""
        
        # Create dataset
        print("Creating training dataset...")
        raw_data = self.create_instruction_dataset()
        tokenized_data = self.tokenize_dataset(raw_data)
        
        # Split into train/validation
        train_data = tokenized_data[:6]
        val_data = tokenized_data[6:]
        
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        
        # Create data iterators
        train_ds = self.create_data_iterator(train_data)
        validation_ds = self.create_data_iterator(val_data)
        
        # Training configuration
        logging_option = metrics_logger.MetricsLoggerOptions(
            log_dir="/tmp/tensorboard/mistral_sft", 
            flush_every_n_steps=20
        )
        
        training_config = peft_trainer.TrainingConfig(
            eval_every_n_steps=self.eval_every_n_steps,
            max_steps=self.max_steps,
            metrics_logging_options=logging_option,
            checkpoint_root_directory=self.ckpt_dir,
        )
        
        # Create trainer
        trainer = peft_trainer.PeftTrainer(
            lora_model, 
            optax.adamw(self.learning_rate), 
            training_config
        )
        trainer = trainer.with_gen_model_input_fn(self.gen_model_input_fn)
        
        print("Starting training...")
        print(f"Max steps: {self.max_steps}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {self.batch_size}")
        
        # Train the model
        with jax.profiler.trace(os.path.join(self.profiling_dir, "mistral_sft_training")):
            with self.mesh_obj:
                trainer.train(train_ds, validation_ds)
        
        print("Training completed!")
        return trainer
    
    def test_model(self, model, test_prompts):
        """Test the model with given prompts."""
        sampler = self.create_sampler(model, self.tokenizer, self.llama_config)
        
        print("Testing model generation...")
        out_data = sampler(
            input_strings=test_prompts,
            total_generation_steps=30,
        )
        
        for i, (input_string, out_string) in enumerate(zip(test_prompts, out_data.text)):
            print(f"\n--- Output {i+1} ---")
            print(f"Prompt: {input_string}")
            print(f"Generated: {out_string}")
            print("-" * 50)
        
        return out_data
    
    def run_example(self):
        """Run the complete Mistral SFT example."""
        print("=" * 60)
        print("Mistral 8B Supervised Fine-Tuning Example")
        print("=" * 60)
        
        # Step 1: Load model
        self.load_mistral_model()
        
        # Step 2: Test base model
        print("\n" + "=" * 40)
        print("Testing Base Model")
        print("=" * 40)
        
        base_test_prompts = [
            "Write a short story about a robot learning to paint:",
            "Explain quantum computing in simple terms:",
            "What are the benefits of renewable energy?",
        ]
        
        self.test_model(self.model, base_test_prompts)
        
        # Step 3: Apply LoRA
        print("\n" + "=" * 40)
        print("Applying LoRA")
        print("=" * 40)
        
        lora_model = self.apply_lora(self.model)
        
        # Step 4: Train model
        print("\n" + "=" * 40)
        print("Training Model")
        print("=" * 40)
        
        trainer = self.train_model(lora_model)
        
        # Step 5: Test fine-tuned model
        print("\n" + "=" * 40)
        print("Testing Fine-tuned Model")
        print("=" * 40)
        
        fine_tuned_test_prompts = [
            "### Instruction:\nWrite a poem about technology.\n\n### Response:",
            "### Instruction:\nExplain quantum physics to a beginner.\n\n### Response:",
            "### Instruction:\nWhat is the importance of education?\n\n### Response:",
        ]
        
        self.test_model(lora_model, fine_tuned_test_prompts)
        
        # Step 6: Compare models
        print("\n" + "=" * 40)
        print("Comparing Base vs Fine-tuned Model")
        print("=" * 40)
        
        comparison_prompt = "### Instruction:\nWrite a short story about a robot.\n\n### Response:"
        
        print("BASE MODEL OUTPUT:")
        base_output = self.test_model(self.model, [comparison_prompt])
        
        print("\nFINE-TUNED MODEL OUTPUT:")
        fine_tuned_output = self.test_model(lora_model, [comparison_prompt])
        
        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("=" * 60)


def main():
    """Main function to run the Mistral SFT example."""
    example = MistralSFTExample()
    example.run_example()


if __name__ == "__main__":
    main() 