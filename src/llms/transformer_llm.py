from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from deepeval.models import DeepEvalBaseLLM
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from lmformatenforcer import JsonSchemaParser
from pydantic import BaseModel
import json
import torch

class HuggingFaceLLM(DeepEvalBaseLLM):
    def __init__(self, model_name: str = 'mistralai/Mistral-7B-Instruct-v0.3', temperature: float = 0.0):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pipeline = pipeline("text-generation", model = self.model, tokenizer = self.tokenizer, temperature = temperature).to(self.device)
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt: str, schema: BaseModel):
        parser = JsonSchemaParser(schema.schema())
        func = build_transformers_prefix_allowed_tokens_fn(self.tokenizer, parser)

        out = json.loads(self.pipeline(prompt, prefix_allowed_tokens_fn = func)[0]['generated_text'][len(prompt):])
        return schema(**out)
    
    async def a_generate(self, prompt: str, schema: BaseModel):
        return self.generate(prompt, schema)

    def load_model(self):
        return self.model
    
    def get_model_name(self):
        return self.model_name