from langchain_openai.chat_models import ChatOpenAI
from deepeval.models import DeepEvalBaseLLM


class OpenAIModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str = 'gpt-4o', temperature: int = 0):
        self.model = ChatOpenAI(model = model_name, temperature = temperature)
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt: str):
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str):
        return self.generate(prompt)

    def load_model(self):
        return self.model
    
    def get_model_name(self):
        return self.model_name

