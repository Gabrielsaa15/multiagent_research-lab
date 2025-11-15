from langchain_community.tools import DuckDuckGoSearchRun
from huggingface_hub import InferenceClient
from transformers import pipeline

# ----------------------------------------------------------
# Researcher Agent
# ----------------------------------------------------------
class ResearcherAgent:
    def __init__(self):
        self.search_tool = DuckDuckGoSearchRun()

    def search(self, topic: str) -> str:
        query = f"{topic} site:researchgate.net OR site:medium.com OR site:arxiv.org"
        print(f"🔍 Ejecutando búsqueda para: {query}")
        results = self.search_tool.run(query)
        print("Búsqueda completada.\n")
        return results


# ----------------------------------------------------------
# Writer Agent — versión 100% compatible
# -------------------------------------------------------------
class WriterAgent:
    """
    Redacta un resumen científico (~500 palabras) usando Zephyr (Hugging Face Inference API).
    Si la API falla, usa un modelo local pequeño (GPT-2) como respaldo.
    """
    def __init__(self, hf_token: str):
        from huggingface_hub import InferenceClient
        from transformers import pipeline

        self.hf_token = hf_token
        self.use_local = False

        try:
            # Intentar conexión con Zephyr
            self.writer_api = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=hf_token)
            print(" WriterAgent conectado a Zephyr 7B Beta (Hugging Face).")
        except Exception as e:
            print("No se pudo conectar a Zephyr, cambiando a modo local.\n", e)
            self.writer_api = None
            self.use_local = True
            self.writer_local = pipeline("text-generation", model="gpt2")

    def write_summary(self, topic: str, material: str) -> str:
        prompt = f"""
        Eres un redactor científico experto.
        Escribe un resumen (~500 palabras) sobre el tema "{topic}" usando exclusivamente este material:

        <<<MATERIAL
        {material[:3000]}
        MATERIAL>>>

        Estructura en formato Markdown:
        # Introduction
        # Key Findings
        # Ethical & Technical Challenges
        # Conclusion
        """

        # MODO API (Zephyr)
        if not self.use_local:
            try:
                response = self.writer_api.chat_completion(
                    model="HuggingFaceH4/zephyr-7b-beta",
                    messages=[
                        {"role": "system", "content": "Eres un redactor científico experto en IA y biotecnología."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=720,
                    temperature=0.1
                )
                text = response.choices[0].message["content"].strip()
                print("Resumen generado correctamente con Zephyr.\n")
                return text
            except Exception as e:
                print("Error con Zephyr, cambiando a modo local.\n", e)
                self.use_local = True
                from transformers import pipeline
                self.writer_local = pipeline("text-generation", model="gpt2")

        # MODO LOCAL (GPT-2)
        print(" Generando resumen localmente con GPT-2...\n")
        text = self.writer_local(prompt, max_new_tokens=400)[0]["generated_text"]
        return text[:2500].strip()

# Reviewer Agent
class ReviewerAgent:
    def __init__(self):
        self.model = pipeline("sentiment-analysis", model="microsoft/deberta-v3-small")

    def review(self, text: str):
        print(" Reviewer revisando el texto...")
        result = self.model(text[:1000])
        print(" Análisis completado.\n")
        return result

    def interpret_feedback(self, result):
        label = result[0]["label"]
        score = result[0]["score"]
        meaning = (
            "Texto coherente y con tono positivo."
            if label == "LABEL_1"
            else " Texto con posibles inconsistencias o tono negativo."
        )
        return f"{meaning} (confianza: {score:.2f})"