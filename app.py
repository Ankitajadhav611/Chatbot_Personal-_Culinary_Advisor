import gradio as gr
import copy
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Initialize the Llama model
llm = Llama(
    ## original model
    # model_path=hf_hub_download(
    #     repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
    #     filename="Phi-3-mini-4k-instruct-q4.gguf",
    # ),
    ## compressed model
    model_path=hf_hub_download(
        repo_id="TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF",
        filename="capybarahermes-2.5-mistral-7b.Q2_K.gguf",
    ),
    n_ctx=2048,
    n_gpu_layers=50,  # Adjust based on your VRAM
)

# Initialize ChromaDB Vector Store
class VectorStore:
    def __init__(self, collection_name):
        self.embedding_model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name=collection_name)
    ## entire dataset
    # def populate_vectors(self, texts):
    #     embeddings = self.embedding_model.encode(texts, batch_size=32).tolist()
    #     for text, embedding in zip(texts, embeddings, ids):
    #         self.collection.add(embeddings=[embedding], documents=[text], ids=[doc_id])

    ## subsetting
    def populate_vectors(self, dataset):
        # Select the text columns to concatenate
        title = dataset['train']['title_cleaned'][:2000]  # Limiting to 2000 examples for the demo
        recipe = dataset['train']['recipe_new'][:2000]
        allergy = dataset['train']['allergy_type'][:2000]
        ingredients = dataset['train']['ingredients_alternatives'][:2000]

        # Concatenate the text from both columns
        texts = [f"{tit} {rep} {ingr} {alle}" for tit, rep, ingr,alle in zip(title, recipe, ingredients,allergy)]
        for i, item in enumerate(texts):
            embeddings = self.embedding_model.encode(item).tolist()
            self.collection.add(embeddings=[embeddings], documents=[item], ids=[str(i)])
    ## Method to populate the vector store with embeddings from a dataset
    def search_context(self, query, n_results=1):
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=n_results)
        return results['documents']

# Example initialization (assuming you've already populated the vector store)
dataset = load_dataset('Thefoodprocessor/recipe_new_with_features_full')
vector_store = VectorStore("embedding_vector")
vector_store.populate_vectors(dataset)


def generate_text(message, max_tokens, temperature, top_p):
    # Retrieve context from vector store
    context_results = vector_store.search_context(message, n_results=1)
    context = context_results[0] if context_results else ""

    # Create the prompt template
    prompt_template = (
        f"SYSTEM: You are a recipe generating bot.\n"
        f"SYSTEM: {context}\n"
        f"USER: {message}\n"
        f"ASSISTANT:\n"
    )

    # Generate text using the language model
    output = llm(
            prompt_template,
            # max_new_tokens=256,
            temperature=0.3,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.1,
            max_tokens=600,
            # repetition_penalty=1.1
        )

    # Process the output
    input_string = output['choices'][0]['text'].strip()
    cleaned_text = input_string.strip("[]'").replace('\\n', '\n')
    continuous_text = '\n'.join(cleaned_text.split('\n'))
    return continuous_text

# Define the Gradio interface
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your message here...", label="Message"),
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Chatbot - Your Personal Culinary Advisor: Discover What to Cook Next!",
    description="Running LLM with context retrieval from ChromaDB",
    examples=[
        ["I have leftover rice, what can I make out of it?"],
        ["I just have some milk and chocolate, what dessert can I make?"],
        ["I am allergic to coconut milk, what can I use instead in a Thai curry?"],
        ["Can you suggest a vegan breakfast recipe?"],
        ["How do I make a perfect scrambled egg?"],
        ["Can you guide me through making a soufflé?"],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
