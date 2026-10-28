import torch
from vector_index import VectorIndex
from transformers import AutoTokenizer, AutoModel

class LocalEmbedding:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", distance_metric: str = "cosine"):
        '''
        '''
        self.model_name = model_name
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"

        print(f"[LocalEmbedding] Loading {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"[LocalEmbedding Model is ready!]")

        self.store = VectorIndex(distance_metric=distance_metric, embedding_fn = self._embed_one,)

    ### private helper
    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        '''
        '''
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)
        token_count = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / token_count

    def _embed_one(self, text: str) -> torch.Tensor:
        
        '''
        '''
        return self.get_embedding([text])[0].tolist()


    ### public api
    def get_embedding(self, texts: list[str]):
        '''
        '''
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            output = self.model(**encoded)
        embeddings = self._mean_pool(output.last_hidden_state, encoded["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu()

    def build_index(self, chunks: list[str]) -> None:
        '''
        '''
        embeddings = self.get_embedding(chunks)
        for chunk, embedding in zip(chunks, embeddings):
            self.store.add_vector(embedding.tolist(), {"content" : chunk})

        print(f"[LocalEmbedding] Indexed {len(chunks)} chunks.")

    def search(self, question: str, k: int = 3) -> list[tuple[dict, float]]:
        '''
        '''
        return self.store.search(question, k=k)

    def get_context(self, question: str, k: int = 3) -> str:
        '''
        '''
        results = self.search(question, k=k)
        return "\n\n".join(doc["content"] for doc, _ in results)