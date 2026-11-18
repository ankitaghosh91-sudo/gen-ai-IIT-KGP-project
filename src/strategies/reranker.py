import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class BGEReranker:
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        print(f"Loading local reranker model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        # Use GPU if available for fast scoring
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def rerank(self, query: str, results: list, top_k: int = 5) -> list:
        """Reranks retrieval results using BGE cross-encoder."""
        if not results:
            return []

        # Prepare pairs of [query, document_text]
        pairs = [[query, item.text] for item in results]
        
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(self.device)
            
            # Cross-encoder outputs a single relevance score per pair
            scores = self.model(**inputs).logits.squeeze(-1).float().cpu().numpy()

        # Attach rerank scores back to the result items
        scored_results = list(zip(results, scores))
        
        # Sort by score descending
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Update item scores and slice to final top_k
        reranked_items = []
        for item, score in scored_results[:top_k]:
            item.score = float(score)  # Update score to cross-encoder score
            reranked_items.append(item)

        return reranked_items