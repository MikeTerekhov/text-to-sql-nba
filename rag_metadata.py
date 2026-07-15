from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
import torch

class SQLMetadataRetriever:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.docs = []
        self.embeddings = None

    def add_documents(self, docs):
        """Store and embed schema documents"""
        self.docs = docs
        self.embeddings = self.model.encode(docs, convert_to_tensor=True)

    def retrieve(self, query, top_k=1):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        if self.embeddings is None or self.embeddings.shape[0] == 0:
            raise ValueError("No embeddings found. Did you call add_documents()?")

        available_docs = self.embeddings.shape[0]
        top_k = min(top_k, available_docs)

        # Explicitly expand the query embedding to match the number of documents
        query_expanded = query_embedding.unsqueeze(0).expand(self.embeddings.size(0), -1)
        scores = cosine_similarity(query_expanded, self.embeddings, dim=1)
        
        # Now scores should be a 1D tensor with length equal to available_docs
        top_indices = torch.topk(scores, top_k).indices.tolist()
        return [self.docs[i] for i in top_indices]


# Example usage:
if __name__ == "__main__":
    retriever = SQLMetadataRetriever()

    metadata_docs = [
    # Table: team
    "Table team: columns are id (Unique team identifier), full_name (Full team name, e.g., 'Los Angeles Lakers'), abbreviation (3-letter team code, e.g., 'LAL'), city, state, year_founded.",
    
    # Table: game
    "Table game: columns are game_date (Date of the game), team_id_home, team_id_away (Unique IDs of home and away teams), team_name_home, team_name_away (Full names of the teams), pts_home, pts_away (Points scored), wl_home (W/L result), reb_home, reb_away (Total rebounds), ast_home, ast_away (Total assists), fgm_home, fg_pct_home (Field goals), fg3m_home (Three-pointers), ftm_home (Free throws), tov_home (Turnovers), and other game-related statistics."
    ]


    retriever.add_documents(metadata_docs)

    question = "What is the most assists by the Celtics in a home game?"
    relevant = retriever.retrieve(question, top_k=1)
    print("Top match:", relevant[0])
