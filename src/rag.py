"""
RAG System using ChromaDB + e5-small + Groq
With streaming support
"""
import json
import logging
from pathlib import Path
from typing import Optional, Generator
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
CHROMA_DIR = Path('Results/chroma_db')
STATS_DB_PATH = Path('Results/stats_db.json')

class ClimateRAG:
    """RAG system for climate data using ChromaDB + Groq"""
    
    def __init__(self, groq_api_key: str):
        """Initialize RAG system"""
        self.groq_client = Groq(api_key=groq_api_key)
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        
    def initialize_embeddings(self):
        """Load e5-small embedding model (fast, local)"""
        if self.embedding_model is None:
            logger.info("Loading e5-small embedding model...")
            self.embedding_model = SentenceTransformer('intfloat/e5-small-v2')
            logger.info("✅ Embedding model loaded")
    
    def initialize_chroma(self, reset: bool = False):
        """Initialize ChromaDB with persistence"""
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        
        if reset:
            try:
                self.chroma_client.delete_collection("climate_data")
                logger.info("🗑️ Deleted existing collection")
            except:
                pass
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="climate_data",
            metadata={"description": "Algeria climate change analysis"}
        )
        
        logger.info(f"✅ ChromaDB initialized ({self.collection.count()} documents)")
    
    def load_and_embed_data(self):
        """Load stats_db.json and embed into ChromaDB"""
        if not STATS_DB_PATH.exists():
            raise FileNotFoundError(
                f"{STATS_DB_PATH} not found. Run 'python src/generate_stats_db.py' first."
            )
        
        with open(STATS_DB_PATH, 'r') as f:
            stats_db = json.load(f)
        
        documents = []
        metadatas = []
        ids = []
        
        # Add trend data
        for var_name, trend_data in stats_db.get('trends', {}).items():
            doc_text = f"Variable: {var_name}\n"
            doc_text += f"Trend: {trend_data.get('trend', 'N/A')}\n"
            doc_text += f"P-value: {trend_data.get('p', 'N/A')}\n"
            doc_text += f"Sen's Slope: {trend_data.get('slope', 'N/A')}\n"
            doc_text += f"Significance: {'Yes' if trend_data.get('h', False) else 'No'}\n"
            
            documents.append(doc_text)
            metadatas.append({'type': 'trend', 'variable': var_name})
            ids.append(f"trend_{var_name}")
        
        # Add forecast summaries
        for model_name, forecast_data in stats_db.get('forecasts', {}).items():
            doc_text = f"Model: {model_name}\n"
            doc_text += f"Forecast Period: {forecast_data.get('forecast_period', 'N/A')}\n"
            doc_text += f"Variables: {', '.join(forecast_data.get('variables', []))}\n"
            
            summary_stats = forecast_data.get('summary_statistics', {})
            for var, stats in summary_stats.items():
                if isinstance(stats, dict):
                    doc_text += f"\n{var} Statistics:\n"
                    doc_text += f"  Mean: {stats.get('mean', 'N/A')}\n"
                    doc_text += f"  Std: {stats.get('std', 'N/A')}\n"
                    doc_text += f"  Min: {stats.get('min', 'N/A')}\n"
                    doc_text += f"  Max: {stats.get('max', 'N/A')}\n"
            
            documents.append(doc_text)
            metadatas.append({'type': 'forecast', 'model': model_name})
            ids.append(f"forecast_{model_name}")
            
            # Add monthly forecast data grouped by year
            full_forecast = forecast_data.get('full_forecast', [])
            if full_forecast:
                # Extract variable info for clear labeling
                variable_name = forecast_data.get('variable', 'Unknown')
                unit = forecast_data.get('unit', '')
                
                # Determine readable label
                if 'temperature' in variable_name.lower():
                    var_label = 'Temperature'
                    unit_suffix = f' {unit}' if unit else ' °C'
                elif 'et0' in variable_name.lower() or 'evapotranspiration' in variable_name.lower():
                    var_label = 'ET0'
                    unit_suffix = f' {unit}' if unit else ' mm'
                else:
                    var_label = variable_name
                    unit_suffix = f' {unit}' if unit else ''
                
                # Group forecasts by year for better retrieval
                year_forecasts = {}
                for entry in full_forecast:
                    date = entry.get('Date', '')
                    year = date[:4] if date else None
                    if year:
                        if year not in year_forecasts:
                            year_forecasts[year] = []
                        year_forecasts[year].append(entry)
                
                # Create a document for each year
                for year, entries in year_forecasts.items():
                    doc_text = f"Monthly {var_label} Forecasts for {year} (Algeria):\n"
                    for entry in entries:
                        date = entry.get('Date', 'N/A')
                        value = entry.get('RL_Best_Forecast', 'N/A')
                        model = entry.get('Model_Used', 'N/A')
                        if isinstance(value, (int, float)):
                            doc_text += f"  {date}: {var_label} = {value:.2f}{unit_suffix} (Model: {model})\n"
                        else:
                            doc_text += f"  {date}: {var_label} = {value}{unit_suffix} (Model: {model})\n"
                    
                    documents.append(doc_text)
                    metadatas.append({'type': 'monthly_forecast', 'model': model_name, 'year': year, 'variable': var_label})
                    ids.append(f"monthly_{model_name}_{year}")
        
        if documents:
            logger.info(f"Embedding {len(documents)} documents...")
            self.initialize_embeddings()
            
            embeddings = self.embedding_model.encode(
                documents,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            self.collection.add(
                documents=documents,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"✅ Added {len(documents)} documents to ChromaDB")
        else:
            logger.warning("No documents to embed")
    
    def _create_balanced_prompt(self, question: str, context: str) -> str:
        """Create a balanced, user-friendly prompt for the showcase"""
        return f"""You are a Climate Data Assistant for Algeria.

CLIMATE DATA:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Examine the CLIMATE DATA carefully - it contains monthly forecasts organized by year (2024-2040)
2. Provide DIRECT answers with specific values from the data
3. Only say "data not available" if the information truly does NOT exist above
4. Do NOT pad responses with unrelated global statistics
5. For trend questions spanning multiple years, include data from beginning, middle, AND end of the range
6. ALWAYS include units: Temperature in °C, ET0 in mm/month

RESPONSE FORMAT:

### Answer
Direct, concise answer using exact values from the data.

### Data Points
- **Metric**: value (unit)

For trend questions, show representative samples across the full time range asked."""

    def _get_context(self, question: str, n_results: int = 10) -> str:
        """Retrieve relevant context for a question"""
        self.initialize_embeddings()
        query_embedding = self.embedding_model.encode([question])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        return "\n\n".join(results['documents'][0])
    
    def query(self, question: str, n_results: int = 10) -> str:
        """Query the RAG system (non-streaming)"""
        context = self._get_context(question, n_results)
        prompt = self._create_balanced_prompt(question, context)
        
        try:
            response = self.groq_client.chat.completions.create(
                model="moonshotai/kimi-k2-instruct-0905",
                messages=[
                    {"role": "system", "content": "You are a climate data analyst specializing in Algeria's climate trends."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1024
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return f"Error generating response: {str(e)}"
    
    def query_stream(self, question: str, n_results: int = 10) -> Generator[str, None, None]:
        """Query the RAG system with streaming response"""
        context = self._get_context(question, n_results)
        prompt = self._create_balanced_prompt(question, context)
        
        try:
            stream = self.groq_client.chat.completions.create(
                model="moonshotai/kimi-k2-instruct-0905",
                messages=[
                    {"role": "system", "content": "You are a climate data analyst specializing in Algeria's climate trends."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1024,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            yield f"Error generating response: {str(e)}"


def init_rag_system(groq_api_key: str, reset_db: bool = False) -> ClimateRAG:
    """Initialize the RAG system"""
    rag = ClimateRAG(groq_api_key)
    rag.initialize_chroma(reset=reset_db)
    
    if rag.collection.count() < 2:
        logger.info(f"Collection count low ({rag.collection.count()}), reloading data...")
        rag.load_and_embed_data()
    
    return rag
