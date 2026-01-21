"""
Test script to verify RAG integration with RL Agent forecasts
"""
import sys
from pathlib import Path

# Go to project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rag import init_rag_system
from dotenv import load_dotenv
import os

load_dotenv()

def test_rag_integration():
    """Test the RAG system with RL Agent forecasts"""
    print("=" * 60)
    print("Testing RAG Integration with RL Agent Forecasts")
    print("=" * 60)
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ Error: GROQ_API_KEY not found in .env")
        return
    
    # Initialize RAG system with reset to reload data
    print("\n1. Initializing RAG system (reloading database)...")
    rag = init_rag_system(groq_api_key, reset_db=True)
    
    # Check collection
    doc_count = rag.collection.count()
    print(f"✅ ChromaDB loaded with {doc_count} documents")
    
    # Test query
    print("\n2. Testing query about RL forecasts...")
    test_question = "What is the forecast for 2040 according to the RL model?"
    
    try:
        response = rag.query(test_question, n_results=3)
        print("\n📊 Response:")
        print("-" * 60)
        print(response)
        print("-" * 60)
        print("\n✅ Test completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during query: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_rag_integration()
