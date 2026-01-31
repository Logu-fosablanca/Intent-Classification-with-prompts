
import asyncio
import ollama
from query_classifier.config import LLM_MODEL_NAME, LLM_API_BASE

async def test_llm_connection():
    print(f"Testing LLM Connection...")
    print(f"URL: {LLM_API_BASE}")
    print(f"Model: {LLM_MODEL_NAME}")
    
    try:
        # Initialize client with configured URL
        client = ollama.AsyncClient(host=LLM_API_BASE)
        
        # 1. Test Connection / List Models
        print("\n[1/2] Checking connection & available models...")
        try:
            models_response = await client.list()
            # Handle different versions of ollama library response structure if needed
            # usually returns {'models': [...]}
            models = models_response.get('models', [])
            print(f"[OK] Connection successful. Found {len(models)} models.")
            
            model_names = []
            for m in models:
                # Handle dictionary or object
                if isinstance(m, dict):
                    name = m.get('name') or m.get('model')
                else:
                    name = getattr(m, 'name', None) or getattr(m, 'model', None)
                model_names.append(name)
            if LLM_MODEL_NAME in model_names or f"{LLM_MODEL_NAME}:latest" in model_names:
                print(f"[OK] Configured model '{LLM_MODEL_NAME}' found on server.")
            else:
                print(f"[WARN] Configured model '{LLM_MODEL_NAME}' NOT found in list: {model_names}")
                print(f"   You might need to run: ollama pull {LLM_MODEL_NAME}")

        except Exception as e:
            print(f"[FAIL] Connection failed: {e}")
            return

        # 2. Test Inference
        print("\n[2/2] Testing inference (simple hello)...")
        try:
            response = await client.chat(
                model=LLM_MODEL_NAME, 
                messages=[{'role': 'user', 'content': 'Say "Hello" and nothing else.'}]
            )
            content = response['message']['content']
            print(f"[OK] Inference successful.")
            print(f"   Response: {content}")
        except Exception as e:
            print(f"[FAIL] Inference failed: {e}")
            if "not found" in str(e).lower():
                 print(f"   HINT: Try running 'ollama pull {LLM_MODEL_NAME}'")

    except Exception as e:
        print(f"[FAIL] An unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(test_llm_connection())
