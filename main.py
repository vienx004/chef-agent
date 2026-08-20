import sys
import logging
from chef_agent import config
from chef_agent.llm import GeminiLLM
from chef_agent.rag import JSONRecipeRetriever, ChromaRecipeRetriever
from chef_agent.agent import ChefAgent
from chef_agent.tools import CulinaryConverter

# Configure console logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main")

def print_welcome_banner():
    """Prints a styled startup banner for the Chef Agent."""
    print("=" * 70)
    print("          *** WELCOME TO THE CHEF AI AGENT CLI ***          ")
    print("=" * 70)
    print("Chef Gemini is ready to design menus, write recipes, and consult on")
    print("culinary science.")
    print("\nAvailable Special Commands:")
    print("  /convert <value> <unit> [ingredient]  - Convert culinary measurements")
    print("                                         e.g. /convert 350 f")
    print("                                         e.g. /convert 2.5 cups flour")
    print("                                         e.g. /convert 8 oz")
    print("  /clear                                - Clear chat conversation memory")
    print("  /exit or /quit                        - Leave the kitchen")
    print("=" * 70)
    print(f"Active Backend: Gemini ({config.LLM_PROVIDER})")
    print(f"RAG Provider:   {config.RAG_PROVIDER.upper()}")
    if config.RAG_PROVIDER == "chroma":
        print(f"Chroma DB Path: {config.CHROMA_DB_PATH}")
    else:
        print(f"Recipe DB Path: {config.RECIPE_DB_PATH}")
    print("=" * 70 + "\n")

def handle_conversion_command(cmd_args: list) -> str:
    """
    Parses and handles the /convert command.
    Supported inputs:
      - /convert 350 f
      - /convert 180 c
      - /convert 8.5 oz
      - /convert 2 cups flour
    """
    if not cmd_args:
        return "Usage: /convert <value> <unit> [ingredient]\nExample: /convert 2 cups flour"

    try:
        val = float(cmd_args[0])
    except ValueError:
        return f"Error: '{cmd_args[0]}' is not a valid number."

    if len(cmd_args) < 2:
        return "Error: Missing target unit (e.g., 'f', 'c', 'oz', 'cups')."

    unit = cmd_args[1].lower().strip()

    if unit in ["f", "fahrenheit"]:
        celsius = CulinaryConverter.fahrenheit_to_celsius(val)
        return f"[TEMP] {val}°F is approximately {celsius}°C"
    
    elif unit in ["c", "celsius"]:
        fahr = CulinaryConverter.celsius_to_fahrenheit(val)
        return f"[TEMP] {val}°C is approximately {fahr}°F"

    elif unit in ["oz", "ounces", "ounce"]:
        grams = CulinaryConverter.ounces_to_grams(val)
        return f"[WEIGHT] {val} oz is approximately {grams}g"

    elif unit in ["g", "grams", "gram"]:
        oz = CulinaryConverter.grams_to_ounces(val)
        return f"[WEIGHT] {val}g is approximately {oz} oz"

    elif unit in ["cup", "cups"]:
        ingredient = "liquid"
        if len(cmd_args) >= 3:
            ingredient = " ".join(cmd_args[2:])
        
        grams = CulinaryConverter.cups_to_grams(val, ingredient)
        return f"[VOLUME] {val} cup(s) of '{ingredient}' is approximately {grams}g"

    else:
        return f"Error: Unsupported unit '{unit}'. Use 'f', 'c', 'oz', 'g', or 'cups'."

def main():
    # 1. Check API Key
    if not config.GEMINI_API_KEY:
        print("❌ Error: GEMINI_API_KEY is not set.")
        print("Please copy the .env.example file to .env and insert your API key:")
        print("   cp .env.example .env")
        sys.exit(1)

    # 2. Instantiate LLM provider
    try:
        # We default to Gemini, but this architecture makes it easy to construct a different class
        if config.LLM_PROVIDER == "gemini":
            llm_client = GeminiLLM(api_key=config.GEMINI_API_KEY)
        else:
            print(f"❌ Error: Unsupported LLM_PROVIDER '{config.LLM_PROVIDER}' in configuration.")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        sys.exit(1)

    # 3. Instantiate Retriever (RAG)
    # Choose between Vector (Chroma) and Keyword (JSON) search
    if config.RAG_PROVIDER == "chroma":
        retriever = ChromaRecipeRetriever(
            llm=llm_client,
            db_path=config.CHROMA_DB_PATH,
            seed_json_path=config.RECIPE_DB_PATH
        )
    else:
        retriever = JSONRecipeRetriever(db_path=config.RECIPE_DB_PATH)

    # 4. Instantiate Agent Coordinator
    agent = ChefAgent(llm_client=llm_client, retriever=retriever)

    # 5. Start CLI interface
    print_welcome_banner()

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            # Check for special commands
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd in ["/exit", "/quit"]:
                print("\nChef: Keep cooking! Until next time. Bon appétit!")
                break

            elif cmd == "/clear":
                agent.clear_history()
                print("\n[SYSTEM] Chat history cleared! Starting a clean slate.\n")
                continue

            elif cmd == "/convert":
                conversion_result = handle_conversion_command(parts[1:])
                print(f"\n{conversion_result}\n")
                continue

            # Normal chat workflow with streaming
            print("\nChef: ", end="", flush=True)
            
            # Request response stream from agent
            stream_gen = agent.add_user_message_stream(user_input)
            
            for update in stream_gen:
                if not update["complete"]:
                    # Print chunk
                    print(update["chunk"], end="", flush=True)
                    
                    # Highlight if RAG matching triggered (only on first chunk check)
                    if update["retrieved_docs"]:
                        # We save it to output after the stream or print a header
                        docs = update["retrieved_docs"]
                        titles = [d.get("title") for d in docs]
                        # Print metadata silently to indicate RAG was queried
                        sys.stdout.write(f"\n[Matched reference recipe: {', '.join(titles)}]\nChef: ")
                        sys.stdout.flush()
                else:
                    # Streaming finished
                    print("\n")
                    if update.get("saved_recipe"):
                        saved = update["saved_recipe"]
                        print(f"[SYSTEM] Saved new recipe to database: {saved['title']}\n")
                    
        except KeyboardInterrupt:
            print("\n\nChef: Leaving in a hurry? Bon appétit!")
            break
        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}\n")

if __name__ == "__main__":
    main()
