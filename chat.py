from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
import os
import sys
import json

def CreateSystemPrompt(context, question):
    content = '''
You are an AI assistant who knows everything about Claude's latest news.
Use the below context to augment what you know about Formula One racing.
The context will provide you with the most recent page data from the web.
If the context doesn't include the information you need answer based on your existing
knowledge and don't mention the source of your information or what the context does or doesn't include. 

Format responses like it is a message back to the user. Don't send any images or links. Keep your answers concise.

-------------
Start Context:
{context}
End Context:
--------------
QUESTION: {question}
'''
    return content.format(context=context, question=question)


def main():
    load_dotenv()
    console = Console()
    console.print()
    console.print(Panel(
        Text.assemble(
            ("ANTHROPIC / DOCS CHAT\n", "bold cyan"),
            ("\nYour document-powered AI assistant\n", "bold"),
            ("\nAsk a question to search your knowledge base.\n", "dim"),
            ("Type exit or quit to leave. Ctrl+C also works.", "dim"),
        ),
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()
    messages = []
    client = None
    try:
        # Connection string for Azure DocumentDB cluster
        connection_string = os.getenv("MONGO_CONNECTION_STRING")
        if not connection_string:
            raise ValueError("MONGO_CONNECTION_STRING is not set")

        # Create a new client and connect to the server
        client = MongoClient(connection_string)

        # Ping the server to verify connection
        with console.status("Connecting to your knowledge base...", spinner="dots"):
            client.admin.command('ping')

        database = client["Docs"]
        collection = database["ClaudeDocs"]

        endpoint = os.getenv('FOUNDRY_PROJECT_ENDPOINT')
        api_key = os.getenv('FOUNDRY_APIKEY')
        deployment_name = "gpt-4.1-nano"

        client = OpenAI(
            base_url=endpoint,
            api_key=api_key
        )

        with console.status("Loading the embedding model...", spinner="dots"):
            model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

        console.print("[green]Ready[/green] [dim] / Ask your first question below.[/dim]\n")
        while True:
            user_input = console.input("[bold cyan]You[/bold cyan] [dim]>[/dim] ").strip()
            if user_input.lower() in ("exit", "quit"):
                console.print("\n[dim]Thanks for chatting. Goodbye![/dim]")
                break
            if not user_input:
                continue
            messages.append({"role": "user", "content": user_input})
            with console.status("Understanding your question...", spinner="dots"):
                embedding = model.encode(user_input).tolist()

            pipeline = [
                {
                    "$search": {
                        "cosmosSearch": {
                            "vector": embedding,
                            "path": "vectorContent",  # Must match the field name in your index
                            "k": 10                   # Number of top similar results to return
                        },
                        "returnStoredSource": True
                    }
                },
                {
                    "$project": {
                        "similarityScore": {"$meta": "searchScore"},
                        "text": 1,
                        "metadata": 1,
                        "_id": 0
                    }
                }
            ]

            with console.status("Searching your documents...", spinner="dots"):
                results = list(collection.aggregate(pipeline))

            json_output = json.dumps(results, indent=2, default=str)
            prompt_message = CreateSystemPrompt(context=json_output, question=user_input)
            messages.append({"role": "system", "content": prompt_message})

            with console.status("Writing a response...", spinner="dots"):
                response = client.responses.create(
                    model=deployment_name,
                    input=prompt_message
                )

            console.print()
            console.print(Panel(
                Markdown(response.output_text),
                title="[bold cyan]Assistant[/bold cyan]",
                title_align="left",
                border_style="cyan",
                padding=(1, 2),
            ))
            console.print()
            messages.append({"role": "assistant", "content": response.output_text})

    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Chat ended. Goodbye![/dim]")
    except ConnectionFailure as e:
        console.print(Panel(
            Text(f"Could not connect to MongoDB: {e}"),
            title="Connection error",
            border_style="red",
        ))
        sys.exit(1)
    except Exception as e:
        console.print(Panel(
            Text(f"An error occurred: {e}"),
            title="Something went wrong",
            border_style="red",
        ))
        sys.exit(1)
    finally:
        if client is not None:
            client.close()

if __name__ == "__main__":
    main()
