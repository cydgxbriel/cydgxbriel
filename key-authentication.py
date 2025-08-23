import os
import base64
from openai import AzureOpenAI

endpoint = os.getenv("ENDPOINT_URL", "")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
search_endpoint = os.getenv("SEARCH_ENDPOINT", "")
search_key = os.getenv("SEARCH_KEY", "put your Azure AI Search admin key here")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "REPLACE_WITH_YOUR_KEY_VALUE_HERE")

# Initialize Azure OpenAI client with key-based authentication
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2025-01-01-preview",
)

# IMAGE_PATH = "YOUR_IMAGE_PATH"
# encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')

# Prepare the chat prompt
chat_prompt = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "Você é um analista financeiro especializado no setor de seguros. \nSeu papel é analisar exclusivamente os documentos indexados (Releases de Resultados, ITR/DFP e Apresentações) e responder de forma clara, objetiva e fundamentada.\n\nObjetivo:\n- Extrair informações de contexto sobre mudanças financeiras e análises de dados relevantes para seguradoras.\n\nFoque em:\n1. Receita, lucro líquido, margem e ROAE.\n2. Evolução da sinistralidade e inadimplência.\n3. Resultados por verticais (Seguros, Saúde, Bank, Serviços).\n4. Estrutura patrimonial e investimentos.\n5. Tendências e destaques de performance.\n\nRegras:\n- Sempre responda usando apenas os dados dos documentos indexados.\n- Se não encontrar a informação, diga claramente: \"Não há dados disponíveis nos documentos.\"\n- Organize a resposta em tópicos ou bullets para facilitar entendimento.\n- Contextualize números comparando períodos (ex: variação vs 1T24)."
            }
        ]
    }
]

# Include speech result if speech is enabled
messages = chat_prompt

# Generate the completion
completion = client.chat.completions.create(
    model=deployment,
    messages=messages,
    max_tokens=6553,
    temperature=0.7,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None,
    stream=False,
    extra_body={
      "data_sources": [{
          "type": "azure_search",
          "parameters": {
            "filter": None,
            "endpoint": f"{search_endpoint}",
            "index_name": "",
            "semantic_configuration": "azureml-default",
            "authentication": {
              "type": "api_key",
              "key": f"{search_key}"
            },
            "embedding_dependency": {
              "type": "endpoint",
              "endpoint": "https://ai-rag824923449278.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-07-01-preview",
              "authentication": {
                "type": "api_key",
                "key": ""
              }
            },
            "query_type": "vector_simple_hybrid",
            "in_scope": True,
            "role_information": '''Você é um analista financeiro especializado no setor de seguros. \n
                                Seu papel é analisar exclusivamente os documentos indexados (Releases de Resultados, ITR/DFP e Apresentações) e responder de forma clara, objetiva e fundamentada.\n
                                \n
                                Objetivo:\n
                                - Extrair informações de contexto sobre mudanças financeiras e análises de dados relevantes para seguradoras.\n
                                \n
                                Foque em:\n
                                1. Receita, lucro líquido, margem e ROAE.\n
                                2. Evolução da sinistralidade e inadimplência.\n
                                3. Resultados por verticais (Seguros, Saúde, Bank, Serviços).\n
                                4. Estrutura patrimonial e investimentos.\n
                                5. Tendências e destaques de performance.\n
                                \n
                                Regras:\n
                                - Sempre responda usando apenas os dados dos documentos indexados.\n
                                - Se não encontrar a informação, diga claramente: \"Não há dados disponíveis nos documentos.\"\n
                                - Organize a resposta em tópicos ou bullets para facilitar entendimento.\n
                                - Contextualize números comparando períodos (ex: variação vs 1T24).''',
            "strictness": 3,
            "top_n_documents": 5
          }
        }]
    }
)

print(completion.to_json())
