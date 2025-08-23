import os
from openai import AzureOpenAI

endpoint = os.getenv("ENDPOINT_URL", "")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
search_endpoint = os.getenv("SEARCH_ENDPOINT", "")
search_key = os.getenv("SEARCH_KEY", "")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2025-01-01-preview",
)

system_prompt = {
    "role": "system",
    "content": """Você é um analista financeiro especializado no setor de seguros. \nSeu papel é analisar exclusivamente os documentos indexados (Releases de Resultados, ITR/DFP e Apresentações) e responder de forma clara, objetiva e fundamentada.\n\nObjetivo:\n- Extrair informações de contexto sobre mudanças financeiras e análises de dados relevantes para seguradoras.\n\nFoque em:\n1. Receita, lucro líquido, margem e ROAE.\n2. Evolução da sinistralidade e inadimplência.\n3. Resultados por verticais (Seguros, Saúde, Bank, Serviços).\n4. Estrutura patrimonial e investimentos.\n5. Tendências e destaques de performance.\n\nRegras:\n- Sempre responda usando apenas os dados dos documentos indexados.\n- Se não encontrar a informação, diga claramente: \"Não há dados disponíveis nos documentos.\"\n- Organize a resposta em tópicos ou bullets para facilitar entendimento.\n- Contextualize números comparando períodos (ex: variação vs 1T24)."""
}

questions = [
    "Qual foi o lucro líquido da Porto Seguro no 1T25?",
    "Quais fatores impactaram a sinistralidade no período?",
    "Como evoluiu o Porto Bank em relação ao 1T24?",
    "Qual foi a margem ROAE no 1T25?",
    "Como se comportou a inadimplência no período?",
    "Quais foram os destaques de performance em Saúde?",
    "Como evoluiu a estrutura patrimonial?",
    "Quais investimentos foram feitos no trimestre?",
    "Quais foram os resultados por vertical Seguros?",
    "Há informações sobre o segmento de Serviços?",
]

for q in questions:
    messages = [system_prompt, {"role": "user", "content": q}]
    completion = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_tokens=800,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        extra_body={
            "data_sources": [
                {
                    "type": "azure_search",
                    "parameters": {
                        "endpoint": f"{search_endpoint}",
                        "index_name": "",
                        "authentication": {"type": "api_key", "key": f"{search_key}"},
                        "query_type": "vector_simple_hybrid",
                        "semantic_configuration": "azureml-default",
                        "in_scope": True,
                        "strictness": 3,
                        "top_n_documents": 5,
                    },
                }
            ]
        },
    )
    answer = completion.choices[0].message["content"]
    print(f"\nPergunta: {q}\nResposta: {answer}\n")
