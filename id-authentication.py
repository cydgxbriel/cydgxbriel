import os
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

endpoint = os.getenv("ENDPOINT_URL", "")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")

# Initialize Azure OpenAI client with Entra ID authentication
cognitiveServicesResource = os.getenv('AZURE_COGNITIVE_SERVICES_RESOURCE', 'YOUR_COGNITIVE_SERVICES_RESOURCE')
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    f'{cognitiveServicesResource}.default'
)

client = AzureOpenAI(
    azure_endpoint=endpoint,
    azure_ad_token_provider=token_provider,
    api_version='2024-05-01-preview',
)

completion = client.chat.completions.create(
    model=deployment,
    messages=[
    {
        "role": "system",
        "content": '''Você é um analista financeiro especializado no setor de seguros. \n
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
                                - Se não encontrar a informação, diga claramente: "Não há dados disponíveis nos documentos."\n
                                - Organize a resposta em tópicos ou bullets para facilitar entendimento.\n
                                - Contextualize números comparando períodos (ex: variação vs 1T24).'''
    }
],
    past_messages=10,
    max_tokens=6553,
    temperature=0.7,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None,
    extra_body={
        "data_sources": [
            {
                "type": "azure_search",
                "parameters": {
                    "endpoint": os.environ["AZURE_AI_SEARCH_ENDPOINT"],
                    "index_name": os.environ["AZURE_AI_SEARCH_INDEX"],
                    "authentication": {
                        "type": "system_assigned_managed_identity"
                    }
                }
            }
        ]
    }
)

print(completion.model_dump_json(indent=2))
