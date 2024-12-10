#! /usr/bin/env python3
from openai import OpenAI
 
client = OpenAI(base_url="https://ollama.fit.fraunhofer.de/v1",
                api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjE4ZWYyY2FiLWZiZmEtNDc2Yi1iYzJjLTk0Nzg2Njk2ZWQwYyJ9.3digMEcl2-OFykdxfVovKOx7-QYYsVMQS8ecVCz2dbo") # Den generierten API-Key verwenden
 


myModel = "teuken0.4-instruct-research:7b"
myModel = "mixtral:latest" # Ein verfügbares Modell aus Ollama wählen
#myModel = "mistral:7b" # Ein verfügbares Modell aus Ollama wählen


response = client.chat.completions.create(
    model=myModel, # Ein verfügbares Modell aus Ollama wählen
    messages=[
        {"role": "system", "content": "Du bist ein hilfreicher Assitent."},
        {"role": "user", "content": "Was ist eine Blockchain und was ist ein Proof of Work?"},
    ]
)

print("=" * len(f"Model: {myModel}")) 
print(f"Model: {myModel}")
print("=" * len(f"Model: {myModel}")) 
print(response.choices[0].message.content)

