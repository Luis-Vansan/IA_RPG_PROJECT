from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import json, os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
Você é o mestre de um RPG de fantasia medieval.
Narre as consequências das ações do jogador de forma
imersiva, justa e divertida. Leve em conta os atributos
e inventário do personagem ao decidir os resultados.

Regras importantes:
- Combates têm chance de falha baseada na situação
- Itens podem ser encontrados, comprados ou perdidos
- A vida nunca passa do valor de vida_max
- Se a vida chegar a 0, retorne fim_de_jogo como true

Responda SEMPRE em JSON válido, sem nenhum texto fora do JSON:
{
  "narracao": "texto narrativo da cena (2 a 4 parágrafos)",
  "estado_atualizado": {
    "jogador": {
      "nome": "string",
      "classe": "string",
      "vida": 0,
      "vida_max": 0,
      "inventario": [],
      "ouro": 0
    },
    "local_atual": "string",
    "missao_atual": "string"
  },
  "opcoes_sugeridas": ["ação 1", "ação 2", "ação 3"],
  "fim_de_jogo": false
}
"""

class AcaoRequest(BaseModel):
    acao: str
    estado: dict

@app.post("/agir")
async def agir(req: AcaoRequest):
    mensagem = f"""
Estado atual do jogo:
{json.dumps(req.estado, ensure_ascii=False, indent=2)}

Ação do jogador: {req.acao}
"""

    try:
        resposta = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": mensagem}
            ],
            temperature=0.8,
            max_tokens=1024,
        )

        texto = resposta.choices[0].message.content.strip()

        # Remove blocos de código se o modelo retornar ```json
        if texto.startswith("```"):
            texto = texto.split("```")[1]
            if texto.startswith("json"):
                texto = texto[4:]
            texto = texto.strip()

        return json.loads(texto)

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="A IA retornou um formato inválido. Tente novamente.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "online", "modelo": "llama-3.3-70b-versatile"}