# import json
# from langchain_openai import ChatOpenAI

# if __name__ == "__main__":

#     llm = ChatOpenAI(model="gpt-4.1", temperature=0)

#     with open("pages_base64.json", "r") as f:
#         pdf_page = json.load(f)[59]

#     print(
#         llm.invoke(
#             [
#                 # {
#                 # }
#                 {
#                     "type": "text",
#                     "text": "me fale o que esta escrito nessa pagina. o titulo. Eu tenho problema de visao. Me de o subtitulo tambem",
#                 },
#                 {
#                     "type": "image",
#                     "source_type": "base64",
#                     "data": pdf_page,
#                     "mime_type": "application/pdf",
#                 },
#             ]
#         )
#     )

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    # pega a página 60 (índice 59) já em base64 PNG
    with open("pages_base64.json", "r") as f:
        pdf_page_b64 = json.load(f)[59]

    # monta mensagem multimodal correta
    msg = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Me diga o título e subtítulo desta página. Tenho baixa visão. Sobre o que eh esse conteiudo. Voce pode me explicar de uma formam bem simples?",
            },
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{pdf_page_b64}"}},
        ]
    )

    resp = llm.invoke([msg])
    print(resp.content)
