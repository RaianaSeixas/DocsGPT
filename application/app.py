import os
import json
import traceback

import dotenv
import requests
from flask import Flask, request, render_template
from langchain import FAISS
from langchain import OpenAI, VectorDBQA, HuggingFaceHub, Cohere
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceHubEmbeddings, CohereEmbeddings, HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from error import bad_request
# os.environ["LANGCHAIN_HANDLER"] = "langchain"

if os.getenv("LLM_NAME") is not None:
    llm_choice = os.getenv("LLM_NAME")
else:
    llm_choice = "openai"

if os.getenv("EMBEDDINGS_NAME") is not None:
    embeddings_choice = os.getenv("EMBEDDINGS_NAME")
else:
    embeddings_choice = "openai_text-embedding-ada-002"



if llm_choice == "manifest":
    from manifest import Manifest
    from langchain.llms.manifest import ManifestWrapper

    manifest = Manifest(
        client_name="huggingface",
        client_connection="http://127.0.0.1:5000"
    )

# Redirect PosixPath to WindowsPath on Windows
import platform

if platform.system() == "Windows":
    import pathlib

    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

# loading the .env file
dotenv.load_dotenv()

with open("combine_prompt.txt", "r") as f:
    template = f.read()

with open("combine_prompt_hist.txt", "r") as f:
    template_hist = f.read()

if os.getenv("API_KEY") is not None:
    api_key_set = True
else:
    api_key_set = False
if os.getenv("EMBEDDINGS_KEY") is not None:
    embeddings_key_set = True
else:
    embeddings_key_set = False

app = Flask(__name__)

''' Esta é a rota principal. Quando o usuário visita a raiz do site, a função home() é chamada. 
Esta função usa a função render_template() do Flask para renderizar um template HTML chamado "index.html". 
Ela também passa três variáveis, api_key_set, llm_choice (modelo llm) e embeddings_choice, que serão usadas pelo template para determinar como renderizar a página.'''
@app.route("/") #rota principal / 
def home():
    return render_template("index.html", api_key_set=api_key_set, llm_choice=llm_choice,
                           embeddings_choice=embeddings_choice)

'''Esta é a rota /api/answer. Quando o usuário envia uma solicitação POST para /api/answer, a função api_answer() é chamada. 
Ela começa obtendo os dados da solicitação POST usando a função get_json() do objeto request do Flask. 
Essa função retorna um dicionário Python com os dados enviados na solicitação POST.'''
@app.route("/api/answer", methods=["POST"])
def api_answer():
    data = request.get_json()
    question = data["question"]
    history = data["history"]
    print('-'*5)
    if not api_key_set:
        api_key = data["api_key"]
    else:
        api_key = os.getenv("API_KEY")
    if not embeddings_key_set:
        embeddings_key = data["embeddings_key"]
    else:
        embeddings_key = os.getenv("EMBEDDINGS_KEY")

    # use try and except  to check for exception
    ''' Essa parte do código usa uma declaração try e except para tratar exceções que possam ocorrer no código. 
    A seguir, ele verifica se há um documento específico para ser usado como base para a vetorização de palavras. Se houver, ele define o caminho para o arquivo de vetorização de palavras. 
    Caso contrário, ele define um caminho vazio.'''
    try:
        # check if the vectorstore is set
        if "active_docs" in data:
            vectorstore = "vectors/" + data["active_docs"]
            if data['active_docs'] == "default":
                vectorstore = ""
        else:
            vectorstore = ""

        # loading the index and the store and the prompt template
        # Note if you have used other embeddings than OpenAI, you need to change the embeddings
        '''seleciona a API de embedding a ser usada com base na opção selecionada pelo usuário.
        '''
        if embeddings_choice == "openai_text-embedding-ada-002":
            docsearch = FAISS.load_local(vectorstore, OpenAIEmbeddings(openai_api_key=embeddings_key))
        elif embeddings_choice == "huggingface_sentence-transformers/all-mpnet-base-v2":
            docsearch = FAISS.load_local(vectorstore, HuggingFaceHubEmbeddings())
        elif embeddings_choice == "huggingface_hkunlp/instructor-large":
            docsearch = FAISS.load_local(vectorstore, HuggingFaceInstructEmbeddings())
        elif embeddings_choice == "cohere_medium":
            docsearch = FAISS.load_local(vectorstore, CohereEmbeddings(cohere_api_key=embeddings_key))

        # create a prompt template
        '''Verifica-se se existe um histórico de conversas. Caso exista, o histórico é convertido de JSON para um objeto Python através do método json.loads().
        Em seguida, é gerado um prompt personalizado para o chatbot utilizando o histórico de conversas através do template template_hist.
        Esse template contém as variáveis {historyquestion} e {historyanswer} que são substituídas pelas perguntas e respostas do histórico. 
        Se não existir um histórico, é gerado um prompt padrão utilizando o template template. Os prompts são gerados utilizando a biblioteca jinja2'''
        if history:
            history = json.loads(history)
            template_temp = template_hist.replace("{historyquestion}", history[0]).replace("{historyanswer}", history[1])
            c_prompt = PromptTemplate(input_variables=["summaries", "question"], template=template_temp, template_format="jinja2")
        else:
            c_prompt = PromptTemplate(input_variables=["summaries", "question"], template=template, template_format="jinja2")

        '''llm_choice difine o modelo a ser usado. Posso definir aqui hyperparâmetros dos modelos (ex: temperatura e max tokens)'''    

        if llm_choice == "openai":
            llm = OpenAI(openai_api_key=api_key, temperature=0)
        elif llm_choice == "manifest":
            llm = ManifestWrapper(client=manifest, llm_kwargs={"temperature": 0.001, "max_tokens": 2048})
        elif llm_choice == "huggingface":
            llm = HuggingFaceHub(repo_id="bigscience/bloom", huggingfacehub_api_token=api_key)
        elif llm_choice == "cohere":
            llm = Cohere(model="command-xlarge-nightly", cohere_api_key=api_key)
        
        '''A função load_qa_chain() é utilizada para criar uma cadeia de perguntas e respostas. 
        A variável llm é o modelo de NPL utilizado na cadeia e a variável c_prompt é o prompt utilizado para gerar as perguntas. 
        O parâmetro chain_type define o tipo de cadeia a ser criada, nesse caso, map_reduce'''

        qa_chain = load_qa_chain(llm=llm, chain_type="map_reduce",
                                combine_prompt=c_prompt)
        
        '''é criada uma instância da classe VectorDBQA que é responsável por encontrar a melhor resposta para a pergunta 
        através de uma busca vetorial em um conjunto de documentos. 
        A variável combine_documents_chain contém a cadeia de processamento criada anteriormente pela função load_qa_chain(). 
        A variável vectorstore contém o conjunto de documentos a serem pesquisados. 
        O parâmetro k define o número de respostas a serem retornadas.'''
        chain = VectorDBQA(combine_documents_chain=qa_chain, vectorstore=docsearch, k=4)

        '''a resposta é atribuída à chave 'answer' do dicionário result. Em seguida, ela é formatada para ser exibida na interface do usuário. 
        As quebras de linha \n são substituídas por <br>, para que a resposta seja exibida com quebras de linha na página da web. 
        Por fim, é feita uma tentativa de remover a seção "SOURCES:" da resposta, caso ela exista.'''
        # fetch the answer
        result = chain({"query": question})
        print(result)

        # some formatting for the frontend
        result['answer'] = result['result']
        result['answer'] = result['answer'].replace("\\n", "<br>")
        try:
            result['answer'] = result['answer'].split("SOURCES:")[0]
        except:
            pass

        # mock result
        # result = {
        #     "answer": "The answer is 42",
        #     "sources": ["https://en.wikipedia.org/wiki/42_(number)", "https://en.wikipedia.org/wiki/42_(number)"]
        # }
        return result
    
    
    except Exception as e:
        # print whole traceback
        traceback.print_exc()
        print(str(e))
        return bad_request(500,str(e))
'''Se alguma exceção for levantada, o traceback (rastreamento) da exceção é impresso no console, junto com a mensagem de erro. 
    Em seguida, uma resposta HTTP é re,0tornada com um status 500 (erro interno do servidor) e a mensagem de erro como corpo da resposta. 
    Isso garante que o usuário receba uma resposta apropriada e informativa caso ocorra algum erro inesperado durante a execução do chatbot.'''


'''Nesta parte do código, é definida uma rota /api/docs_check para verificar se um conjunto de documentos está disponível no formato vectorstore. 
A rota recebe uma solicitação HTTP POST contendo um objeto JSON com informações sobre o conjunto de documentos a ser verificado. 
O nome do diretório que contém o vetor de documentos é obtido a partir do objeto JSON e armazenado na variável vectorstore.

Se o diretório vectorstore existir ou se o conjunto de documentos for o padrão, a rota retorna um objeto JSON com uma chave "status" definida como "exists", 
indicando que o conjunto de documentos já existe. Se o diretório vectorstore não existir, a rota tentará baixar o índice do vetor de documentos da URL especificada
e salvá-lo no diretório vectorstore. Se o download for bem-sucedido, a rota criará o diretório vectorstore 
se ele ainda não existir e salvará o índice do vetor de documentos no diretório. Em seguida, a rota tentará baixar o arquivo de armazenamento do vetor de documentos 
e salvá-lo no diretório vectorstore. Se o download for bem-sucedido, a rota salvará o arquivo no diretório. 
Por fim, a rota retornará um objeto JSON com uma chave "status" definida como "loaded", indicando que o conjunto de documentos foi carregado com sucesso.
Se ocorrer algum erro durante o processo de download ou salvamento dos arquivos, a rota retornará um objeto JSON 
com uma chave "status" definida como "null", indicando que o conjunto de documentos não pôde ser carregado.'''
@app.route("/api/docs_check", methods=["POST"])
def check_docs():
    # check if docs exist in a vectorstore folder
    data = request.get_json()
    vectorstore = "vectors/" + data["docs"]
    base_path = 'https://raw.githubusercontent.com/arc53/DocsHUB/main/'
    if os.path.exists(vectorstore) or data["docs"] == "default":
        return {"status": 'exists'}
    else:
        r = requests.get(base_path + vectorstore + "index.faiss")

        if r.status_code != 200:
            return {"status": 'null'}
        else:
            if not os.path.exists(vectorstore):
                os.makedirs(vectorstore)
            with open(vectorstore + "index.faiss", "wb") as f:
                f.write(r.content)

            # download the store
            r = requests.get(base_path + vectorstore + "index.pkl")
            with open(vectorstore + "index.pkl", "wb") as f:
                f.write(r.content)

        return {"status": 'loaded'}


# handling CORS
'''o aplicativo Flask está configurado para adicionar os cabeçalhos CORS (Cross-Origin Resource Sharing) a cada resposta HTTP, 
permitindo que as solicitações AJAX feitas por um navegador da web acessem recursos em outro domínio.
A função after_request é decorada com @app.after_request, o que significa que ela é executada depois que uma resposta é retornada 
por qualquer rota da aplicação. Essa função adiciona os cabeçalhos CORS necessários à resposta HTTP.
Por fim, o servidor é iniciado usando o método app.run(), que começa a executar o servidor Flask na porta 5001 com a opção debug=True, 
permitindo que mensagens de depuração sejam exibidas no console do servidor.'''
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


if __name__ == "__main__":
    app.run(debug=True, port=5001)
