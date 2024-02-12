import asyncio
import copy
import logging
import time
from operator import itemgetter
from typing import Optional, List, AsyncIterator, TypeVar, Protocol

import yaml
from databricks.vector_search.client import VectorSearchClient
from databricks_genai_inference import ChatCompletion
from dotenv import load_dotenv
from langchain.chat_models import ChatDatabricks
from langchain.embeddings import DatabricksEmbeddings
from langchain.vectorstores.databricks_vector_search import DatabricksVectorSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

import chainlit as cl
from pydantic import BaseModel, ValidationError

load_dotenv()

class ChatConfig(BaseModel):
    prompt_with_history_str: str
    vector_search_endpoint_name: str
    vector_search_index_name: str
    vector_search_text_column: str
    qa_chat_model_str: str = "databricks-llama-2-70b-chat"
    vector_search_rewrite_chat_model_str: str =  "databricks-llama-2-70b-chat"
    prompt_guard_chat_model_str: str = "databricks-llama-2-70b-chat"
    welcome_message_str: str = "Welcome to the Databricks chat bot!"
    prompt_guard_failed_response_str: str = "I am sorry I am not able to answer that question."
    vector_search_index_metadata_columns: Optional[List[str]] = None
    vector_search_embeddings_endpoint_name: str = "databricks-bge-large-en"
    prompt_guard_str: Optional[str] = None


def read_yaml_file(file_path: str) -> ChatConfig:
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    try:
        config = ChatConfig.parse_obj(yaml_data)
        return config
    except ValidationError as e:
        print(f"Error parsing YAML file: {e}")


# Example usage:
yaml_file_path = 'config.yaml'
config: ChatConfig = read_yaml_file(yaml_file_path)


class ChatGuard(ChatDatabricks):
    pass


class RewriteSearch(ChatDatabricks):
    pass


chat_guard_model = ChatGuard(endpoint=config.prompt_guard_chat_model_str, max_tokens=200)
rewrite_search_model = RewriteSearch(endpoint=config.vector_search_rewrite_chat_model_str, max_tokens=200)


prompt_with_history_str = config.prompt_with_history_str

question_with_history_and_context_prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=prompt_with_history_str
)

def truncate_chat_history(input_):
    max_number_words = 750
    chars_in_word = 7
    history = []

    total_words = 0

    for item in reversed(input_):
        content = item.get("content", "")

        words = [content[i:i + chars_in_word] for i in range(0, len(content), chars_in_word)]

        # Check if adding the words exceeds the character limit
        if total_words + len(words) <= max_number_words:
            history.insert(0, {"content": content})
            total_words += len(words)
        else:
            # Break if adding the words exceeds the character limit
            break

    return history


def extract_question(input_) -> str:
    return input_[-1]["content"]


def extract_history(input_) -> str:
    return truncate_chat_history(input_[:-1])


prompt_guard_str = config.prompt_guard_str

prompt_guard = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=prompt_guard_str
)

prompt_guard_chain = (
        {
            "question": itemgetter("messages") | RunnableLambda(extract_question),
            "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
        }
        | prompt_guard
        | chat_guard_model
        | StrOutputParser()
)

class VectorSearchTokenFilter(logging.Filter):
    def __init__(self, module_path):
        super().__init__()
        self.module_path = module_path

    @staticmethod
    def remove_bearer_content(msg):
        import re
        # Define a regex pattern to match 'Bearer ' followed by anything
        pattern = re.compile(r'\'Bearer\s.*\'')

        # Use re.sub to replace the matched pattern with 'Bearer '
        clean_token = re.sub(pattern, '\'Bearer *****\'', msg)

        return clean_token

    @staticmethod
    def remove_query_vector_content(msg):
        import re
        pattern = re.compile(r'\'query_vector\': \[.*\]')
        return re.sub(pattern, '\'query_vector\': [...]', msg)

    def filter(self, record):
        # Check if the record's module matches the specified module_name
        if record.pathname.endswith(self.module_path):
            record.msg = self.remove_bearer_content(record.msg)
            record.msg = self.remove_query_vector_content(record.msg)
        return True


def get_retriever(columns=None):
    custom_filter = VectorSearchTokenFilter("databricks/vector_search/utils.py")
    logging.getLogger().addFilter(custom_filter)
    columns = columns or config.vector_search_index_metadata_columns or []
    # Get the vector search index
    vsc = VectorSearchClient()
    vector_search_endpoint_name = config.vector_search_endpoint_name
    index_name = config.vector_search_index_name
    embedding_model = DatabricksEmbeddings(endpoint=config.vector_search_embeddings_endpoint_name)
    vs_index = vsc.get_index(
        endpoint_name=vector_search_endpoint_name,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column=config.vector_search_text_column, embedding=embedding_model, columns=columns
    )
    return vectorstore.as_retriever(search_kwargs={'k': 3})

retriever = get_retriever()

generate_query_to_retrieve_context_template = """
Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natual language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

Chat history: {chat_history}

Question: {question}
"""

generate_query_to_retrieve_context_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=generate_query_to_retrieve_context_template
)



def format_relevant_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def extract_sources(docs) -> dict:
    def get_title(d):
        return d.metadata.get("title", d.metadata.get("url", None))

    def get_url(d):
        return d.metadata.get("url", None)

    urls = [(get_title(d), get_url(d)) for d in docs]
    if any([url[0] is not None for url in urls]):
        return {
            "markdown_urls": list(set([f'[{url_parts[0]}]({url_parts[1]})' for url_parts in urls]))
        }

    return {
        "markdown_urls": []
    }


def rewrite_question_for_better_search(inp):
    # logging for callback
    resp = f"""**Before Question:** \n{inp["question"]}
\n
**After Rewrite via LLM:** \n{inp["relevant_question"]}
    """
    return resp


relevant_question_chain_prompt = (
        RunnablePassthrough() |
        {
            "question": itemgetter("messages") | RunnableLambda(extract_question),
            "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
        } |
        {
            "relevant_question": generate_query_to_retrieve_context_prompt | rewrite_search_model | StrOutputParser(),
            "chat_history": itemgetter("chat_history"),
            "question": itemgetter("question"),
        }
        |
        {
            "audit": {"question": itemgetter("question"),
                      "relevant_question": itemgetter("relevant_question")} |
                     RunnableLambda(rewrite_question_for_better_search),
            "relevant_question": itemgetter("relevant_question"),
            "chat_history": itemgetter("chat_history"),
            "question": itemgetter("question"),
        }
        |
        {
            "relevant_docs": itemgetter("relevant_question") | retriever,
            "chat_history": itemgetter("chat_history"),
            "question": itemgetter("question"),
        }
        |
        {
            "context": itemgetter("relevant_docs") | RunnableLambda(format_relevant_docs),
            "sources": itemgetter("relevant_docs") | RunnableLambda(extract_sources),
            "chat_history": itemgetter("chat_history"),
            "question": itemgetter("question")
        }
        |
        {
            "prompt": question_with_history_and_context_prompt,
            "sources": itemgetter("sources"),
        }
)


class HasMessage(Protocol):
    message: str


T = TypeVar('T', bound=HasMessage)


class AsyncGeneratorWrapper(AsyncIterator[T]):
    def __init__(self, gen):
        self.gen = gen

    def __aiter__(self) -> 'AsyncGeneratorWrapper[T]':
        return self

    async def __anext__(self) -> T:
        try:
            # Use asyncio to yield control and create asynchronous behavior
            await asyncio.sleep(0)
            return next(self.gen)
        except StopIteration:
            raise StopAsyncIteration


def run_chat_completion(msgs) -> AsyncGeneratorWrapper[HasMessage]:
    chat_model = config.qa_chat_model_str.replace("databricks-", "")
    resp = ChatCompletion.create(model=chat_model,
                                 messages=[{"role": "system", "content": "You are a helpful assistant."},
                                           *msgs],
                                 temperature=0.1,
                                 stream=True, )
    return AsyncGeneratorWrapper(resp)


def get_history(input_):
    history = cl.user_session.get("history")
    if history is None:
        history = []
    history_copy = copy.deepcopy(history)
    history_copy.append({"role": "user", "content": f"{input_}"})
    return {
        "messages": history_copy
    }

@cl.step(
    name="ChatBot: Prompt Guard",
)
async def log_prompt_guard(history, resp):
    messages = history["messages"]
    return {
        "question": extract_question(messages),
        "chat_history": extract_history(messages),
        "prompt_guard_response": resp
    }


@cl.step(
    name="ChatBot: Final Generation Prompt",
)
async def final_prompt(content: str):
    return content


class ChatBot:

    @staticmethod
    async def _aguard(content, cfg=None) -> str | None:
        if config.prompt_guard_str is None:
            return None
        start = time.time()
        is_valid_question = await prompt_guard_chain.ainvoke(content, cfg)
        is_valid_question = is_valid_question.strip()
        print(f"[INFO] Time to compute guard: {time.time() - start}")
        print(f"[INFO] Is valid prompt: {content}; Guard Response: {is_valid_question}")
        await log_prompt_guard(content, is_valid_question)
        if is_valid_question.lower().startswith("no"):
            print(f"[INFO] Invalid Question: Prompt: {content} Guard Response: {is_valid_question}")
            return "I am sorry I am not able to answer that question." + \
                " Please feel free to ask me questions about the various data, " + \
                "technology and service partners in the Databricks partner ecosystem."
        return None

    def intro_message(self) -> cl.Message | None:
        return cl.Message(config.welcome_message_str)


    async def complete(self, content: str, input_message, response):

        await response.send()
        history = get_history(content)
        guard_resp = await self._aguard(history)
        if config.prompt_guard_str is not None and guard_resp is not None:
            # short circuit
            await response.stream_token(config.prompt_guard_failed_response_str)
            return guard_resp

        processed_context = await cl.make_async(relevant_question_chain_prompt.invoke)(history, {
            "callbacks": [cl.LangchainCallbackHandler()]
        })
        processed_prompt = processed_context["prompt"]
        msgs = [{"content": msg.content, "role": "user"} for msg in processed_prompt.to_messages()]

        final_prompt_content = "\n".join([msg.content for msg in processed_prompt.to_messages()])
        await final_prompt(final_prompt_content)

        buff = []
        token_stream = await cl.make_async(run_chat_completion)(msgs)
        async for token_chunk in token_stream:
            chunk: HasMessage = token_chunk  # noqa token_chunk is a ChatCompletionChunkObject not Future
            buff.append(chunk.message)
            await response.stream_token(chunk.message)

        sources = processed_context["sources"]["markdown_urls"]
        sources_text = ""
        if len(sources) > 0:
            sources_text += "\n\nSources: \n\n* " + "\n* ".join(sources)
        await response.stream_token(sources_text)

        result = "".join(buff)
        return result


@cl.on_chat_start
async def chat_init():
    completion = ChatBot()
    cl.user_session.set("completion", completion)
    await completion.intro_message().send()


def update_chat_history(message: cl.Message, response: cl.Message):
    history = cl.user_session.get("history")
    if history is None:
        history = []
    history.append({"role": "user", "content": f"{message.content}"})
    history.append({"role": "assistant", "content": f"{response.content}"})
    cl.user_session.set("history", history)



@cl.on_message
async def main(message: cl.Message):
    completion: ChatBot = cl.user_session.get("completion")

    resp_msg = cl.Message(content="")
    await completion.complete(message.content, message, resp_msg)
    await resp_msg.update()
    update_chat_history(message, resp_msg)
