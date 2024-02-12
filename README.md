# Composable Chainlit Chat Demo

## Usage

1. clone the repo
2. Use a single user cluster
3. modify config file
4. Only the following fields are required in `config.yaml`
    * prompt_with_history_str: str
    * vector_search_endpoint_name: str
    * vector_search_index_name: str
    * vector_search_text_column: str
5. Run the driver-notebook

Features: 

* supports Q&A with Sources
* streaming with FM apis
* example of guarding prompts to avoid offtopic questions
* configure and run, convenient for demos
* Optional fields for customizing experience

  ```python
  from pydantic import BaseModel

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
  ```

