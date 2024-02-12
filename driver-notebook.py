# Databricks notebook source
# MAGIC %pip install -r requirements.txt
# MAGIC %pip install dbtunnel[chainlit,asgiproxy]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os

current_directory = os.getcwd()
script_path = current_directory + "/chatbot.py"

# COMMAND ----------

from dbtunnel import dbtunnel

# COMMAND ----------

dbtunnel.chainlit(script_path).inject_auth().run()

# COMMAND ----------
