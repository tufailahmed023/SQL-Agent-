from langchain_google_genai import GoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX 
from few_shots import few_shots 

import os 
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')
load_dotenv()

class sql_agent:
    def __init__(self) -> None:
        self.__api = os.environ.get("GOOGLE_API_KEY")
        self.db_user = os.environ.get("DATABASE_USER")
        self.__db_password = os.environ.get('DATABASE_PASSWORD')
        self.db_host = os.environ.get('DATABASE_HOST')
        self.db_name = "emp_info"
        self.db_port = '3306'
    
    def database_con(self):
        try:
            db = SQLDatabase.from_uri(f"mysql+pymysql://{self.db_user}:{self.__db_password}@{self.db_host}:{self.db_port}/{self.db_name}", sample_rows_in_table_info=3)
            return db 
        except Exception as e :
            return e 
    
    def llm(self,):
        try:
            llm = GoogleGenerativeAI(model="models/text-bison-001",google_api_key=self.__api,temperature=0.3)
            return llm 
        except Exception as e :
            print(e)
    
    def embedding(self,):
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        to_vectorize = [" ".join(str(example.values())) for example in few_shots]
        vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)
        return vectorstore 
    
    def mysql_prompt(self,):

        mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
                        Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
                        Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
                        Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
                        Pay attention to use CURDATE() function to get the current date, if the question involves "today".

                        Use the following format:

                        Question: Question here
                        SQLQuery: Query to run with no pre-amble
                        SQLResult: Result of the SQLQuery
                        Answer: Final answer here

                        No pre-amble."""
        
        example_prompt = PromptTemplate(input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
                                        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}")
        return (mysql_prompt,example_prompt)
    
    
    def chain(self,):

        llm = self.llm()
        db = self.database_con()
        vectorstore = self.embedding()
        example_selector = SemanticSimilarityExampleSelector(vectorstore=vectorstore,k=2)
        mysql_prompt,example_prompt = self.mysql_prompt()
        few_shot_prompt = FewShotPromptTemplate(example_selector=example_selector,example_prompt=example_prompt,
                                                prefix=mysql_prompt,suffix=PROMPT_SUFFIX,
                                                input_variables=["input", "table_info", "top_k"] )
        
        sql_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt,return_direct = True)

        return sql_chain
    
    def start_app(self,query):
        query_chain = self.chain()
        return query_chain(query)
    
# obj = sql_agent()
# obj.start_app("which all employee have a rating of 3 or more ?")