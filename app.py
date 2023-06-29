# Standard library imports
import os
from datetime import datetime
from io import StringIO
import logging
import sys
import subprocess
import tempfile
import traceback

# Third-party library imports
import streamlit as st
from streamlit.components.v1 import html
from apikey import apikey 
import gpt4all
import openai

# Langchain specific imports
from langchain.llms import OpenAI, GPT4All, OpenAI as LangChainOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 
from langchain.tools.python.tool import PythonREPLTool

# from langchain.agents import SequentialChain, PromptTemplate, ConversationBufferMemory, LLMChain, initialize_agent, AgentType

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
import langchain.agents as lc_agents


# Set APIkey for OpenAI Service
if apikey:
    os.environ['OPENAI_API_KEY'] = apikey

PATH = 'D:/Projects/LLM/GPT4/Models/ggml-model-q4_0.bin'



def start_gpt4():
    PATH = 'D:/Projects/LLM/GPT4/Models/ggml-gpt4all-j-v1.3-groovy.bin'
    prompt = st.text_input('Enter your prompt here!')
    llm = gpt4all.GPT4All(PATH, model_type='gptj')
    if prompt: 
        message = [{"role": "user", "content": prompt}]
        #response = agent_executor.run(prompt)
        response = llm.chat_completion(message)['choices'][0]['message']['content']
        st.write(response)
        
        
def open_ai_gpt():
    llm = OpenAI(temperature=0.5)
    prompt = st.text_input('Plug in your prompt here!')
    template = PromptTemplate(input_variables=['action'], template="""
            As a creative agent, {action}
    """)
    template = PromptTemplate(input_variables=['action'], template="""
            ### Instruction: 
            The prompt below is a question to answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response.
            ### Prompt: 
            {action}
            ### Response:""")
    chain = LLMChain(llm=llm, prompt=template, verbose=True) 
    
    # if we hit enter  
    if prompt:
        # Pass the prompt to the LLM Chain
        response = chain.run(prompt) 
        # do this
        st.write(response)  

class PythonREPL:
    # Define the initialization method
    def __init__(self):
        pass

    # Define the run method
    def run(self, command: str) -> str:
        # Store the current value of sys.stdout
        old_stdout = sys.stdout
        # Create a new StringIO object
        sys.stdout = mystdout = StringIO()
        # Try to execute the code
        try:
            # Execute the code
            exec(command, globals())
            sys.stdout = old_stdout
            output = mystdout.getvalue()
        # If an error occurs, print the error message
        except Exception as e:
            # Restore the original value of sys.stdout
            sys.stdout = old_stdout
            # Get the error message
            output = str(e)
        return output

def read_fin():
    
    def save_uploaded_file(uploaded_file):
        with open(os.path.join('saved', "annualreport.pdf"), "wb") as f:
         f.write(uploaded_file.getvalue())
    
    st.title("GPT Banker Portfolio")
    
    file = st.file_uploader("Upload Your Report", type="pdf")
    if file:
        save_uploaded_file(file)
    # Create a text input box for the user
    prompt = st.text_input('Input your prompt here')
    # Create instance of OpenAI LLM
    llm = OpenAI(temperature=0.9, verbose=True)
    embeddings = OpenAIEmbeddings()
    # Create and load PDF Loader
    loader = PyPDFLoader('annualreport.pdf')
    # Split pages from pdf 
    pages = loader.load_and_split()
    # Load documents into vector database aka ChromaDB
    store = Chroma.from_documents(pages, embeddings, collection_name='annualreport')

    # Create vectorstore info object - metadata repo?
    vectorstore_info = VectorStoreInfo(
        name="annual_report",
        description="a banking annual report as a pdf",
        vectorstore=store
    )
    # Convert the document store into a langchain toolkit
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

    # Add the toolkit to an end-to-end LC
    agent_executor = create_vectorstore_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )
    # If the user hits enter
    if prompt:
        # Then pass the prompt to the LLM
        response = agent_executor.run(prompt)
        # ...and write it out to the screen
        st.write(response)

        # With a streamlit expander  
        with st.expander('Document Similarity Search'):
            # Find the relevant pages
            search = store.similarity_search_with_score(prompt) 
            # Write out the first 
            st.write(search[0][0].page_content) 

def wiz():

    prompt = st.text_input('Key in your topic here') 

    # Prompt templates
    title_template = PromptTemplate(
        input_variables = ['topic'], 
        template='you are an elite content creator, write me a youtube video or tiktok title about {topic}, do not longer than 40 words'
    )

    script_template = PromptTemplate(
        input_variables = ['title', 'wikipedia_research'], 
        template='you are a content creator, now write me a youtube video or tiktok script based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research}, do not longer than 350 words'
    )
    if prompt: 
        # Memory 
        title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
        script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

        PATH = "D:/Projects/LLM/GPT4/Models/ggml-model-q4_0.bin"

        # Llms
        #llm = OpenAI(temperature=0.9) 
        llm = GPT4All(model = PATH, n_ctx=2048, n_threads=8)
        title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
        script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

        wiki = WikipediaAPIWrapper()

        # Show stuff to the screen if there's a prompt

        title = title_chain.run(prompt)
        wiki_research = wiki.run(prompt) 
        wiki_research_tr = wiki_research.split('\n', 1)[1]
        script = script_chain.run(title=title, wikipedia_research=wiki_research_tr)

        st.write(title) 
        st.write(script) 

        with st.expander('Title History'): 
            st.info(title_memory.buffer)

        with st.expander('Script History'): 
            st.info(script_memory.buffer)

        with st.expander('Wikipedia Research'): 
            st.info(wiki_research)
            
            
            
def openwiz():

    prompt = st.text_input('Key in your topic here') 

    # Prompt templates
    title_template = PromptTemplate(
        input_variables = ['topic'], 
        template='you are an elite content creator, write me a youtube video title or tiktok about {topic}, do not longer than 40 words'
    )

    script_template = PromptTemplate(
        input_variables = ['title', 'wikipedia_research'], 
        template='you are a content creator, now write me a youtube video or tiktok script based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research}, do not longer than 350 words'
    )
    if prompt: 
        # Memory 
        title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
        script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

        # Llms
        llm = OpenAI(temperature=0.9) 
        title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
        script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

        wiki = WikipediaAPIWrapper()

        # Show stuff to the screen if there's a prompt

        title = title_chain.run(prompt)
        wiki_research = wiki.run(prompt) 
        wiki_research_tr = wiki_research.split('\n', 1)[1]
        script = script_chain.run(title=title, wikipedia_research=wiki_research_tr)

        st.write(title) 
        st.write(script) 

        with st.expander('Title History'): 
            st.info(title_memory.buffer)

        with st.expander('Script History'): 
            st.info(script_memory.buffer)

        with st.expander('Wikipedia Research'): 
            st.info(wiki_research)
        

st.title('🔗 My GPT4')

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", ["Chat", "Wizard", "Read Finance", "Code"])

if selection == "Chat":
    st.header("Chat with your Local GPT4!")
    if apikey:
        open_ai_gpt()
    else:
        start_gpt4()

elif selection == "Wizard":
    st.header("🦜🔗Wizard Content Generator!")
    if apikey:
        open_ai_gpt()
    else:
        wiz()

elif selection == "Read Finance":
    st.header("Let me help you to read your Portfolio!")
    if apikey:
        read_fin()
    else:
        st.info("No api keys detected!")

elif selection == "Code":
    
    # Define Constants
    LANGUAGE_CODES = {
    'C': 'c',
    'C++': 'c++',
    'Java': 'java',
    'Ruby': 'ruby',
    'Scala': 'scala',
    'C#': 'csharp',
    'Objective C': 'objc',
    'Swift': 'swift',
    'JavaScript': 'nodejs',
    'Kotlin': 'kotlin',
    'Python': 'python3',
    'GO Lang': 'go',
}

    # Define App Headers
    st.title("LangChain Coder - AI 🦜🔗")
    st.header("Code Interpreter")

    # Define User Inputs
    code_prompt = st.text_input("Enter a prompt to generate the code")
    code_language = st.selectbox("Select a language", list(LANGUAGE_CODES.keys()))
    compiler_mode = st.radio("Compiler Mode", ("Online", "Offline"))

    # Define Buttons
    button_generate = st.button("Generate Code")
    button_run = st.button("Run Code")

    # Define Code Chains and Memory
    code_template = PromptTemplate(
        input_variables=['code_topic'],
        template='Write me code in ' + f'{code_language} language' + ' for {code_topic}'
    )

    code_fix_template = PromptTemplate(
        input_variables=['code_topic'],
        template='Fix any error in the following code in ' + f'{code_language} language' + ' for {code_topic}, only give me the raw code'
    )

    memory = ConversationBufferMemory(
        input_key='code_topic', memory_key='chat_history')

    open_ai_llm = OpenAI(temperature=0.7, max_tokens=1000)

    code_chain = LLMChain(llm=open_ai_llm, prompt=code_template,
                        output_key='code', memory=memory, verbose=True)

    code_fix_chain = LLMChain(llm=open_ai_llm, prompt=code_fix_template,
                            output_key='code_fix', memory=memory, verbose=True)

    sequential_chain = SequentialChain(chains=[code_chain, code_fix_chain], input_variables=[
        'code_topic'], output_variables=['code', 'code_fix'])


    # Define Other Functions
    def generate_dynamic_html(language, code_prompt):
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Python App with JavaScript</title>
        </head>
        <body>
            <div data-pym-src='https://www.jdoodle.com/plugin' data-language="{language}"
                data-version-index="0" data-libs="">
                {script_code}
            </div>
            <script src="https://www.jdoodle.com/assets/jdoodle-pym.min.js" type="text/javascript"></script>
        </body>
        </html>
        """.format(language=LANGUAGE_CODES[language], script_code=code_prompt)


    def setup_logging(log_file):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%H:%M:%S",
            filename=log_file,  
            filemode='a',  
        )


    def run_query(query, model_kwargs, max_iterations):
        llm = LangChainOpenAI(**model_kwargs)
        python_repl = lc_agents.Tool("Python REPL", PythonREPL().run,
                                    "A Python shell. Use this to execute python commands.")
        tools = [python_repl]
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                model_kwargs=model_kwargs, verbose=True, max_iterations=max_iterations)
        return agent.run(query)


    def run_code(code, language):
        if language == "Python":
            output = subprocess.run(
                ["python", "-c", code], capture_output=True, text=True)
        elif language in ["C", "C++"]:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".c" if language == "C" else ".cpp", delete=True) as f:
                f.write(code)
                f.flush()
                output = subprocess.run(
                    ["gcc" if language == "C" else "g++", "-o", f.name + ".out", f.name], capture_output=True, text=True)
                if output.returncode == 0:
                    output = subprocess.run(
                        [f.name + ".out"], capture_output=True, text=True)
        elif language == "JavaScript":
            output = subprocess.run(
                ["node", "-e", code], capture_output=True, text=True)
        else:
            output = subprocess.CompletedProcess(None, 1, "Unsupported language.")
        return output


    def generate_code():
        try:
            st.session_state.generated_code = code_chain.run(code_prompt)
            st.session_state.code_language = code_language
            st.code(st.session_state.generated_code,
                    language=st.session_state.code_language.lower())
            st.expander('Message History').info(memory.buffer)
        except Exception:
            st.error('Error in code generation.')
            st.error(traceback.format_exc())


    def execute_code(compiler_mode):
        try:
            if compiler_mode == "online":
                html_template = generate_dynamic_html(
                    st.session_state.code_language, st.session_state.generated_code)
                html(html_template, width=720, height=800, scrolling=True)
            else:
                output = run_code(st.session_state.generated_code,
                                st.session_state.code_language)
                if "error" in output.stderr.lower():
                    response = sequential_chain(
                        {'code_topic': st.session_state.generated_code})
                    st.session_state.generated_code = response['code_fix']
                    st.expander('Message History').info(memory.buffer)
                    output = run_code(st.session_state.generated_code,
                                    st.session_state.code_language)
                st.code(st.session_state.generated_code,
                        language=st.session_state.code_language.lower())
                st.write("Execution Output:")
                st.write("Return Code: " + str(output.returncode))
                st.write("Standard Out: " + str(output.stdout))
        except Exception:
            st.error('Error in code execution.')
            st.error(traceback.format_exc())


    # Initialize Session State
    if "generated_code" not in st.session_state:
        st.session_state.generated_code = ""
    if "code_language" not in st.session_state:
        st.session_state.code_language = ""

    # Handle Button Clicks
    if button_generate and code_prompt:
        generate_code()
    if button_run and code_prompt:
        execute_code(compiler_mode.lower())
        
    
    
    # st.header("Paraphrase and Summarize")
    # global generated_code

    # LANGUAGE_CODES = {
    #     'C': 'c',
    #     'C++': 'c++',
    #     'Java': 'java',
    #     'Ruby': 'ruby',
    #     'Scala': 'scala',
    #     'C#': 'csharp',
    #     'Objective C': 'objc',
    #     'Swift': 'swift',
    #     'JavaScript': 'nodejs',
    #     'Kotlin': 'kotlin',
    #     'Python': 'python3',
    #     'GO Lang': 'go',
    # }

    # # App title and description
    # st.title("LangChain Coder - AI 🦜🔗")
    # code_prompt = st.text_input("Enter a prompt to generate the code")
    # code_language = st.selectbox("Select a language", list(LANGUAGE_CODES.keys()))

    # # Generate and Run Buttons
    # button_generate = st.button("Generate Code")
    # # code_file = st.text_input("Enter a file name:")
    # # button_save = st.button("Save Code")

    # compiler_mode = st.radio("Compiler Mode", ("Online", "Offline"))
    # button_run = st.button("Run Code")

    # # Prompt Templates
    # code_template = PromptTemplate(
    #     input_variables=['code_topic'],
    #     template='Write me code in ' +
    #     f'{code_language} language' + ' for {code_topic}'
    # )

    # code_fix_template = PromptTemplate(
    #     input_variables=['code_topic'],
    #     template='Fix any error in the following code in ' +
    #     f'{code_language} language' + ' for {code_topic}'
    # )

    # # Memory for the conversation
    # memory = ConversationBufferMemory(
    #     input_key='code_topic', memory_key='chat_history')

    # # LLM Chains definition
    # # Create an OpenAI LLM model
    # open_ai_llm = OpenAI(temperature=0.7, max_tokens=1000)

    # # Create a chain that generates the code
    # code_chain = LLMChain(llm=open_ai_llm, prompt=code_template,
    #                     output_key='code', memory=memory, verbose=True)

    # # Create a chain that fixes the code
    # code_fix_chain = LLMChain(llm=open_ai_llm, prompt=code_fix_template,
    #                         output_key='code_fix', memory=memory, verbose=True)

    # # Create a sequential chain that combines the two chains above
    # sequential_chain = SequentialChain(chains=[code_chain, code_fix_chain], input_variables=[
    #                                 'code_topic'], output_variables=['code', 'code_fix'])


    # # Generate Dynamic HTML for JDoodle Compiler iFrame Embedding.
    # def generate_dynamic_html(language, code_prompt):
    #     logger = logging.getLogger(__name__)
    #     logger.info("Generating dynamic HTML for language: %s", language)
    #     html_template = """
    #     <!DOCTYPE html>
    #     <html lang="en">
    #     <head>
    #         <meta charset="UTF-8">
    #         <title>Python App with JavaScript</title>
    #     </head>
    #     <body>
    #         <div data-pym-src='https://www.jdoodle.com/plugin' data-language="{language}"
    #             data-version-index="0" data-libs="">
    #             {script_code}
    #         </div>
    #         <script src="https://www.jdoodle.com/assets/jdoodle-pym.min.js" type="text/javascript"></script>
    #     </body>
    #     </html>
    #     """.format(language=LANGUAGE_CODES[language], script_code=code_prompt)
    #     return html_template

    # # Setup logging method.


    # def setup_logging(log_file):
    #     logging.basicConfig(
    #         level=logging.INFO,
    #         format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    #         datefmt="%H:%M:%S",
    #         filename=log_file,  # Add this line to save logs to a file
    #         filemode='a',  # Append logs to the file
    #     )


    # # Setup logging
    # log_file = __file__.replace(".py", ".log")
    # setup_logging(log_file)


    # # Create a class
    # class PythonREPL:
    #     # Define the initialization method
    #     def __init__(self):
    #         pass

    #     # Define the run method
    #     def run(self, command: str) -> str:
    #         # Store the current value of sys.stdout
    #         old_stdout = sys.stdout
    #         # Create a new StringIO object
    #         sys.stdout = mystdout = StringIO()
    #         # Try to execute the code
    #         try:
    #             # Execute the code
    #             exec(command, globals())
    #             sys.stdout = old_stdout
    #             output = mystdout.getvalue()
    #         # If an error occurs, print the error message
    #         except Exception as e:
    #             # Restore the original value of sys.stdout
    #             sys.stdout = old_stdout
    #             # Get the error message
    #             output = str(e)
    #         return output

    # # Define the Run query function


    # def run_query(query, model_kwargs, max_iterations):
    #     # Create a LangChainOpenAI object
    #     llm = LangChainOpenAI(**model_kwargs)
    #     # Create the python REPL tool
    #     python_repl = lc_agents.Tool("Python REPL", PythonREPL(
    #     ).run, "A Python shell. Use this to execute python commands.")
    #     # Create a list of tools
    #     tools = [python_repl]
    #     # Initialize the agent
    #     agent = lc_agents.initialize_agent(tools, llm, agent=lc_agents.AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #                                     model_kwargs=model_kwargs, verbose=True, max_iterations=max_iterations)
    #     # Run the agent
    #     response = agent.run(query)
    #     return response

    # # Define the Run code function


    # def run_code(code, language):
    #     logger = logging.getLogger(__name__)
    #     logger.info(f"Running code: {code} in language: {language}")

    #     if language == "Python":
    #         with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
    #             f.write(code)
    #             f.flush()

    #             logger.info(f"Input file: {f.name}")
    #             output = subprocess.run(
    #                 ["python", f.name], capture_output=True, text=True)
    #             logger.info(f"Runner Output execution: {output.stdout + output.stderr}")
    #             return output.stdout + output.stderr

    #     elif language == "C" or language == "C++":
    #         ext = ".c" if language == "C" else ".cpp"
    #         with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=True) as src_file:
    #             src_file.write(code)
    #             src_file.flush()

    #             logger.info(f"Input file: {src_file.name}")

    #             with tempfile.NamedTemporaryFile(mode="w", suffix="", delete=True) as exec_file:
    #                 compile_output = subprocess.run(
    #                     ["gcc" if language == "C" else "g++", "-o", exec_file.name, src_file.name], capture_output=True, text=True)

    #                 if compile_output.returncode != 0:
    #                     return compile_output.stderr

    #                 logger.info(f"Output file: {exec_file.name}")
    #                 run_output = subprocess.run(
    #                     [exec_file.name], capture_output=True, text=True)
    #                 logger.info(f"Runner Output execution: {run_output.stdout + run_output.stderr}")
    #                 return run_output.stdout + run_output.stderr

    #     elif language == "JavaScript":
    #         with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=True) as f:
    #             f.write(code)
    #             f.flush()

    #             logger.info(f"Input file: {f.name}")
    #             output = subprocess.run(
    #                 ["node", f.name], capture_output=True, text=True)
    #             logger.info(f"Runner Output execution: {output.stdout + output.stderr}")
    #             return output.stdout + output.stderr

    #     else:
    #         return "Unsupported language."

    # # Generate the code


    # def generate_code():
    #     logger = logging.getLogger(__name__)
    #     try:
    #         st.session_state.generated_code = code_chain.run(code_prompt)
    #         st.session_state.code_language = code_language
    #         st.code(st.session_state.generated_code,
    #                 language=st.session_state.code_language.lower())

    #         with st.expander('Message History'):
    #             st.info(memory.buffer)
    #     except Exception as e:
    #         st.write(traceback.format_exc())
    #         logger.error(f"Error in code generation: {traceback.format_exc()}")

    # # Execute the code
    # def execute_code(compiler_mode: str):
    #     logger = logging.getLogger(__name__)
    #     logger.info(f"Executing code: {st.session_state.generated_code} in language: {st.session_state.code_language} with Compiler Mode: {compiler_mode}")

    #     try:
    #         if compiler_mode == "online":
    #             html_template = generate_dynamic_html(
    #                 st.session_state.code_language, st.session_state.generated_code)
    #             html(html_template, width=720, height=800, scrolling=True)

    #         else:
    #             output = run_code(st.session_state.generated_code,
    #                             st.session_state.code_language)
    #             logger.info(f"Output execution: {output}")

    #             if "error" in output.lower() or "exception" in output.lower() or "SyntaxError" in output.lower() or "NameError" in output.lower():

    #                 logger.error(f"Error in code execution: {output}")
    #                 response = sequential_chain(
    #                     {'code_topic': st.session_state.generated_code})
    #                 fixed_code = response['code_fix']
    #                 st.code(fixed_code, language=st.session_state.code_language.lower())

    #                 with st.expander('Message History'):
    #                     st.info(memory.buffer)
    #                 logger.warning(f"Trying to run fixed code: {fixed_code}")
    #                 output = run_code(fixed_code, st.session_state.code_language)
    #                 logger.warning(f"Fixed code output: {output}")

    #             st.code(st.session_state.generated_code,
    #                     language=st.session_state.code_language.lower())
    #             st.write("Execution Output:")
    #             st.write(output)
    #             logger.info(f"Execution Output: {output}")

    #     except Exception as e:
    #         st.write("Error in code execution:")
    #         # Output the stack trace
    #         st.write(traceback.format_exc())
    #         logger.error(f"Error in code execution: {traceback.format_exc()}")
    #     # Session state variables
    # if "generated_code" not in st.session_state:
    #     st.session_state.generated_code = ""

    # if "code_language" not in st.session_state:
    #     st.session_state.code_language = ""

    # # Generate the code
    # if button_generate and code_prompt:
    #     generate_code()


    # # Execute the code
    # if button_run and code_prompt:
    #     code_state_option = "online" if compiler_mode == "Online" else "offline"
    #     execute_code(code_state_option)








