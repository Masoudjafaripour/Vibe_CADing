# Vibe-CADing: Modular RAG-Driven CAD Generation Pipeline using LangChain + LangGraph
# Author: Masoud Jafaripour

# This script builds a LangGraph workflow composed of LangChain modules
# for extracting a part spec from text, retrieving CADs, generating geometry,
# and verifying constraints — with a feedback loop.

# --- IMPORTS ---
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langgraph.graph import StateGraph, END

# --- SETUP LLM ---
llm = ChatOpenAI(model="gpt-4", temperature=0)

# --- STEP 1: Spec Extractor Agent (Planner) ---
planner_prompt = PromptTemplate.from_template(
    """
    You are a CAD planning assistant. 
    Given a user's request: "{input}", extract a CAD design specification in structured form.
    Include: object type, shape, material, key constraints (e.g., holes, dimensions, symmetry).
    """
)
planner_chain = LLMChain(llm=llm, prompt=planner_prompt)

# --- STEP 2: Retriever Agent (FAISS DB of CAD embeddings) ---
retriever = FAISS.load_local("cad_index", OpenAIEmbeddings())  # Requires pre-built FAISS index

def run_retriever(state):
    query = state["spec"]
    results = retriever.similarity_search(query, k=2)
    state["retrieved"] = [r.page_content for r in results]
    return state

# --- STEP 3: Generator Agent (LLM or diffusion wrapper) ---
generator_prompt = PromptTemplate.from_template(
    """
    Given this CAD spec: "{spec}"
    And similar parts: "{retrieved}"
    Propose a geometry or design strategy for the CAD model.
    Output a textual parametric description or instructions.
    """
)
generator_chain = LLMChain(llm=llm, prompt=generator_prompt)

def run_generator(state):
    input_data = {
        "spec": state["spec"],
        "retrieved": "\n\n".join(state.get("retrieved", []))
    }
    state["generated"] = generator_chain.run(input_data)
    return state

# --- STEP 4: Critic Agent (Constraint Validator) ---
critic_prompt = PromptTemplate.from_template(
    """
    Review the following CAD design: "{generated}"
    Does it satisfy constraints such as symmetry, holes, and dimensions from spec: "{spec}"
    Answer Yes or No. Then explain why.
    """
)
critic_chain = LLMChain(llm=llm, prompt=critic_prompt)

def run_critic(state):
    input_data = {
        "generated": state["generated"],
        "spec": state["spec"]
    }
    state["critic"] = critic_chain.run(input_data)
    return state

# --- BUILD LANGGRAPH WORKFLOW ---
workflow = StateGraph()

workflow.add_node("planner", planner_chain)
workflow.add_update("planner", lambda s, o: {**s, "spec": o})
workflow.add_edge("planner", "retriever")

workflow.add_node("retriever", run_retriever)
workflow.add_edge("retriever", "generator")

workflow.add_node("generator", run_generator)
workflow.add_edge("generator", "critic")

workflow.add_node("critic", run_critic)

# --- CONDITIONAL EDGE: END OR LOOP ---
def critic_decision(state):
    feedback = state["critic"].lower()
    if "yes" in feedback:
        return END
    return "planner"  # Loop back to refine

workflow.add_conditional_edges("critic", critic_decision)

# --- COMPILE AND RUN ---
app = workflow.compile()

if __name__ == "__main__":
    # Example user prompt
    user_input = {
        "input": "I want a small metal bracket with two mounting holes and symmetry."
    }
    result = app.invoke(user_input)
    print("\n\n✅ Final CAD Design:\n", result["generated"])
    print("\n🧪 Critic Feedback:\n", result["critic"])
    print("\n📐 Spec Extracted:\n", result["spec"])
    print("\n📁 Retrieved Parts:\n", result["retrieved"])
