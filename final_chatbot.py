from fastapi import FastAPI, Depends
from pydantic import BaseModel
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools import StructuredTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import requests
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.prompts import PromptTemplate
from deep_translator import GoogleTranslator

load_dotenv()

app = FastAPI()

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

api_key = os.getenv('NINGA_API_KEY')
groq = os.getenv('GROQ_API_KEY')
nvidia = os.getenv('NVIDIA_API_KEY')
llm = ChatGroq(api_key=groq, model_name='deepseek-r1-distill-qwen-32b')
api_url = 'https://api.calorieninjas.com/v1/nutrition?query='

nutrition_keywords = {
    "calories", "nutrition", "diet",
    "rice", "wheat", "oats", "corn", "barley", "quinoa", "millet", "rye",
    "spinach", "lettuce", "kale", "carrot", "potato", "beet", 
    "broccoli", "cauliflower", "cabbage", "tomato", "bell pepper", "eggplant",
    "peas", "green beans", "zucchini", "pumpkin",
    "orange", "lemon", "grapefruit", "strawberry", "blueberry", "raspberry",
    "banana", "mango", "pineapple", "peach", "cherry", "plum",
    "watermelon", "cantaloupe", "apple", "pear", "grape",
    "beef", "pork", "lamb", "chicken", "turkey", "duck",
    "salmon", "tuna", "shrimp", "egg", 'meat',
    "lentils", "chickpeas", "black beans", "kidney beans", 
    "almonds", "walnuts", "sunflower seeds", "tofu", "tempeh", "edamame",
    "milk", "goat milk", "cheddar cheese", "mozzarella cheese", "feta cheese",
    "yogurt", "butter", "almond milk", "soy milk", "coconut yogurt",
    "ghee", "olive oil", "coconut oil", "sunflower oil", "canola oil", "soybean oil",
    "flaxseeds", "chia seeds", "peanuts", "avocado",
    "basil", "cilantro", "parsley", "garlic", "onion", "ginger",
    "black pepper", "cumin", "paprika", "turmeric", "cinnamon", "cloves",
    "sugar", "brown sugar", "honey", "maple syrup", "molasses",
    "tea", "coffee", "orange juice"
}

class QueryRequest(BaseModel):
    question: str
    session_id: str

def translate_text(text, target_lang):
    return GoogleTranslator(source='auto', target=target_lang).translate(text)

def detect_language(text):
    return GoogleTranslator(source='auto', target='en').translate(text) != text

def Ninga(query: str):
    response = requests.get(f'{api_url}{query}', headers={'X-Api-Key': api_key})
    if response.status_code == 200:
        data = response.json()
        if data.get("items"):
            item = data["items"][0]
            # Format as string instead of dict
            return (
                f"Nutrition data for {item.get('name', 'Unknown')}:\n"
                f"- Serving Size: {item.get('serving_size_g', 'N/A')}g\n"
                f"- Calories: {item.get('calories', 'N/A')}\n"
                f"- Total Fat: {item.get('fat_total_g', 'N/A')}g\n"
                f"- Protein: {item.get('protein_g', 'N/A')}g\n"
                f"- Sugar: {item.get('sugar_g', 'N/A')}g"
            )
        return "No nutrition data found."
    return f"API Error: {response.status_code}"

api_calling_tool = StructuredTool.from_function(
    func=Ninga,
    name='Ninga',
    description='Returns the nutrition information of a food item'
)

tools = [api_calling_tool]

prompt = ChatPromptTemplate.from_messages([
    ('system', """You are a nutritionist and you have two primary goals:
     1. If the user asks about a food item, always provide its nutritional details first by calling the API.
     2. If the user asks to replace an item, first find an alternative food that has a similar nutritional profile give two in minimum.
     3. Then, compare the new food with the original item based on nutrition data.
     
     Always format your answer like this:
     - **Replacement Food:** [Food Name]
     - **Comparison:**
         - **Original:** [Calories, Protein, Fat, etc.]
         - **Replacement:** [Calories, Protein, Fat, etc.]

     Never make assumptions or provide information without calling the API first.
     """),
    ('human', '{input}'),
    ("placeholder", "{agent_scratchpad}")
])

tool_agent = create_tool_calling_agent(llm, tools, prompt)
agent_ex = AgentExecutor(agent=tool_agent, tools=tools, verbose=True)

llm_chatbot = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", api_key=nvidia)

prompt_template = PromptTemplate(
    input_variables=['question'],
    template="""You are a fitness coach with 10 years of experience. You are an expert in both nutrition and gym training.
You also have knowledge of human anatomy, body parts, and how to target them in weightlifting training.
You can answer fitness/nutrition questions in **both Arabic and English**.  
**Important Rules:**
- Always reply in the **same language** as the user's question.
- If the question is related to nutrition, extract and explain the nutritional data.
- Keep responses **concise yet informative**.
question:
{question}"""
)

sop = StrOutputParser()
fitness_chain = prompt_template | llm_chatbot | sop

with_message_history = RunnableWithMessageHistory(
    fitness_chain,
    get_session_history,
    input_messages_key="question"
)

@app.post("/query")
def query_nutrition(request: QueryRequest):
    user_prompt = request.question
    session_id = request.session_id
    user_lang = 'ar' if detect_language(user_prompt) else 'en'
    translated_prompt = translate_text(user_prompt, 'en') if user_lang == 'ar' else user_prompt
    user_prompt_lower = translated_prompt.lower()
    if any(keyword in user_prompt_lower for keyword in nutrition_keywords):
        answer_ = agent_ex.invoke({'input': user_prompt_lower}, config={"configurable": {"session_id": session_id}})
        answer = with_message_history.invoke({'question': answer_['output']}, config={"configurable": {"session_id": session_id}})
    else:
        answer = with_message_history.invoke({'question': user_prompt_lower}, config={"configurable": {"session_id": session_id}})   

    final_answer = translate_text(answer, 'ar') if user_lang == 'ar' else answer
    
    return {"response": final_answer}
