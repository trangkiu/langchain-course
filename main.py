from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()



def main():
    information = """Bulbasaur is a small, quadrupedal amphibian Pokémon that has blue-green skin with darker patches.
     It has red eyes with white pupils, pointed, ear-like structures on top of its head, and a short, blunt snout with a wide mouth. Small, pointed teeth are visible in the upper jaw when the mouth is open. Each of its thick legs ends with three sharp claws. On Bulbasaur's back is a green plant bulb that conceals two slender, tentacle-like vines, which grow from a seed planted there at birth. The bulb also provides it with energy through photosynthesis and from the nutrient-rich seeds contained within"""
    summary_template = """
        Given the information {info} about this pokemon , I want to create : 
        1. A short summary 
        2. A haiku about it"""
    summary_prompt_template = PromptTemplate(
        input_variables= ["info"],
        template= summary_template
    )

    llm = ChatOpenAI(temperature=0.7, model="gpt-4o")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"info": information})
    print(response.content)


# run with local ollama
def main_local_ollama():
    print("Hello from langchain-course!")
    information = """Bulbasaur is a small, quadrupedal amphibian Pokémon that has blue-green skin with darker patches.
     It has red eyes with white pupils, pointed, ear-like structures on top of its head, and a short, blunt snout with a wide mouth. Small, pointed teeth are visible in the upper jaw when the mouth is open. Each of its thick legs ends with three sharp claws. On Bulbasaur's back is a green plant bulb that conceals two slender, tentacle-like vines, which grow from a seed planted there at birth. The bulb also provides it with energy through photosynthesis and from the nutrient-rich seeds contained within"""
    summary_template = """
        Given the information {info} about this pokemon , I want to create : 
        1. A short summary 
        2. A haiku about it"""
    summary_prompt_template = PromptTemplate(
        input_variables= ["info"],
        template= summary_template
    )

    llm = ChatOllama(temperature=0.7, model="gemma3:latest")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"info": information})
    print(response.content)

if __name__ == "__main__":
    main_local_ollama()
