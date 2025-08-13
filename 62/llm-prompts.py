from transformers import AutoModelForCausalLM, AutoTokenizer

revision_id = "e5ef2ecae00bee901d5063bc86e1f86eba183702"

model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        revision=revision_id,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        )

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", revision=revision_id)

from transformers import pipeline

generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=500,
        do_sample=False
        )

while True:
    print("-" *50) # Horizontal line
    print("What do you want?")
    user_input = input("\033[92mType something, or X to exit: \033[0m") # Ask the user for input in green color
    if user_input in ['X', 'x']: # If user types X or x, exit the program
        print("Exiting.")
        break
    else:
        system_prompt = "You are a helpful assistant that helps with cooking recipes. You are also a great poet, and you can write poems about anything. When you respond, make sure to include a small 4 line poem like William Shakespeare would."
        messages = [
            {"role":"system", "content": system_prompt},
            {"role":"user", "content": user_input}
        ]
        response = generator(messages)
        print(response[0]["generated_text"])