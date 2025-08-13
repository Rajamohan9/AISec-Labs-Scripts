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

print("What do you want?")
user_input = input()

messages = [{"role":"user", "content": user_input}]
response = generator(messages)
print(response[0]["generated_text"])