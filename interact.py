from transformers import AutoTokenizer, AutoModelForCausalLM


def main(model_path="./model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    while True:
        try:
            prompt = input("You: ")
        except EOFError:
            break
        if not prompt:
            break
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50)
        print("LLM:", tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
