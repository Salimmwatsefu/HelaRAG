
from transformers import MT5ForConditionalGeneration, MT5TokenizerFast
import torch

def load_model_and_tokenizer(model_name):
    tokenizer = MT5TokenizerFast.from_pretrained(model_name, legacy=True)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

def run_generation(query, context, config):
    print(f"Generating for: {query}")
    print(f"Context used: {context}")
    
    # Get config values
    model_name = config['model']['generation_model']
    max_length = config['generation']['max_length']
    prompt_template = config.get('generation', {}).get('prompt_template', "Reviews: {context}\nQuestion: {query}\nAnswer:")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Prepare prompt
    prompt = prompt_template.format(context=context, query=query)
    print(f"Prompt: {prompt}")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, padding=True)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=5,
            temperature=config['generation'].get('temperature', 1.0),
            length_penalty=1.0,
            early_stopping=True
        )
    
    # Decode response
    raw_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"Raw response: {raw_response}")

# Extract generation after <extra_id_0>
    if "<extra_id_0>" in raw_response:
        parts = raw_response.split("<extra_id_0>")
        response = parts[1].strip() if len(parts) > 1 else ""
    else:
            response = raw_response.strip()

    if not response or "<extra_id_" in response:
        response = "Unable to generate a clear answer based on reviews"

    scores = {"score": 0.9 if response != "Unable to generate a clear answer based on reviews" else 0.0}

    
    print(f"Response: {response}")
    return response, scores
