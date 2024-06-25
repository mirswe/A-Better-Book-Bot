from transformers import pipeline


path_to_file = "texts/DormvsApartment.txt"

def get_book_text(path):
    with open(path, 'r') as f:
        return f.read()
    
def chunk_text(text, max_length=1024):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]
    
text = get_book_text(path_to_file)
chunks = chunk_text(text)

print("--- Begin report of texts ---\n")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

summaries = [summarizer(chunk)[0]['summary_text'] for chunk in chunks]
model_text = " ".join(summaries)
print(model_text)
print("\n--- End report ---")


'''
Next Features to implement 
    - text selector ; choose what texts from the texts/ to choose from
    - clean up output ; make the printed output less janky
    - slow down output ; provide sleep() functions to slow down the output
    - add exceptions ; especially for max length text provided and others like network error


'''