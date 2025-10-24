#Test for a simple chatbot

import pandas as pd
import numpy as np
from transformers import pipeline

#Run for Text Generation
#generator = pipeline('text-generation', model='distilgpt2')

#response = generator("The future of AI is", max_length=50, num_return_sequences=1)
#print(response[0]['generated_text'])

#Run for Text Classification
#classifier = pipeline('text-classification', model='distilbert-base-uncased')

#result = classifier("This is a test sentence")
#print(result)


#Run for Sentiment Analysis

#sentiment = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

#result = sentiment("I love this model!")
#print(result)


#Run For QA purposes
qa_model = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

result = qa_model(
    question="What is Darden?",
    context="Wherever you are in your career or whatever your goals, your education must be more than a transaction. At Darden, we educate the doers with the skills, smarts, and sense of purpose and ethics to forge the future.To reach your full potential, you have to do the work. We set the stage for an experience that is anything but typical. At Darden, you will not find lectures. You will find immersive experiences that will teach you to discover your purpose, ask the right questions, empower others, and go from inclusive vision to enduring impact."
)

display(result['answer'])


import ipywidgets as widgets
from IPython.display import display, clear_output

#generator = pipeline('text-generation', model='distilgpt2')

qa_model = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

context_input = widgets.Textarea(
    placeholder='Enter context/passage here...',
    description='Context:',
    layout=widgets.Layout(width='500px', height='100px')
)

question_input = widgets.Text(
    placeholder='Ask a question about the context...',
    description='Question:',
    layout=widgets.Layout(width='500px')
)

answer_button = widgets.Button(
    description='Get Answer',
    button_style='info'
)

output_area = widgets.Output()

def on_answer_click(b):
    with output_area:
        clear_output()
        if context_input.value and question_input.value:
            print("Finding answer...")
            result = qa_model(
                question=question_input.value,
                context=context_input.value
            )
            print(f"\nAnswer: {result['answer']}")
            print(f"Confidence: {result['score']:.2%}")
        else:
            print("Please provide both context and question!")

answer_button.on_click(on_answer_click)

display(context_input)
display(question_input)
display(answer_button)
display(output_area)

