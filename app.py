from flask import Flask, request, render_template, jsonify
import spacy
from spacy.util import minibatch



app = Flask(__name__)

@app.route('/')
def upload():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/train', methods=['POST'])
def train():
    nlp = spacy.load("en")  # Load the English language model
    ner = spacy.blank("en")
    train_data = request.files['train_data']
    test_data = request.files['test_data']

    ner.add_pipe(nlp.create_pipe("ner"))

    optimizer = ner.begin_training()


    for i in range(20):
        losses = {}
        # Create the batch generator with batch size = 8
        batches = minibatch(train_data, size=8)
        for batch in batches:
            texts, annotations = zip(*batch)
            ner.update(texts, annotations, sgd=optimizer, losses=losses)
    print(losses)

    return 'Model trained!'


@app.route('/test', methods=['GET', 'POST'])
def test():
    
    if request.method == 'POST':
        nlp = spacy.load("path/to/trained/model")  # Load the trained NER model

        text = request.form['text']
        doc = nlp(text)
        categories = doc.cats
        return render_template('test.html', categories=categories)
    return render_template('test.html')



if __name__ == '__main__':
    app.run()
