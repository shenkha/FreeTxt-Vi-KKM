import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import os
import time
import scattertext as st
import spacy
nlp = spacy.load('/freetxt/en_core_web_sm-3.2.0')  # Load the spaCy model
nlp.max_length = 9000000
from nltk.corpus import stopwords
import nltk

### stopwords_files
# Update with the Welsh stopwords (source: https://github.com/techiaith/ataleiriau)
en_stopwords = list(stopwords.words('english'))
cy_stopwords = open('/freetxt/website/data/welsh_stopwords.txt', 'r', encoding='iso-8859-1').read().split('\n') # replaced 'utf8' with 'iso-8859-1'
STOPWORDS = set(en_stopwords + cy_stopwords)
PUNCS = '''!→()-[]{};:'"\,<>?@#$%^&*_~'''

class SentimentAnalyser:
    """
    A class for performing sentiment analysis on textual data using pre-trained BERT models.
    
    Methods:
    preprocess_text(text): Preprocesses the text for sentiment analysis.
    analyse_sentiment(input_text, language, num_classes, max_seq_len=512): Analyzes the sentiment of the input text.
    generate_scattertext_visualization(dfanalysis, language): Generates a scattertext visualization for the sentiment analysis results.
    """
    
    def __init__(self):
        """
        Initializes the SentimentAnalyser class, loading the tokenizer and model for sentiment analysis.
        """
        # Loading tokenizer and model during initialization to avoid doing it multiple times.
        self.tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

    def preprocess_text(self,text):
        # remove URLs, mentions, and hashtags
        text = re.sub(r"http\S+|@\S+|#\S+", "", text)
        # remove punctuation and convert to lowercase
        text = re.sub(f"[{re.escape(''.join(PUNCS))}]", "", text.lower())
        # remove stopwords
        text = " ".join(word for word in text.split() if word not in STOPWORDS)
        return text

    def analyse_sentiment(self, input_text, language,num_classes, max_seq_len=512):
    # Split the input text into separate reviews
        print(num_classes)
        reviews = input_text
        print(reviews)
        print(language)

    # Initialize sentiment counters based on num_classes
        if int(num_classes) == 3:
            
            if language == 'en':
                sentiment_counts = {'Negative': 0, 'Neutral': 0, 'Positive': 0}
                
            elif language == 'cy':
                sentiment_counts = {'Negyddol': 0, 'Niwtral': 0, 'Cadarnhaol': 0}
                
        else:  # num_classes == 5
            if language == 'en':
                sentiment_counts = {'Very negative': 0, 'Negative': 0, 'Neutral': 0, 'Positive': 0, 'Very positive': 0}
                
            elif language == 'cy':
                sentiment_counts = {'Negyddol Iawn': 0, 'Negyddol': 0, 'Niwtral': 0, 'Cadarnhaol': 0, 'Cadarnhaol Iawn': 0}
               
    # Predict sentiment for each review


    # Sentiment labels for 5 classes
        if language == 'en':
            sentiment_labels = ['Very negative', 'Negative', 'Neutral', 'Positive', 'Very positive']
        elif language == 'cy':
            sentiment_labels = ['Negyddol Iawn', 'Negyddol', 'Niwtral', 'Cadarnhaol', 'Cadarnhaol Iawn']
    # Predict sentiment for each review
        sentiments = []
        for review in reviews:
            original_review = review
            review = self.preprocess_text(review)
        
            if review:
            # Tokenize the review
                tokens = self.tokenizer.encode(review, add_special_tokens=True, truncation=True)

            # If the token length exceeds the maximum, split into smaller chunks
                token_chunks = []
                if len(tokens) > max_seq_len:
                    token_chunks = [tokens[i:i + max_seq_len] for i in range(0, len(tokens), max_seq_len)]
                else:
                    token_chunks.append(tokens)

            # Process each chunk
                sentiment_scores = []
                for token_chunk in token_chunks:
                    input_ids = torch.tensor([token_chunk])
                    attention_mask = torch.tensor([[1] * len(token_chunk)])

                # Run the model
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    scores = outputs.logits.softmax(dim=1).detach().numpy()[0]
                    sentiment_scores.append(scores)

            # Aggregate the scores
                avg_scores = np.mean(sentiment_scores, axis=0)
                sentiment_index = avg_scores.argmax()

            # Handle sentiment categorization based on number of classes
                if int(num_classes) == 3:
                    if language == 'en':
                        sentiment_labels_3 = ['Negative', 'Neutral', 'Positive']
                    elif language == 'cy':
                        sentiment_labels_3 = ['Negyddol', 'Niwtral', 'Cadarnhaol']
                    if sentiment_index < 2:
                        sentiment_label = sentiment_labels_3[0]  # Negative
                    elif sentiment_index > 2:
                        sentiment_label = sentiment_labels_3[2]  # Positive
                    else:
                        sentiment_label = sentiment_labels_3[1]  # Neutral
                else:  # num_classes == 5
                    sentiment_label = sentiment_labels[sentiment_index]

                sentiment_score = float(format(avg_scores[sentiment_index], ".2f"))
                sentiments.append((original_review, sentiment_label, sentiment_score))
                sentiment_counts[sentiment_label] += 1
        #print(sentiment_counts)
        return sentiments, sentiment_counts

    def generate_scattertext_visualization(self, dfanalysis,language):
        # Get the DataFrame with sentiment analysis results
        df = dfanalysis
        positive_label = "Cadarnhaol" if language == 'cy' else "Positive"
        if positive_label not in dfanalysis['Sentiment Label'].unique():
            # Notify the user that the 'Positive' category is not present
            # This could be a return statement, raising an exception, 
            return f"No data for the '{positive_label}' category found. Scattertext visualization cannot be generated."

        # Parse the text using spaCy
        df['ParsedReview'] = df['Review'].apply(nlp)
       
        corpus = st.CorpusFromParsedDocuments(
            df,
             category_col="Sentiment Label",
            parsed_col="ParsedReview"
            ).build()
        
        term_scorer = st.RankDifference()
         ## # Determine which text to use based on the selected language
        visualisation_text_en = "Visualisation by"
        visualisation_text_cy = "Gweledigaeth gan"
        if language == 'en':
            visualisation_text = visualisation_text_en
        elif language == 'cy':
             visualisation_text = visualisation_text_cy       
        if language == 'en':
            html = st.produce_scattertext_explorer(
            corpus,
            category="Positive",
            category_name="Positive",   
            not_category_name='Negative_and_Neutral',
            not_categories=df["Sentiment Label"].unique().tolist(),
            minimum_term_frequency=5,
            pmi_threshold_coefficient=5,
            width_in_pixels=900,
            metadata=df["Sentiment Label"],
            term_scorer=term_scorer
        ) 
        elif language == 'cy':
            html = st.produce_scattertext_explorer(
            corpus,
            category="Cadarnhaol",
            category_name="Cadarnhaol",   
            not_category_name='Negyddol_a_Niwtral',
            not_categories=df["Sentiment Label"].unique().tolist(),
            minimum_term_frequency=5,
            pmi_threshold_coefficient=5,
            width_in_pixels=900,
            metadata=df["Sentiment Label"],
            term_scorer=term_scorer
        ) 
            html = html.replace('Frequent', 'Aml')
            html = html.replace('Average', 'Cyfartalog')
            html = html.replace('Infrequent', 'Anaml')
            html = html.replace('Negative_and_Neutral Frequency', 'Amlder Negyddol_a_Niwtral')
            html = html.replace('document count', 'cyfrif y ddogfen')
            html = html.replace('word count', 'cyfrif geiriau')
            html = html.replace('document count', 'cyfrif y ddogfen')
            html = html.replace('Frequency', 'Amlder')
            html = html.replace('Top', 'Uchaf')
            html = html.replace('Characteristic', 'Nodweddion')
            html = html.replace('Search the chart', 'Chwilio’r siart')
            html = html.replace('per', 'fesul')
            html = html.replace('words', 'gair')
            html = html.replace('score', 'sgôr')
            html = html.replace('frequency','amlder')        
            html = html.replace('terms','termau')
            html = html.replace('docs','dogfennau')
            html = html.replace('documents','dogfennau')
            html = html.replace('Not found in any','Heb ei g/eu canfod o gwbl')
            html = html.replace('Some of the','Rhai o’r')
            html = html.replace('mentions','crybwylliadau')

        timestamp = int(time.time())

        # Constructing the file path
        filename = os.path.join("/freetxt/website/static/wordcloud", f"scattertext_visualization_{timestamp}.html")
        with open(filename, "w", encoding='utf-8') as f:
            f.write(html)
            f.close()
        addition = f"""
    <div style="text-align:center; margin-top:30px;">
        {visualisation_text} <img src="https://ucrel-freetxt-2.lancs.ac.uk/static/images/logo.png" alt="Logo" style="height:40px;">
    </div>
    """
        html += addition
        filename_logo = os.path.join("/freetxt/website/static/wordcloud", f"scattertext_visualization_{timestamp}_logo.html")
    # Saving the updated HTML content to the file with UTF-8 encoding
        with open(filename_logo, "w", encoding='utf-8') as f_logo:
            f_logo.write(html)
            f_logo.close()
        

        

        # Returning the relative path for web access
        return f"static/wordcloud/scattertext_visualization_{timestamp}.html"
    
def wrap_html_content(file_name):
       # Step 1: Read the File
        with open(file_name, 'r', encoding='utf-8') as file:
            content = file.read()

        # Step 2: Wrap Content
        wrapped_content = f"""
        <!DOCTYPE html>
            <html lang="en">
            <head>
             <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Wrapped Content</title>
            </head>
            <body>
            {content}
        </body>
        </html>
         """

        # Step 3: Save to New File
        output_file_name = file_name
        with open(output_file_name, 'w', encoding='utf-8') as file:
            file.write(wrapped_content)

        #print(f"Content wrapped and saved to {output_file_name}")
        return output_file_name
        
