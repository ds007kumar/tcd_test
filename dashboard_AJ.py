# === dashboard.py ===
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from src.openai_client import OpenAIAnalyzer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import io
import base64
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from typing import Dict, List, Tuple
# import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
import os
# from google import genai
import re
import google.generativeai as genai
##add
from textblob import TextBlob
from gensim import corpora
from gensim.models import LdaModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

from nltk.tokenize import word_tokenize

import json  # For potential serialization if needed


# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

class DashboardGenerator:
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd'
        }
        self.nlp_stopwords = set(stopwords.words('english')).union(STOPWORDS)
        self.lemmatizer = WordNetLemmatizer()

    # def incident_trends_over_time(self, df: pd.DataFrame) -> go.Figure:
    #     df['Created'] = pd.to_datetime(df['Created'], errors='coerce')
    #     df_trend = df.groupby(df['Created'].dt.date).size().reset_index(name='Count')
    #     fig = px.line(df_trend, x='Created', y='Count', title='Incident Trends Over Time', color_discrete_sequence=[self.color_palette['primary']])
    #     return fig

    def get_kpi_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        df = df.copy()

        total_incidents = len(df)

        # Critical incidents
        critical_incidents = df['Is_Critical'].sum() if 'Is_Critical' in df.columns else 0

        # Average resolution time
        if 'Resolved' in df.columns and 'Created' in df.columns:
            df['Created'] = pd.to_datetime(df['Created'], errors='coerce')
            df['Resolved'] = pd.to_datetime(df['Resolved'], errors='coerce')
            df['ResolutionHours'] = (df['Resolved'] - df['Created']).dt.total_seconds() / 3600
            avg_resolution_time = df['ResolutionHours'].mean()
        elif 'Resolve time' in df.columns:
            avg_resolution_time = df['Resolve time'].mean()
        elif 'Business resolve time' in df.columns:
            avg_resolution_time = df['Business resolve time'].mean()
        else:
            avg_resolution_time = 0

        # Resolution Rate
        if 'Is_Resolved' in df.columns:
            resolution_rate = df['Is_Resolved'].sum() / total_incidents * 100
        elif 'Is_Closed' in df.columns:
            resolution_rate = df['Is_Closed'].sum() / total_incidents * 100
        else:
            resolution_rate = 0

        # SLA Compliance
        sla_compliance = 0
        if 'Made SLA' in df.columns:
            sla_made = df['Made SLA'].astype(str).str.lower().eq('true').sum()
            sla_compliance = sla_made / total_incidents * 100

        return {
            'Total Incidents': total_incidents,
            'Critical Incidents': int(critical_incidents),
            'Avg Resolution Time (Hours)': round(avg_resolution_time, 2) if pd.notna(avg_resolution_time) else 0,
            'Resolution Rate (%)': round(resolution_rate, 2),
            'SLA Compliance (%)': round(sla_compliance, 2)
        }

    def incident_trends_over_time(self, df: pd.DataFrame) -> go.Figure:
        df['Created'] = pd.to_datetime(df['Created'], errors='coerce')
        df = df[df['Created'].notnull() & df['Predicted Category'].notnull()]
        
        df['Date'] = df['Created'].dt.date
        trend_df = df.groupby(['Date', 'Predicted Category']).size().reset_index(name='Count')

        fig = px.line(
            trend_df,
            x='Date',
            y='Count',
            color='Predicted Category',
            # title='Incident Trends Over Time by Category',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Number of Incidents',
            hovermode='x unified',
            legend_title='Predicted Category'
        )
        
        return fig
    
    # def analyze_recurring_incidents(self, df: pd.DataFrame) -> Dict:
    #     text_columns = ['Short description', 'Additional comments', 'Work notes', 'Comments and Work notes', 'Description']
    #     df['Combined_Text'] = df[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
    #     recurring_patterns = df.groupby('Combined_Text').size().reset_index(name='Occurrences')
    #     recurring_patterns = recurring_patterns[recurring_patterns['Occurrences'] > 1].sort_values('Occurrences', ascending=False)
    #     top_recurring = recurring_patterns.head(10).to_dict('records')
    #     return {'top_recurring_incidents': top_recurring}
    

    # def generate_rca_recommendations(self, df: pd.DataFrame) -> Dict:
    #     text_columns = ['Short description', 'Additional comments', 'Work notes', 'Comments and Work notes', 'Description']
    #     df['Combined_Text'] = df[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
    #     sample_text = df['Combined_Text'].iloc[0] if not df.empty else ""
    #     prompt = f"Analyze the following incident text for root cause analysis (RCA) and provide recommendations: {sample_text}"
        
    #     genai.configure(api_key=api_key)

    #     model = genai.GenerativeModel(model_name="gemini-2.0-flash")

    #     response = model.generate_content(prompt)
        
    #     # return response.text       
    #     return {'rca': response.text.split('Recommendations:')[0].strip(), 'recommendations': response.text.split('Recommendations:')[1].strip() if 'Recommendations:' in response.text else 'No recommendations available'}

    # def sentiment_analysis(self, df: pd.DataFrame) -> Dict:
    #     text_columns = ['Short description', 'Additional comments', 'Work notes', 'Comments and Work notes', 'Description']
    #     df['Combined_Text'] = df[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
    #     sentiments = df['Combined_Text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    #     sentiment_summary = {
    #         'Average_Sentiment': sentiments.mean(),
    #         'Positive_Percentage': (sentiments > 0).mean() * 100,
    #         'Negative_Percentage': (sentiments < 0).mean() * 100,
    #         'Neutral_Percentage': (sentiments == 0).mean() * 100
    #     }
    #     return sentiment_summary

    # def lda_topic_modeling(self, df: pd.DataFrame, num_topics: int = 5) -> Dict:
    #     text_columns = ['Short description', 'Additional comments', 'Work notes', 'Comments and Work notes', 'Description']
    #     df['Combined_Text'] = df[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
    #     df['Cleaned_Text'] = df['Combined_Text'].apply(lambda x: ' '.join([self.lemmatizer.lemmatize(word.lower()) for word in x.split() if word.lower() not in self.nlp_stopwords and len(word) > 2]))
    #     tokenized_docs = [doc.split() for doc in df['Cleaned_Text'].dropna()]
    #     dictionary = corpora.Dictionary(tokenized_docs)
    #     corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    #     lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=15)
    #     topics = lda_model.print_topics()
    #     topic_details = {f"Topic {i+1}": topic[1] for i, topic in enumerate(topics)}
    #     return {'topics': topic_details}
    
    # def analyze_recurring_incidents(self,df: pd.DataFrame, eps: float = 0.20, batch_size: int = 64) -> Dict:
    #     """
    #     Analyze recurring incidents in a DataFrame by clustering similar texts using embeddings.
    #     s
    #     Args:
    #         df (pd.DataFrame): Input DataFrame with incident data.
    #         eps (float): DBSCAN epsilon parameter for clustering (default: 0.25).
    #         api_key (str): API key for Google Generative AI (required for summaries).
    #         batch_size (int): Batch size for embedding creation (default: 32).
        
    #     Returns:
    #         Dict: Dictionary containing up to 7 clusters with summaries, occurrences, MTTR, and incidents.
    #     """
    #     # Validate inputs
    #     if not isinstance(df, pd.DataFrame) or df.empty:
    #         return {'clusters': []}
    #     if api_key is None:
    #         print("Warning: API key not provided. Summaries will be set to 'No summary generated'.")
        
    #     # Define text columns to combine
    #     text_columns = ['Short description', 'Additional comments', 'Work notes', 
    #                 'Comments and Work notes', 'Description', 'Predicted Category']
        
    #     # Ensure all text columns exist in DataFrame
    #     text_columns = [col for col in text_columns if col in df.columns]
    #     if not text_columns:
    #         return {'clusters': []}
        
    #     # Combine text columns, prioritizing 'Short description' and 'Description'
    #     def combine_text(row):
    #         weighted_texts = []
    #         for col in text_columns:
    #             if pd.notna(row[col]):
    #                 text = str(row[col])
    #                 if col in ['Short description', 'Description']:
    #                     weighted_texts.append(text * 2)  # Double weight for key columns
    #                 else:
    #                     weighted_texts.append(text)
    #         return ' '.join(weighted_texts)
        
    #     df['Combined_Text'] = df[text_columns].apply(combine_text, axis=1)
        
    #     # Compute MTTR
    #     df['Created'] = pd.to_datetime(df['Created'], errors='coerce')
    #     df['Resolved'] = pd.to_datetime(df['Resolved'], errors='coerce')
    #     df['MTTR'] = (df['Resolved'] - df['Created']).dt.total_seconds() / 3600.0
    #     df['MTTR'] = df['MTTR'].fillna(0)
        
    #     # Get unique texts
    #     unique_texts = df['Combined_Text'].drop_duplicates().tolist()
    #     if not unique_texts:
    #         return {'clusters': []}
        
    #     # Advanced text preprocessing
    #     stop_words = set(stopwords.words('english'))
    #     preserve_terms = {'mttr', 'sla', 'itil', 'incident', 'ticket', 'p1', 'p2', 'rca', 'slm'}  # ITSM-specific terms
    #     stop_words = stop_words - preserve_terms
    #     lemmatizer = WordNetLemmatizer()
        
    #     def preprocess_text(text: str) -> str:
    #         # Remove HTML tags, URLs, special characters, and normalize
    #         text = re.sub(r'<[^>]+>', '', text)  # HTML tags
    #         text = re.sub(r'http\S+', '', text)  # URLs
    #         text = re.sub(r'[^\w\s]', '', text)  # Special characters
    #         text = text.lower().strip()
            
    #         # Tokenize, remove stop words, and lemmatize
    #         tokens = word_tokenize(text)
    #         tokens = [lemmatizer.lemmatize(token) for token in tokens 
    #                 if token not in stop_words and len(token) > 2]
            
    #         # Truncate to 200 words to fit model token limit
    #         tokens = tokens[:500]
    #         return ' '.join(tokens)
        
    #     # Apply preprocessing to unique texts
    #     try:
    #         processed_texts = [preprocess_text(text) for text in unique_texts]
    #     except Exception as e:
    #         print(f"Warning: Text preprocessing failed: {e}")
    #         return {'clusters': []}
        
    #     # Initialize SentenceTransformer model
    #     try:
    #         embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    #     except Exception as e:
    #         print(f"Error: Failed to load SentenceTransformer model: {e}")
    #         return {'clusters': []}
        
    #     # Encode texts in batches
    #     try:
    #         embeddings = embed_model.encode(
    #             processed_texts,
    #             batch_size=batch_size,
    #             show_progress_bar=True,
    #             normalize_embeddings=True
    #         )
    #     except Exception as e:
    #         print(f"Error: Failed to generate embeddings: {e}")
    #         return {'clusters': []}
        
    #     # Optional: Dimensionality reduction with UMAP (uncomment to use)
    #     # try:
    #     #     import umap
    #     #     reducer = umap.UMAP(n_components=100, metric='cosine', random_state=42)
    #     #     embeddings = reducer.fit_transform(embeddings)
    #     # except ImportError:
    #     #     print("Warning: UMAP not installed. Skipping dimensionality reduction.")
    #     # except Exception as e:
    #     #     print(f"Warning: UMAP failed: {e}")
        
    #     # Clustering with DBSCAN
    #     try:
    #         clustering = DBSCAN(eps=eps, min_samples=2, metric='cosine').fit(embeddings)
    #         labels = clustering.labels_
    #     except Exception as e:
    #         print(f"Error: Clustering failed hahahah: {e}")
    #         return {'clusters': []}
        
    #     # Initialize LLM for summaries
    #     clusters = []
    #     if api_key:
    #         try:
    #             genai.configure(api_key=api_key)
    #             llm_model = genai.GenerativeModel("gemini-2.0-flash")
    #         except Exception as e:
    #             print(f"Warning: Failed to initialize LLM: {e}")
    #             llm_model = None
    #     else:
    #         llm_model = None
        
    #     # Generate clusters
    #     for label in set(labels):
    #         if label == -1:  # Skip noise points
    #             continue
    #         cluster_indices = [i for i, l in enumerate(labels) if l == label]
    #         cluster_texts = [unique_texts[i] for i in cluster_indices]  # Use original texts for summary
            
    #         # Generate cluster summary
    #         summary = "No summary generated"
    #         if llm_model:
    #             try:
    #                 prompt = (f"Act as an ITSM expert & generate a concise common description as "
    #                         f"category heading or cluster heading for these similar incidents "
    #                         f"within 10-15 words: {'; '.join(cluster_texts[:10])}")
    #                 response = llm_model.generate_content(prompt)
    #                 summary = response.text.strip() if response and hasattr(response, 'text') else "No summary generated"
    #             except Exception as e:
    #                 print(f"Warning: LLM summary generation failed for cluster {label}: {e}")
            
    #         # Compute cluster metrics
    #         cluster_df = df[df['Combined_Text'].isin(cluster_texts)]
    #         occurrences = len(cluster_df)
    #         avg_mttr = cluster_df['MTTR'].mean()
            
    #         clusters.append({
    #             'summary': summary,
    #             'occurrences': occurrences,
    #             'avg_mttr': round(avg_mttr, 2) if pd.notna(avg_mttr) else 0,
    #             'incidents': cluster_df.to_dict('records')
    #         })
        
    #     # Sort clusters by occurrences and return top 7
    #     clusters.sort(key=lambda x: x['occurrences'], reverse=True)
    #     return {'clusters': clusters[:7]}
    
    # def generate_rca_for_clusters(self, clusters: List[Dict]) -> List[Dict]:
    #     rca_list = []
    #     genai.configure(api_key=api_key)
    #     model = genai.GenerativeModel("gemini-2.0-flash")
        
    #     for cluster in clusters:
    #         # Combine texts from incidents in the cluster, truncate to avoid token limits
    #         incident_texts = [inc.get('Combined_Text', '') for inc in cluster['incidents'] if 'Combined_Text' in inc]
    #         combined_text = ' '.join(incident_texts)[:5000]  # Limit to ~2000 chars; adjust as needed
            
    #         if not combined_text:
    #             rca_list.append({'rca': 'No text available', 'recommendations': 'No recommendations available'})
    #             continue
            
    #         prompt = f"Perform root cause analysis (RCA) on this group of similar incidents and suggest preventive recommendations. Structure as: RCA: [analysis]. Recommendations: [list]. Incident Group Text: {combined_text}"
            
    #         response = model.generate_content(prompt)
    #         text = response.text if response else ""
    #         rca = text.split('Recommendations:')[0].replace('RCA:', '').strip() if 'RCA:' in text else text
    #         recommendations = text.split('Recommendations:')[1].strip() if 'Recommendations:' in text else 'No recommendations'
            
    #         rca_list.append({'rca': rca, 'recommendations': recommendations})
        
    #     return rca_list

    def sentiment_analysis(self, df: pd.DataFrame) -> Dict:
        text_columns = ['Short description', 'Additional comments', 'Work notes', 'Comments and Work notes', 'Description']
        df['Combined_Text'] = df[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
        
        def get_sentiment(text):
            blob = TextBlob(text)
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        
        df[['Polarity', 'Subjectivity']] = df['Combined_Text'].apply(lambda x: pd.Series(get_sentiment(x)))
        
        overall = {
            'Average_Polarity': df['Polarity'].mean(),
            'Average_Subjectivity': df['Subjectivity'].mean(),
            'Positive_Percentage': (df['Polarity'] > 0).mean() * 100,
            'Negative_Percentage': (df['Polarity'] < 0).mean() * 100,
            'Neutral_Percentage': (df['Polarity'] == 0).mean() * 100
        }
        
        per_category = {}
        for category in df['Predicted Category'].unique():
            cat_df = df[df['Predicted Category'] == category]
            per_category[category] = {
                'Average_Polarity': cat_df['Polarity'].mean(),
                'Average_Subjectivity': cat_df['Subjectivity'].mean(),
                'Positive_Percentage': (cat_df['Polarity'] > 0).mean() * 100,
                'Negative_Percentage': (cat_df['Polarity'] < 0).mean() * 100,
                'Neutral_Percentage': (cat_df['Polarity'] == 0).mean() * 100
            }
        
        return {'overall': overall, 'per_category': per_category}

    def lda_topic_modeling(self, df: pd.DataFrame, num_topics: int = 5) -> Dict:
        text_columns = ['Short description', 'Additional comments', 'Work notes', 'Comments and Work notes', 'Description']
        df['Combined_Text'] = df[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
        df['Cleaned_Text'] = df['Combined_Text'].apply(lambda x: ' '.join([self.lemmatizer.lemmatize(word.lower()) for word in x.split() if word.lower() not in self.nlp_stopwords and len(word) > 2]))
        tokenized_docs = [doc.split() for doc in df['Cleaned_Text'].dropna()]
        dictionary = corpora.Dictionary(tokenized_docs)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=15)
        
        # Readable topics: list of top words
        topics = {}
        for i, topic in lda_model.print_topics():
            words = [w.split('*')[1].strip('"') for w in topic.split(' + ')]
            topics[f"Topic {i+1}"] = ', '.join(words)
        
        # Assign dominant topic
        dominant_topics = []
        for bow in corpus:
            if bow:  # Skip empty
                topic_dist = lda_model.get_document_topics(bow)
                dominant = max(topic_dist, key=lambda x: x[1])[0] + 1
            else:
                dominant = None
            dominant_topics.append(dominant)
        df['Dominant_Topic'] = dominant_topics
        
        distribution = df['Dominant_Topic'].value_counts(normalize=True).to_dict()
        
        # Top 3 examples per topic
        examples = {}
        for topic in range(1, num_topics + 1):
            topic_df = df[df['Dominant_Topic'] == topic]
            examples[f"Topic {topic}"] = topic_df['Combined_Text'].head(3).tolist()
        
        return {'topics': topics, 'distribution': distribution, 'examples': examples}

    def category_distribution(self, df: pd.DataFrame) -> go.Figure:
        cat_counts = df['Predicted Category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        fig = px.pie(cat_counts, values='Count', names='Category')
        return fig

    def severity_by_hour_heatmap(self, df: pd.DataFrame) -> go.Figure:
        df['Created'] = pd.to_datetime(df['Created'], errors='coerce')
        df['Hour'] = df['Created'].dt.hour
        df['Weekday'] = df['Created'].dt.day_name()
        pivot = df.pivot_table(index='Weekday', columns='Hour', values='Priority', aggfunc='count', fill_value=0)
        fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale='Blues'))
        fig.update_layout(xaxis_title='Hour', yaxis_title='Weekday')
        return fig
    
    def weekday_incident_count_bar(self, df: pd.DataFrame) -> go.Figure:
        df['Created'] = pd.to_datetime(df['Created'], errors='coerce')
        df['Weekday'] = df['Created'].dt.day_name()
        order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        count_df = df['Weekday'].value_counts().reindex(order).reset_index()
        count_df.columns = ['Weekday', 'Count']
        
        fig = px.bar(count_df, x='Weekday', y='Count', title='Incident Count by Weekday',
                    color='Count', color_continuous_scale='Blues')
        return fig


    def wordcloud_bigrams_by_category(self, df: pd.DataFrame, category_col: str = 'Predicted Category', text_col: str = 'Short description') -> Dict[str, str]:
        result = {}
        categories = df[category_col].dropna().unique()
        for category in categories:
            df_cat = df[df[category_col] == category]
            text = ' '.join(df_cat[text_col].dropna().astype(str))
            text = text.lower()
            text = re.sub(r'[^a-z\s]', ' ', text)
            words = [self.lemmatizer.lemmatize(w) for w in text.split() if w not in self.nlp_stopwords and len(w) > 2]
            cleaned_text = ' '.join(words)

            vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english')
            X = vectorizer.fit_transform([cleaned_text])
            freqs = zip(vectorizer.get_feature_names_out(), X.toarray().sum(axis=0))
            freq_dict = dict(freqs)

            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq_dict)

            # Save image to buffer
            buf = io.BytesIO()
            fig, ax = plt.subplots(figsize=(10, 5))  # Use subplots to get the Axes object
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode()
            result[category] = img_base64

        return result

    def geographic_map(self, df: pd.DataFrame) -> go.Figure:
        if 'Country' not in df.columns:
            return go.Figure()
        geo_counts = df['Country'].value_counts().reset_index()
        geo_counts.columns = ['Country', 'Count']
        fig = px.choropleth(geo_counts, locations='Country', locationmode='country names', color='Count', title='Incidents Count by Country')
        return fig

    def priority_vs_state_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Generate sunburst chart for Priority vs Incident State distribution."""
        if 'Priority' in df.columns and 'Incident state' in df.columns:
            priority_state = df.groupby(['Priority', 'Incident state']).size().reset_index(name='Count')

            fig = px.sunburst(
                priority_state,
                path=['Priority', 'Incident state'],
                values='Count',
                title='Priority vs State Distribution',
                color='Count',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=500)
            return fig
        else:
            return go.Figure()  # Return empty figure if columns missing
        

    def priority_vs_state_sunburst_by_category(self,df: pd.DataFrame,selected_category: str,category_col: str = 'Predicted Category') -> go.Figure:
        """Generate sunburst chart for a specific category."""
        if 'Priority' not in df.columns or 'Incident state' not in df.columns or category_col not in df.columns:
            return go.Figure()

        filtered_df = df[df[category_col] == selected_category]
        if filtered_df.empty:
            return go.Figure()

        group_df = (
            filtered_df.groupby(['Priority', 'Incident state'])
            .size()
            .reset_index(name='Count')
        )

        if group_df.empty:
            return go.Figure()

        fig = px.sunburst(
            group_df,
            path=['Priority', 'Incident state'],
            values='Count',
            title=f'Priority vs Incident State - {selected_category}',
            color='Count',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=500)
        return fig


        
    def priority_vs_state_bar_all(self, df: pd.DataFrame) -> go.Figure:
        if 'Priority' in df.columns and 'Incident state' in df.columns:
            priority_state = df.groupby(['Priority', 'Incident state']).size().reset_index(name='Count')
            fig = px.bar(
                priority_state,
                x='Priority',
                y='Count',
                color='Incident state',
                barmode='group',  # or 'stack' if you want stacked bars
                title='Priority vs Incident State Distribution'
            )
            fig.update_layout(xaxis_title='Priority', yaxis_title='Number of Incidents')
            return fig
        return go.Figure()

    def priority_vs_state_bar_category(self, df: pd.DataFrame, category: str = None) -> go.Figure:
        """Generate grouped bar chart for Priority vs Incident State, optionally filtered by Predicted Category."""
        if 'Priority' in df.columns and 'Incident state' in df.columns:
            if category and 'Predicted Category' in df.columns:
                df = df[df['Predicted Category'] == category]

            priority_state = df.groupby(['Priority', 'Incident state']).size().reset_index(name='Count')
            fig = px.bar(
                priority_state,
                x='Priority',
                y='Count',
                color='Incident state',
                barmode='group',
                title=f'Priority vs Incident State Distribution{f" - {category}" if category else ""}'
            )
            fig.update_layout(xaxis_title='Priority', yaxis_title='Number of Incidents')
            return fig
        return go.Figure()



    def hierarchical_treemap(self, df: pd.DataFrame) -> List[go.Figure]:
        """Create multiple treemaps based on hierarchical columns."""
        figures = []

        if 'Predicted Category' in df.columns and 'Category 2' in df.columns:
            cat_data = df.groupby(['Predicted Category', 'Category 2']).size().reset_index(name='Count')
            fig = px.treemap(cat_data, path=['Predicted Category', 'Category 2'], values='Count',
                            title='Treemap: Predicted Category → Category 2')
            figures.append(fig)

        if 'Channel' in df.columns and 'Predicted Category' in df.columns:
            config_data = df.groupby(['Channel', 'Predicted Category']).size().reset_index(name='Count')
            fig = px.treemap(config_data, path=['Channel', 'Predicted Category'], values='Count',
                            title='Treemap: Channel → Predicted Category')
            figures.append(fig)


        return figures

    # def generate_boxplots(self, df: pd.DataFrame) -> List[go.Figure]:
    #     plots = []

    #     # Clean dates
    #     df['Created'] = pd.to_datetime(df['Created'], errors='coerce')
    #     df['Resolved'] = pd.to_datetime(df['Resolved'], errors='coerce')
    #     df['MTTR'] = (df['Resolved'] - df['Created']).dt.total_seconds() / 3600

    #     if 'Resolve time' in df.columns:
    #         fig1 = px.box(df, x='Predicted Category', y='MTTR', title='MTTR by Category')
    #         plots.append(fig1)

    #     if 'Priority' in df.columns and 'MTTR' in df.columns:
    #         fig4 = px.box(df, x='Priority', y='MTTR', title='MTTR by Priority')
    #         plots.append(fig4)

    #     return plots

    def generate_boxplot_for_category(self, df: pd.DataFrame, selected_category: str) -> go.Figure:
        df['Created'] = pd.to_datetime(df['Created'], errors='coerce')
        df['Resolved'] = pd.to_datetime(df['Resolved'], errors='coerce')
        df['MTTR'] = (df['Resolved'] - df['Created']).dt.total_seconds() / 3600

        df_filtered = df[df['Predicted Category'] == selected_category]

        fig = px.box(df_filtered, x='Priority', y='MTTR', title=f'MTTR by Priority - {selected_category}')
        return fig

    def reopen_analysis(self, df: pd.DataFrame) -> Tuple[int, float, go.Figure]:
        if 'Last reopened at' not in df.columns or 'Resolved' not in df.columns:
            return 0, 0.0, go.Figure()

        df['Resolved'] = pd.to_datetime(df['Resolved'], errors='coerce')
        df['Last reopened at'] = pd.to_datetime(df['Last reopened at'], errors='coerce')

        # Only those with valid reopened timestamps
        reopened_df = df[df['Last reopened at'].notna()]
        total_reopened = reopened_df.shape[0]
        reopen_rate = (total_reopened / df.shape[0]) * 100 if df.shape[0] > 0 else 0

        return total_reopened, round(reopen_rate, 2)

    def category_wise_reopens(self, df: pd.DataFrame, category_col: str = 'Predicted Category', reopen_col: str = 'Last reopened at') -> go.Figure:
        df[reopen_col] = pd.to_datetime(df[reopen_col], errors='coerce')
        df['Was Reopened'] = df[reopen_col].notna()

        summary_df = df.groupby(category_col).agg(
            Total_Incidents=('Was Reopened', 'count'),
            Reopened_Count=('Was Reopened', 'sum')
        ).reset_index()

        summary_df['Reopen_Rate (%)'] = (summary_df['Reopened_Count'] / summary_df['Total_Incidents']) * 100
        summary_df = summary_df.sort_values(by='Reopened_Count', ascending=False)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=summary_df[category_col],
            y=summary_df['Total_Incidents'],
            name='Total Incidents',
            marker_color='lightgrey'
        ))

        fig.add_trace(go.Bar(
            x=summary_df[category_col],
            y=summary_df['Reopened_Count'],
            name='Reopened Incidents',
            marker_color='orangered',
            text=[f"{rate:.1f}%" for rate in summary_df['Reopen_Rate (%)']],
            textposition='outside'
        ))

        fig.update_layout(
            title='🔁 Category-wise Reopened Incidents with Reopen Rate',
            xaxis_title='Category',
            yaxis_title='Incident Count',
            barmode='group',
            xaxis_tickangle=45,
            height=500
        )

        return fig


    # def mttr_trend(self, df: pd.DataFrame) -> go.Figure:
    #     df['Created'] = pd.to_datetime(df['Created'], errors='coerce')
    #     df['Resolved'] = pd.to_datetime(df['Resolved'], errors='coerce')
    #     df['Duration'] = (df['Resolved'] - df['Created']).dt.total_seconds() / 3600
    #     df['Week'] = df['Created'].dt.to_period('W').dt.start_time
    #     mttr = df.groupby('Week')['Duration'].mean().reset_index()
    #     fig = px.line(mttr, x='Week', y='Duration', title='Mean Time to Resolution (MTTR)', color_discrete_sequence=[self.color_palette['success']])
    #     fig.update_layout(yaxis_title='MTTR (Hours)')
    #     return fig
    def mttr_trend(self, df: pd.DataFrame) -> go.Figure:
        df['Created'] = pd.to_datetime(df['Created'], errors='coerce')
        df['Resolved'] = pd.to_datetime(df['Resolved'], errors='coerce')
        df['Duration'] = (df['Resolved'] - df['Created']).dt.total_seconds() / 3600
        df['Week'] = df['Created'].dt.to_period('W').dt.start_time

        # Filter rows with valid durations and categories
        df = df[df['Duration'].notnull() & df['Predicted Category'].notnull()]

        mttr = df.groupby(['Week', 'Predicted Category'])['Duration'].mean().reset_index()

        fig = px.line(
            mttr,
            x='Week',
            y='Duration',
            color='Predicted Category',
            # title='Mean Time to Resolution (MTTR) by Category',
            color_discrete_sequence=px.colors.qualitative.Set2  # change if needed
        )

        fig.update_layout(
            yaxis_title='MTTR (Hours)',
            xaxis_title='Week',
            legend_title='Predicted Category',
            hovermode='x unified'
        )
        return fig
    
    def incident_sankey_channel_category_priority_state(self, df):
        df = df[['Channel', 'Predicted Category', 'Priority', 'Incident state']].dropna()

        channels = df['Channel'].unique().tolist()
        categories_1 = df['Predicted Category'].unique().tolist()
        priorities = df['Priority'].unique().tolist()
        incident_states = df['Incident state'].unique().tolist()

        labels = channels + categories_1 + priorities + incident_states
        label_index = {label: i for i, label in enumerate(labels)}

        source = []
        target = []
        value = []

        # Channel → Predicted Category
        ch_to_cat = df.groupby(['Channel', 'Predicted Category']).size().reset_index(name='count')
        for _, row in ch_to_cat.iterrows():
            source.append(label_index[row['Channel']])
            target.append(label_index[row['Predicted Category']])
            value.append(row['count'])

        # Predicted Category → Priority
        cat_to_priority = df.groupby(['Predicted Category', 'Priority']).size().reset_index(name='count')
        for _, row in cat_to_priority.iterrows():
            source.append(label_index[row['Predicted Category']])
            target.append(label_index[row['Priority']])
            value.append(row['count'])

        # Priority → Incident State
        priority_to_state = df.groupby(['Priority', 'Incident state']).size().reset_index(name='count')
        for _, row in priority_to_state.iterrows():
            source.append(label_index[row['Priority']])
            target.append(label_index[row['Incident state']])
            value.append(row['count'])

        fig = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
            ),
            link=dict(
                source=source,
                target=target,
                value=value
            )
        ))

        fig.update_layout(title_text="Channel → Predicted Category → Priority → Incident State", font_size=10)
        return fig
    

    def facet_bar_chart(self, df):
        df_grouped = df.groupby(['Company', 'Service offering', 'Channel']).size().reset_index(name='Count')
        fig = px.bar(df_grouped, x="Company", y="Count", color="Service offering", 
                     facet_col="Channel", height=800)
        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor="white",
            paper_bgcolor="lightgrey"
        )
        fig.update_layout(title="Facet Bar Chart: Incident count by Company, Service offering and Channel")
        return fig
    

    def mask_sensitive_data(self,text: str) -> str:
        if not isinstance(text, str):
            return text

        # Mask IP addresses
        text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 'XXX.XXX.XXX.XXX', text)

        # Mask email addresses
        text = re.sub(r'[\w\.-]+@[\w\.-]+', 'user@domain.com', text)

        # Mask Incident numbers
        text = re.sub(r'INC\d+', 'INCXXXXX', text)

        # Mask Request numbers
        text = re.sub(r'RITM\d+', 'RITMXXXXX', text)

        # Mask DateTime (basic pattern, adaptable)
        text = re.sub(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}', '2025-XX-XX XX:XX:XX', text)

        return text


    # def prepare_reopen_incidents_for_llm(self,df_reopened):
    #     df_filtered = df_reopened[
    #         ['Short description', 'Created', 'Additional comments', 'Work notes', 'Channel',
    #         'Closed', 'Comments and Work notes', 'Configuration item', 'Last reopened at',
    #         'Opened', 'Resolved']
    #     ].copy()

    #     # Apply masking to all relevant columns
    #     for col in df_filtered.columns:
    #         df_filtered[col] = df_filtered[col].astype(str).apply(mask_sensitive_data)

    #     sample_rows = df_filtered.to_string(index=False)

    #     return sample_rows

    
    def get_reopened_incidents_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        df_reopened = df[df['Last reopened at'].notna()].copy()

        selected_columns = [
            "Short description", "Created", "Additional comments", "Work notes",
            "Channel", "Closed", "Comments and Work notes", "Configuration item",
            "Last reopened at", "Opened", "Resolved"
        ]
        df_reopened = df_reopened[selected_columns]
        df_filtered = df_reopened.copy()

        for col in df_filtered.columns:
            df_filtered[col] = df_filtered[col].astype(str).apply(self.mask_sensitive_data)

        return df_filtered
    
    def format_reopen_prompt(self,df_reopened: pd.DataFrame) -> str:
        sample_rows = df_reopened.to_string(index=False)  # Limit for token size
        prompt = """You are an expert Incident Analyst. Your task is to perform a concise Root Cause Analysis (RCA) on the provided dataset of reopened incidents. The goal is to deliver actionable insights and recommendations to reduce future reopen rates, specifically for a customer.

**Data provided is a table of reopened incidents with the following columns:**
- "Short description" (Contains brief summaries of the incident)
- "Created" (Timestamp of when the incident was created)
- "Additional comments" (Comments from the user/reporter)
- "Work notes" (Internal notes from technicians)
- "Channel" (How the incident was reported, e.g., 'email', 'phone')
- "Closed" (Timestamp of when the incident was closed)
- "Comments and Work notes" (A combined text field of all notes)
- "Configuration item" (The specific hardware or software affected)
- "Last reopened at" (Timestamp of the last reopening)
- "Opened" (Timestamp of the initial opening)
- "Resolved" (Timestamp of when the incident was resolved)

Based on this data:
    # - Provide key insights and possible reasons why these incidents were reopened.
    # - Highlight patterns such as common short descriptions, channels, or configuration items related to reopened incidents.
    # - Suggest actionable recommendations to reduce future incident reopen rates.
    # - Kindly include references/citations of the sources from where you are generating recommendations to reduce future reopen rates & show them in reference block under every actionable recommendation.
    # - Also, mention below masking techniques in the end of response that you are following -
    #   Masking Techniques Followed:
    #     To ensure data privacy and prevent leakage, the following masking techniques were applied to the raw incident data:

    #     IP addresses: (?:\d{1,3}\.){3}\d{1,3} replaced with XXX.XXX.XXX.XXX
    #     Email addresses: [\w\.-]+@[\w\.-]+ replaced with user@domain.com
    #     Incident numbers: INC\d+ replaced with INCXXXXX
    #     Request numbers: RITM\d+ replaced with RITMXXXXX
    #     Date/Time: \d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2} replaced with 2025-XX-XX XX:XX:XX
        
        

**Important Instructions:**
- Focus on the content of the "Short description", "Additional comments", and "Work notes" to identify the root cause of reopenings. Look for phrases like "not fixed," "still an issue," or repeated troubleshooting steps but not just limited to it.
- The tone should be professional and direct.
- Do not add any introductory or concluding remarks outside of the requested sections.
- Keep the overall response as brief and to the point as possible, focusing on value and conciseness."""
    #     prompt = f"""
    # You are an IT Service Management Analyst.

    # Here is a list of reopened incidents with relevant details:

    # {sample_rows}

    # Based on this data:
    # - Provide key insights and possible reasons why these incidents were reopened.
    # - Highlight patterns such as common short descriptions, channels, or configuration items related to reopened incidents.
    # - Suggest actionable recommendations to reduce future incident reopen rates.
    # - Kindly include references/citations of the sources from where you are generating recommendations to reduce future reopen rates & show them in reference block under every actionable recommendation & if it is general ITSM knowledge then mention that dont leave any reference blank.
    # - Also, mention below masking techniques in the end of response that you are following -
    #   Masking Techniques Followed:
    #     To ensure data privacy and prevent leakage, the following masking techniques were applied to the raw incident data:

    #     IP addresses: (?:\d{1,3}\.){3}\d{1,3} replaced with XXX.XXX.XXX.XXX
    #     Email addresses: [\w\.-]+@[\w\.-]+ replaced with user@domain.com
    #     Incident numbers: INC\d+ replaced with INCXXXXX
    #     Request numbers: RITM\d+ replaced with RITMXXXXX
    #     Date/Time: \d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2} replaced with 2025-XX-XX XX:XX:XX"""
        
        
        # prompt = f"""You are an IT Service Management Analyst.

        # Here is a list of reopened incidents with relevant details:

        # {sample_rows}

        # Your task:

        # 1 Provide 3-5 key insights summarizing why incidents were reopened. Be concise, avoid repeating similar points.

        # 2 Highlight any noticeable patterns (like common short descriptions, channels, configuration items) in bullet points, max 5 items.

        # 3 Provide 3 actionable recommendations to reduce future incident reopen rates.

        # Important:
        # - Focus on clarity and brevity.
        # - No lengthy explanations or repeating the same insights in different words.
        # - Format strictly as:

        # Insights:
        # - Point 1
        # - Point 2
        # ...

        # Patterns:
        # - Pattern 1
        # - Pattern 2
        # ...

        # Recommendations:
        # - Action 1
        # - Action 2
        # ...
        # """

        return prompt



    def get_gemini_reopen_insights(self,prompt: str) -> str:
        # client = genai.Client()

        # response = client.models.generate_content(
        #     model="gemini-2.5-flash", contents=prompt
        # )
        # return response.text
        

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(model_name="gemini-2.0-flash")

        response = model.generate_content(prompt)
        
        return response.text
class DashboardGenerator:
    def __init__(self):
        self.openai_analyzer = OpenAIAnalyzer()
        # ... existing initialization code ...
    
    def get_openai_reopen_insights(self, prompt_text: str) -> str:
        """Get insights on reopened incidents using OpenAI"""
        try:
            return self.openai_analyzer.get_reopen_insights(prompt_text)
        except Exception as e:
            print(f"Error getting OpenAI insights: {str(e)}")
            return "Unable to generate insights at this time. Please check OpenAI configuration."
    
    def format_reopen_prompt(self, df_reopened: pd.DataFrame) -> str:
        """Format prompt for OpenAI analysis of reopened incidents"""
        # Limit to top 20 reopened incidents for analysis
        sample_size = min(20, len(df_reopened))
        df_sample = df_reopened.head(sample_size)
        
        prompt = f"""Analyze the following {sample_size} reopened incidents and provide insights:

REOPENED INCIDENTS DATA:
"""
        
        for idx, row in df_sample.iterrows():
            prompt += f"""
Incident {idx + 1}:
- Number: {row.get('Number', 'N/A')}
- Category: {row.get('Predicted Category', 'N/A')}
- Description: {row.get('Short description', 'N/A')[:150]}...
- Assignment Group: {row.get('Assignment group', 'N/A')}
- Priority: {row.get('Priority', 'N/A')}
- Service Offering: {row.get('Service offering', 'N/A')}
- Reopen Count: {row.get('Reopen count', 'N/A')}
"""
        
        prompt += """

Please provide:
1. Common patterns in these reopened incidents
2. Likely root causes for reopening
3. Specific recommendations to reduce reopen rates
4. Process improvements that could be implemented
5. Any notable trends by category or assignment group

Focus on actionable insights that can help prevent future reopenings."""
        
        return prompt
    
    def analyze_recurring_incidents(self, df: pd.DataFrame, eps: float = 0.3, batch_size: int = 64) -> dict:
        """Analyze recurring incidents using OpenAI"""
        try:
            # Convert DataFrame to list of dictionaries
            incidents_data = df.to_dict('records')
            
            # Use OpenAI analyzer for recurring incidents analysis
            result = self.openai_analyzer.analyze_recurring_incidents(incidents_data, batch_size)
            
            return result
            
        except Exception as e:
            print(f"Error in recurring incidents analysis: {str(e)}")
            return {
                "clusters": [],
                "summary": "Analysis could not be completed due to an error.",
                "recommendations": []
            }
    
    def generate_rca_for_clusters(self, clusters: list) -> list:
        """Generate Root Cause Analysis for clusters using OpenAI"""
        try:
            return self.openai_analyzer.generate_rca_for_clusters(clusters)
        except Exception as e:
            print(f"Error generating RCA: {str(e)}")
            return [{"cluster": "Error", "rca": "Could not generate RCA", "recommendations": "N/A"}]
    
    def openai_health_check(self) -> bool:
        """Check if OpenAI service is available"""
        try:
            return self.openai_analyzer.health_check()
        except Exception:
            return False


    