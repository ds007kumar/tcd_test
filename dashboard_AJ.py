# === dashboard_AJ.py ===
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
import streamlit as st
from dotenv import load_dotenv
import os
import re

# Add TextBlob and other ML imports
from textblob import TextBlob
from gensim import corpora
from gensim.models import LdaModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from nltk.tokenize import word_tokenize
import json

# Load environment variables
load_dotenv()

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

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
        self.openai_analyzer = OpenAIAnalyzer()

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
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Number of Incidents',
            hovermode='x unified',
            legend_title='Predicted Category'
        )
        
        return fig

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
            fig, ax = plt.subplots(figsize=(10, 5))
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
            return go.Figure()
        
    def priority_vs_state_sunburst_by_category(self, df: pd.DataFrame, selected_category: str, category_col: str = 'Predicted Category') -> go.Figure:
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
                barmode='group',
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

    def generate_boxplot_for_category(self, df: pd.DataFrame, selected_category: str) -> go.Figure:
        df['Created'] = pd.to_datetime(df['Created'], errors='coerce')
        df['Resolved'] = pd.to_datetime(df['Resolved'], errors='coerce')
        df['MTTR'] = (df['Resolved'] - df['Created']).dt.total_seconds() / 3600

        df_filtered = df[df['Predicted Category'] == selected_category]

        fig = px.box(df_filtered, x='Priority', y='MTTR', title=f'MTTR by Priority - {selected_category}')
        return fig

    def reopen_analysis(self, df: pd.DataFrame) -> Tuple[int, float]:
        if 'Last reopened at' not in df.columns or 'Resolved' not in df.columns:
            return 0, 0.0

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
            title='Category-wise Reopened Incidents with Reopen Rate',
            xaxis_title='Category',
            yaxis_title='Incident Count',
            barmode='group',
            xaxis_tickangle=45,
            height=500
        )

        return fig

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
            color_discrete_sequence=px.colors.qualitative.Set2
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

    def mask_sensitive_data(self, text: str) -> str:
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
- Description: {row.get('Short description', 'N/A')[:150]}...
- Created: {row.get('Created', 'N/A')}
- Channel: {row.get('Channel', 'N/A')}
- Configuration Item: {row.get('Configuration item', 'N/A')}
- Last Reopened: {row.get('Last reopened at', 'N/A')}
- Work Notes: {row.get('Work notes', 'N/A')[:100]}...
"""
        
        prompt += """

Please provide:
1. Common patterns in these reopened incidents
2. Likely root causes for reopening
3. Specific recommendations to reduce reopen rates
4. Process improvements that could be implemented
5. Any notable trends by channel or configuration item

Focus on actionable insights that can help prevent future reopenings."""
        
        return prompt
    
    def get_openai_reopen_insights(self, prompt_text: str) -> str:
        """Get insights on reopened incidents using OpenAI"""
        try:
            return self.openai_analyzer.get_reopen_insights(prompt_text)
        except Exception as e:
            print(f"Error getting OpenAI insights: {str(e)}")
            return "Unable to generate insights at this time. Please check OpenAI configuration."
    
    def openai_health_check(self) -> bool:
        """Check if OpenAI service is available"""
        try:
            return self.openai_analyzer.health_check()
        except Exception:
            return False
