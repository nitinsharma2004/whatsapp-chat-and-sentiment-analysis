import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Set page configuration
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "A simple WhatsApp chat analyzer application using Streamlit."
    }
)

# Custom CSS for Dark Mode
st.markdown("""
    <style>
    .main {
        # background-color: #333333;
        padding: 20px;
        border-radius: 10px;
        # color: #ffffff;
    }
    .header {
        text-align: center;
        font-size: 2.5em;
        color: #4CAF50;
        margin-bottom: 30px;
    }
    .stat-box {
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        text-align: center;
        background-color: #444444;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        color: #ffffff;
    }
    .section-title {
        color: #4CAF50;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="header">WhatsApp Chat Analyzer</h1>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # Fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.selectbox("Show analysis with respect to", user_list)

    if st.button("Show Analysis"):
        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.markdown('<h2 class="section-title">Top Statistics</h2>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <h3>Total Messages</h3>
                <h2>{num_messages}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-box">
                <h3>Total Words</h3>
                <h2>{words}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stat-box">
                <h3>Media Shared</h3>
                <h2>{num_media_messages}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="stat-box">
                <h3>Links Shared</h3>
                <h2>{num_links}</h2>
            </div>
            """, unsafe_allow_html=True)

        # Monthly Timeline
        st.markdown('<h2 class="section-title">Monthly Timeline</h2>', unsafe_allow_html=True)
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        plt.title("Messages Over Time")
        plt.xlabel("Time")
        plt.ylabel("Number of Messages")

        st.pyplot(fig)

        # Daily Timeline
        st.markdown('<h2 class="section-title">Daily Timeline</h2>', unsafe_allow_html=True)
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        plt.title("Messages Over Time")
        plt.xlabel("Date")
        plt.ylabel("Number of Messages")
        st.pyplot(fig)

        # Activity Map
        st.markdown('<h2 class="section-title">Activity Map</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<h3>Most Busy Day</h3>', unsafe_allow_html=True)
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            plt.xlabel("Day")
            plt.ylabel("Number of Messages")
            st.pyplot(fig)

        with col2:
            st.markdown('<h3>Most Busy Month</h3>', unsafe_allow_html=True)
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            plt.xlabel("Month")
            plt.ylabel("Number of Messages")
            st.pyplot(fig)

        st.markdown('<h2 class="section-title">Weekly Activity Map</h2>', unsafe_allow_html=True)
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        sns.heatmap(user_heatmap, cmap="YlGnBu", ax=ax)
        plt.title("Heatmap of Weekly Activity")
        st.pyplot(fig)

        # Finding the busiest users in the group (Group level)
        if selected_user == 'Overall':
            st.markdown('<h2 class="section-title">Most Busy Users</h2>', unsafe_allow_html=True)
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                plt.title("Busiest Users")
                plt.xlabel("Users")
                plt.ylabel("Number of Messages")
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.markdown('<h2 class="section-title">Wordcloud</h2>', unsafe_allow_html=True)
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        # Most common words
        st.markdown('<h2 class="section-title">Most Common Words</h2>', unsafe_allow_html=True)
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1], color='skyblue')
        plt.xticks(rotation='vertical')
        plt.xlabel("Frequency")
        plt.ylabel("Words")
        plt.title("Most Common Words")
        st.pyplot(fig)

        # Sentiment analysis
        st.markdown('<h2 class="section-title">Sentiment Analysis</h2>', unsafe_allow_html=True)
        sentiments = SentimentIntensityAnalyzer()
        positive = [sentiments.polarity_scores(i)["pos"] for i in df['message']]
        negative = [sentiments.polarity_scores(i)["neg"] for i in df['message']]
        neutral = [sentiments.polarity_scores(i)["neu"] for i in df['message']]
        new_df = pd.DataFrame({'positive': positive, 'negative': negative, 'neutral': neutral})

        average_sentiments = new_df.mean()
        fig, ax = plt.subplots(figsize=(8, 6))
        average_sentiments.plot(kind='bar', color=['green', 'red', 'blue'], ax=ax)
        plt.xlabel('Sentiment')
        plt.ylabel('Average Value')
        plt.title('Average Sentiment Distribution')
        plt.xticks(rotation=0)
        st.pyplot(fig)

        # # Sentiment Analysis App
        # analyzer = SentimentIntensityAnalyzer()
        #
        # st.title('Sentiment Analysis App')
        #
        # # Text input for the message
        # ms = st.text_input("Enter a message")
        #
        # # Button to predict sentiment
        # if st.button("Predict"):
        #     if ms:
        #         # Perform sentiment analysis
        #         sentiment = analyzer.polarity_scores(ms)
        #         score = sentiment['compound']
        #
        #         # Determine sentiment
        #         if score >= 0.05:
        #             prediction = "Predicted sentiment: Positive"
        #         elif score <= -0.05:
        #             prediction = "Predicted sentiment: Negative"
        #         else:
        #             prediction = "Predicted sentiment: Neutral"
        #
        #         # Save the prediction in session state
        #         if 'predictions' not in st.session_state:
        #             st.session_state.predictions = []
        #
        #         st.session_state.predictions.append((ms, prediction))
        #     else:
        #         st.warning("Please enter a message.")
        #
        # # Display previous predictions
        # if 'predictions' in st.session_state:
        #     st.write("Previous predictions:")
        #     for message, prediction in st.session_state.predictions:
        #         st.write(f"Message: {message} - {prediction}")
# Main content container
st.markdown('<div class="main">', unsafe_allow_html=True)
# Add content and functionalities here...
st.markdown('</div>', unsafe_allow_html=True)
