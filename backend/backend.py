import os
import re
import json
import praw
import openai 
import uvicorn
import requests
from dotenv import load_dotenv
from pydantic import BaseModel 
from datetime import datetime
from bs4 import BeautifulSoup
from collections import Counter
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from googleapiclient.discovery import build
from fastapi.middleware.cors import CORSMiddleware
from google.oauth2.credentials import Credentials
from reportlab.lib.styles import getSampleStyleSheet
from google_auth_oauthlib.flow import InstalledAppFlow
from google_auth_oauthlib.flow import Flow
from typing import List, Dict
from collections import defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import json
import base64
import io
from wordcloud import WordCloud
import pandas as pd
from dateutil import parser


from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
load_dotenv(override=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

openai.api_key =os.getenv('openai_key')




class AnalyzeRequest(BaseModel):
    url: str

class CommentRequest(BaseModel):
    post_url: str
    comment: str

class TopicRequest(BaseModel):
    topics: list[str] 
    subreddit: str = "all"  
    limit: int      

class Mention(BaseModel):
    title: str
    content: str
    url: str
    created_utc: str
    platform: str

class RequestPdfData(BaseModel):
    title:str

class CompetitorRequest(BaseModel):
    domain: str


class CompetitionAnalysisRequest(BaseModel):
    brands: List[str]
platforms = ["Instagram", "Facebook", "TikTok", "YouTube", "Twitter"]


class CommentYoutubeRequest(BaseModel):
    youtube_video_url: str
    comment_text: str

class CommentBlueskyRequest(BaseModel):
    bluesky_url:str
    bluesky_comment:str

class CommentRedditRequest(BaseModel):
    reddit_url:str
    reddit_comment:str

class CommentInstagramRequest(BaseModel):
    instagram_url:str
    instagram_comment:str

class ChatRequest(BaseModel):
    message: str  


with open("mentions.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

analyzer = SentimentIntensityAnalyzer()
records = []
positive_words = []
negative_words = []

for entry in raw_data:
    text = f"{entry['title']} {entry['content']}"
    score = analyzer.polarity_scores(text)
    sentiment = 'Positive' if score['compound'] > 0.05 else 'Negative' if score['compound'] < -0.05 else 'Neutral'
    words = re.findall(r'\b\w+\b', text.lower())
    if sentiment == 'Positive':
        positive_words.extend(words)
    elif sentiment == 'Negative':
        negative_words.extend(words)

    records.append({
        "Title": entry['title'],
        "Sentiment": sentiment,
        "Platform": entry['platform'],
        "URL": entry['url']
    })

# Convert to DataFrame
df = pd.DataFrame(records)
sentiment_counts = df['Sentiment'].value_counts().to_dict()
sentiment_by_platform = df.groupby(['Platform', 'Sentiment']).size().unstack(fill_value=0).to_dict(orient='index')



def generate_wordcloud_base64(words, color):
    if not words:
        return ""
    wc = WordCloud(width=800, height=400, background_color='white', colormap=color).generate(' '.join(words))
    img = io.BytesIO()
    wc.to_image().save(img, format='PNG')
    return base64.b64encode(img.getvalue()).decode('utf-8')

@app.get("/sentiment_analysis")
def get_sentiment():
    return {
        "sentiment_counts": sentiment_counts,
        "sentiment_by_platform": sentiment_by_platform,
        "wordclouds": {
            "positive": generate_wordcloud_base64(positive_words, 'Greens'),
            "negative": generate_wordcloud_base64(negative_words, 'Reds')
        },
        "mentions": records
    }

@app.post("/chat/")
async def chat_with_ai(request: ChatRequest):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": request.message}
            ]
        )
        return {"response": response["choices"][0]["message"]["content"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"response": str(e)})

class CommentSuggestion(BaseModel):
    content: str

@app.post("/analyze_comment/")
async def analyze_mention(mention: CommentSuggestion):
    try:
        prompt = f"Suggest an appropriate and thoughtful response to this social media post:\n\n'{mention.content}'\n\nResponse:"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7,
        )
        return {"suggested_response": response.choices[0].message["content"].strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/competitor-list")
async def chat_with_ai(request: CompetitorRequest):
    try:
        prompt = (
            f"Give a Python list of the top direct competitors of {request.domain}. "
            f"Respond only with the list of competitors. Do not explain anything else."
        )
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a competitive intelligence assistant. "
                        "You always answer with a list of competitors in Python list format. "
                        "Do not say you can't provide real-time data. Make a best guess based on general knowledge."
                    )
                },
                {"role": "user", "content": prompt}
            ]
        )
        return {"response": response["choices"][0]["message"]["content"]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail={"response": str(e)})


@app.post("/competitor-analysis")
async def analyze_brands(data: CompetitionAnalysisRequest) -> Dict[str, Dict[str, str]]:
    prompt = f"""
    Provide a detailed social media performance analysis for the following brands: {', '.join(data.brands)}.
    For each brand, break the response by platform (Instagram, Facebook, TikTok, YouTube, Twitter).
    Format: Start brand sections with **BrandName** and begin each platform summary with 'On PLATFORM,' or 'PLATFORM is used...'
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a social media analyst. Return well-structured, natural language analysis per brand and platform."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1600
    )

    raw_output = response['choices'][0]['message']['content']

    result = {}
    current_brand = None

    for line in raw_output.splitlines():
        line = line.strip()

        if not line:
            continue

        # Brand name line e.g. **Booking.com**
        brand_match = re.match(r"\*\*(.*?)\*\*", line)
        if brand_match:
            current_brand = brand_match.group(1).lower()
            result[current_brand] = {}
            continue

        # If a brand is active and this line contains a platform
        if current_brand:
            for platform in platforms:
                if re.search(fr"\b{platform}\b", line, re.IGNORECASE):
                    if platform not in result[current_brand]:
                        result[current_brand][platform] = line
                    else:
                        result[current_brand][platform] += " " + line
                    break

    return result


class KeywordRequest(BaseModel):
    keywords: List[str]

# Load mentions.json once at startup
if not os.path.exists("mentions.json"):
    raise FileNotFoundError("mentions.json file not found.")

with open("mentions.json", "r") as f:
    try:
        mentions_data = json.load(f)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in mentions.json")

@app.post("/analyze_keywords")
def analyze_keywords(request: KeywordRequest) :
    with open('mentions.json', 'r') as f:
        data = json.load(f)

    normalized_keywords = [k.lower() for k in request.keywords]

    # Nested dictionary: platform -> keyword -> count
    platform_keyword_counts = defaultdict(lambda: defaultdict(int))

    # Iterate through data
    for item in data:
        platform = item.get("platform", "unknown").lower()
        text = (item.get("title", "") + " " + item.get("content", "")).lower()
        
        for i, keyword in enumerate(normalized_keywords):
            if keyword in text:
                original_keyword = request.keywords[i]
                platform_keyword_counts[platform][original_keyword] += 1

    # Convert to regular dict for saving
    output = {platform: dict(kw_counts) for platform, kw_counts in platform_keyword_counts.items()}

    # Print or save output
    return output


def get_authenticated_service():
    credentials_file = "./credentials.json"  
    
    token_file = "token.json"
    creds = None
    SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
    # Use saved credentials if they exist
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    else:
        flow = Flow.from_client_secrets_file(
            credentials_file,
            scopes=SCOPES,
            redirect_uri="urn:ietf:wg:oauth:2.0:oob"
        )

        auth_url, _ = flow.authorization_url(prompt='consent')
        print("Please go to this URL and authorize the app:\n", auth_url)

        code = input("Enter the authorization code: ")
        flow.fetch_token(code=code)
        creds = flow.credentials

        with open(token_file, "w") as token:
            token.write(creds.to_json())

    return build("youtube", "v3", credentials=creds)

@app.post("/post_comment_youtube/")
async def post_comment(request: CommentYoutubeRequest):
    try:
        
        youtube = get_authenticated_service()
        video_id=request.youtube_video_url.split("=")[-1]
        response = youtube.commentThreads().insert(
            part="snippet",
            body={
                "snippet": {
                    "videoId": video_id ,
                    "topLevelComment": {
                        "snippet": {
                            "textOriginal": request.comment_text
                        }
                    }
                }
            }
        ).execute()

        return {
            "message": "Comment posted successfully!",
            "details": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    
@app.post("/post_comment_bluesky/")
async def post_comment_bluesky(request:CommentBlueskyRequest):
    from atproto import Client
    import datetime
    USERNAME = os.getenv("BLUESKY_USERNAME")
    PASSWORD = os.getenv("BLUESKY_PASSWORD")
    post_url = request.bluesky_url
    client = Client()
    client.login(USERNAME, PASSWORD)
    def extract_post_details(post_url):
        parts = post_url.split("/")
        handle = parts[4]  
        post_id = parts[-1] 
        return handle, post_id

    handle, post_id = extract_post_details(post_url)
    profile_data = client.app.bsky.actor.get_profile({"actor": handle})
    did = profile_data.did 
    post_uri = f"at://{did}/app.bsky.feed.post/{post_id}"
    post_ref = client.app.bsky.feed.get_posts({"uris": [post_uri]})

    if not post_ref.posts:
        raise ValueError("Post not found or invalid URL.")

    post_cid = post_ref.posts[0].cid 
    comment_text = request.bluesky_comment
    current_time = datetime.datetime.utcnow().isoformat() + "Z"  
    response = client.com.atproto.repo.create_record(
        data={
            "repo": client.me.did,  # Use DID instead of username
            "collection": "app.bsky.feed.post",
            "record": {
                "text": comment_text,
                "createdAt": current_time,
                "reply": {
                    "root": {"uri": post_uri, "cid": post_cid},
                    "parent": {"uri": post_uri, "cid": post_cid}
                }
            }
        }
    )

    return response.uri
        
@app.post("/post_comment_reddit/")
async def post_comment_reddit(request:CommentRedditRequest):
    import praw
    reddit = praw.Reddit(
        client_id=os.getenv('reddit_client_id'),
        client_secret=os.getenv('reddit_client_secret'),
        user_agent=os.getenv('reddit_user_agent'),
        username=os.getenv('reddit_user_name'),
        password=os.getenv('reddit_password')
    )
    submission = reddit.submission(url=request.reddit_url)
    comment = submission.reply(request.reddit_comment)
    return comment.id

@app.post("/post_comment_instagram/")
async def post_comment_instagram(request:CommentInstagramRequest):
    from instagrapi import Client

    # Login Credentials
    USERNAME = os.getenv('INSTAGRAM_USERNAME')
    PASSWORD = os.getenv('INSTAGRAM_PASSWORD')
    cl = Client()
    cl.login(USERNAME, PASSWORD)
    media_url = request.instagram_url
    media_id = cl.media_id(cl.media_pk_from_url(media_url))
    comment_text = request.instagram_comment
    cl.media_comment(media_id, comment_text)

def generate_pie_chart(data, labels, output_filename):
    """
    Generates a pie chart and saves it as an image.
    """
    plt.figure(figsize=(4, 4))
    plt.pie(data, labels=labels, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff', '#99ff99'])
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.savefig(output_filename, format='png', bbox_inches='tight')
    plt.close()

def save_summary_as_pdf(summary_text,title, positive_mentions, negative_mentions, neutral_mentions, platform_counts, output_filename="mentions_summary.pdf"):
    """
    Saves the provided summary in a well-formatted PDF with logo, tables, and charts.
    """
    doc = SimpleDocTemplate(output_filename, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Add Logo
    # logo_path = "logo.png"  
    # try:
    #     logo = Image(logo_path, width=120, height=50)
    #     elements.append(logo)
    # except Exception:
    #     print("Logo not found, skipping logo section.")

    # Title
    title = Paragraph(f"<b><font size=16>{title} Report</font></b>", styles["Title"])
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Extract key statistics from summary
    total_mentions = positive_mentions + negative_mentions + neutral_mentions

    
    table_data = [
        ["Metric", "Value"],
        ["Total Mentions", f"{total_mentions:,}"],
        ["Positive Mentions", f"{positive_mentions:,} ({(positive_mentions / total_mentions * 100):.1f}%)"],
        ["Negative Mentions", f"{negative_mentions:,} ({(negative_mentions / total_mentions * 100):.1f}%)"],
        ["Neutral Mentions", f"{neutral_mentions:,} ({(neutral_mentions / total_mentions * 100):.1f}%)"]
    ]
    table = Table(table_data, colWidths=[200, 200])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.black),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Generate Pie Chart for Sentiment
    sentiment_counts = [positive_mentions, negative_mentions, neutral_mentions]
    sentiment_labels = ["Positive", "Negative", "Neutral"]
    generate_pie_chart(sentiment_counts, sentiment_labels, "sentiment_chart.png")

    # Add Sentiment Chart
    try:
        sentiment_img = Image("sentiment_chart.png", width=300, height=200)
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("<b>Sentiment Analysis</b>", styles["Heading2"]))
        elements.append(sentiment_img)
    except Exception:
        print("Sentiment chart not found, skipping chart section.")

    # Generate Pie Chart for Platform Distribution
    platform_labels = list(platform_counts.keys())
    platform_values = list(platform_counts.values())
    generate_pie_chart(platform_values, platform_labels, "platform_chart.png")

    # Add Platform Chart
    try:
        platform_img = Image("platform_chart.png", width=300, height=200)
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("<b>Platform Distribution</b>", styles["Heading2"]))
        elements.append(platform_img)
    except Exception:
        print("Platform chart not found, skipping chart section.")

    # Add Summary Section
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<b>Context of the Debate:</b>", styles["Heading2"]))
    elements.append(Spacer(1, 6))

    sections = summary_text.strip().split("\n\n")
    for sec in sections:
        lines = sec.split("\n")
        if lines:
            heading = Paragraph(f"<b>{lines[0]}</b>", styles["Heading3"])
            elements.append(heading)
            elements.append(Spacer(1, 6))
            for line in lines[1:]:
                elements.append(Paragraph(line, styles["BodyText"]))
            elements.append(Spacer(1, 12))

    doc.build(elements)
    return "mentions_summary.pdf"


def analyze_context(titles, contents):
    """
    Generates a context summary of the debate using OpenAI API.
    """
    combined_text = "\n".join(titles[:10] + contents[:10])  

    prompt = f"""
    Analyze the following discussion data and summarize the main themes, sentiments, and key points:

    {combined_text}

    Provide a concise summary of the debate in 3-5 sentences.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an expert in social media analysis."},
                      {"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error analyzing context: {str(e)}"

def classify_sentiment(title, content):
    """
    Uses OpenAI API to classify sentiment (Positive, Negative, or Neutral).
    """
    prompt = f"""
    Analyze the sentiment of the following post.

    Title: {title}
    Content: {content}

    Respond with "Positive", "Negative", or "Neutral".
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a sentiment analysis expert."},
                      {"role": "user", "content": prompt}],
            temperature=0
        )
        sentiment = response["choices"][0]["message"]["content"].strip()
        return sentiment
    except Exception as e:
        return "Error"

@app.post("/analyze")
async def analyze_website(data: AnalyzeRequest):
    url = data.url
    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch the website.")

        soup = BeautifulSoup(response.content, "html.parser")
        content = soup.get_text()

        
        # Step 2: Analyze content using OpenAI API
        prompt = f"""
        Analyze the following website content and extract the following details:
        - Product Name: Identify the primary product or service mentioned on the website.
        - Brief Description: Summarize the product or service in one or two sentences.
        - Problem Solved: What problem does this product or service address or solve?
        - Benefits: What are the key benefits or advantages of using this product or service?
        - Suggested Keywords: Provide a list of relevant and suggested keywords for this website.
        Content:
        {content[:2000]}  # Truncate to avoid exceeding token limits
        """

        openai_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert website analyzer."},
                {"role": "user", "content": prompt},
            ],
        )
        analysis = openai_response.choices[0].message["content"]

        # Parse the response to extract details
        json_response = {
            "Product Name": None,
            "Brief Description": None,
            "Problem Solved": None,
            "Benefits": None,
            "Suggested Keywords": None,
        }

        for line in analysis.split("\n"):
            
            line = line.strip()
            line=line[2:]
            
            if line.startswith("Product Name:"):
                json_response["Product Name"] = line.replace("Product Name:", "").strip()
            elif line.startswith("Brief Description:"):
                json_response["Brief Description"] = line.replace("Brief Description:", "").strip()
            elif line.startswith("Problem Solved:"):
                json_response["Problem Solved"] = line.replace("Problem Solved:", "").strip()
            elif line.startswith("Benefits:"):
                json_response["Benefits"] = line.replace("Benefits:", "").strip()
            elif line.startswith("Suggested Keywords:"):
                json_response["Suggested Keywords"] = line.replace("Suggested Keywords:", "").strip()

        return {"analysis": json_response}

    except openai.error.OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/fetch_mentions/")
async def fetch_mentions(request: TopicRequest):
    try:
        
        all_mentions=[]
        for topic in request.topics:
            reddit_scraping = []
            try:

                reddit = praw.Reddit(
                    client_id=os.getenv('reddit_client_id'),
                    client_secret=os.getenv('reddit_client_secret'),
                    user_agent=os.getenv('reddit_user_agent')
                    # ,
                    # username=os.getenv('reddit_user_name'),
                    # password=os.getenv('reddit_password')
                )
                subreddit = reddit.subreddit("all")
                
                

                # ⚡ Use 'search' with 'relevance' sort for better matching
                for submission in subreddit.search(topic, sort="new", limit=request.limit):
                    created_utc = submission.created_utc
                    created_datetime = datetime.utcfromtimestamp(created_utc).strftime('%Y-%m-%d')
                    
                    reddit_scraping.append({
                        "title": submission.title,
                        "content": submission.selftext[:200] if submission.selftext else "No content available.",
                        "url": submission.url,
                        "score": submission.score,
                        "created_utc": created_datetime,
                        "platform": "reddit.com"
                    })
            except Exception as e:
                pass
            
            all_mentions.extend(reddit_scraping)
            youtube_scraping=[]
            youtube_URL = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={topic}&type=video&maxResults={request.limit}&key={os.getenv('youtube_api_key')}"

            response = requests.get(youtube_URL)
            data = response.json()
            # datetime.utcfromtimestamp(item['snippet']['publishedAt']).strftime('%Y-%m-%d')
            for item in data.get('items', []):
                youtube_title = item['snippet']['title']
                youtube_description = item['snippet']['description']
                youtube_video_url = f"https://www.youtube.com/watch?v={item['id']['videoId']}"
                dt=datetime.strptime(item['snippet']['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
                youtube_scraping.append({
                    "title": youtube_title,
                    "content": youtube_description[:200] if youtube_description else "No content available.",
                    "url": youtube_video_url,
                    "score": "N/A",
                    "created_utc": dt.strftime("%Y-%m-%d"),
                    "platform":"youtube.com"
                })
            all_mentions.extend(youtube_scraping)
            news_scraping=[]
            news_url = f"https://newsapi.org/v2/everything?q={topic}&pageSize={request.limit}&apiKey={os.getenv('newsorg_api_key')}"

            news_response = requests.get(news_url)
            news_response.raise_for_status()
            news_articles = news_response.json()
            
            for i in range(len(news_articles['articles'])):

                published_at = news_articles['articles'][i].get('publishedAt', 'N/A')
                news_scraping.append({
                            "title": news_articles['articles'][i]['title'],
                            "content": news_articles['articles'][i]['description'][:200] if news_articles['articles'][i]['description'] else "No content available.",
                            "url": news_articles['articles'][i]['source']['name'],
                            "score": datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d') if published_at != "N/A" else "N/A",
                            "created_utc": "N/A",
                            "platform":"web"
                        })
            all_mentions.extend(news_scraping)
            # try:

            #     twitter_scraping=[]
            #     url = "https://api.twitter.com/2/tweets/search/recent"
            #     headers = {
            #         "Authorization": f"Bearer {os.getenv('twitter_bearer_token')}"
            #     }
            #     params = {
            #         "query": topic,
            #         "max_results": request.limit,  
            #         "tweet.fields": "author_id,text,created_at"
            #     }
            #     response = requests.get(url, headers=headers, params=params)
            #     if response.status_code == 200:
            #             tweets = response.json().get("data", [])
            
            #             for tweet in tweets:
            #                         twitter_scraping.append({
            #                             "title": tweet['text'][:50] + "..." if len(tweet['text']) > 50 else tweet['text'],  # Truncate to 50 chars
            #                             "content": tweet['text'][:200],  # Truncate to 200 chars
            #                             "url": f"https://twitter.com/user/status/{tweet['id']}",  # Construct tweet URL
            #                             "score": "N/A",  # Placeholder for score
            #                             "created_utc": tweet['created_at'],  # Creation time
            #                             "platform": "twitter.com"  # Platform name
            #                         }) 
            # except Exception as e:
            #     pass
            # all_mentions.extend(twitter_scraping)
            url = "https://bsky.social/xrpc/com.atproto.server.createSession"
            data = {
                "identifier":os.getenv("BLUESKY_USERNAME"),
                "password": os.getenv("BLUESKY_PASSWORD")
            }

            response = requests.post(url, json=data)
            BLUESKY_ACCESS_TOKEN=response.json()['accessJwt']
            BLUESKY_API_URL = "https://bsky.social/xrpc/app.bsky.feed.searchPosts"
            headers = {
                "Authorization": f"Bearer {BLUESKY_ACCESS_TOKEN}"
            }
            params = {
                "q": topic,  
                "limit": request.limit  
            }

            response = requests.get(BLUESKY_API_URL, headers=headers, params=params)
            
            data = response.json()
            
            results = []
            for post in data.get("posts", []):
                handle = post["author"]["handle"]
                post_id = post["uri"].split("/")[-1]  # Extract the post ID
                post_url = f"https://bsky.app/profile/{handle}/post/{post_id}"
                created_at_iso = post["record"].get("createdAt", None)
                
                results.append({
                    "title": handle,
                    "content": post["record"].get("text", "No text available"),
                    "url": post_url,
                    "score": "N/A",
                    "created_utc": parser.parse(created_at_iso).strftime('%Y-%m-%d') if created_at_iso else "N/A",
                    "platform":"bluesky.com"

                })
            all_mentions.extend(results)
            # GRAPH_API_URL = "https://graph.facebook.com/v19.0"

            # url = f"{GRAPH_API_URL}/{os.getenv('INSTAGRAM_USER_ID')}/top_media"
            # params = {
            #     "user_id": os.getenv("INSTAGRAM_USER_ID"),
            #     "fields": "id,caption,media_type,media_url,permalink",
            #     "access_token": ""
            # }
            # response = requests.get(url, params=params)
            # data = response.json()
            
            # instagram_mentions=[]
            # for item in data['data']:
            #     instagram_mentions.append({
            #         "title": item.get("caption", "No caption available").split("\n")[0], 
            #         "content": item.get("caption", "No content available")[:200],  
            #         "url": item.get("permalink", "No URL available"),
            #         "score": "N/A",
            #         "created_utc": "N/A",
            #         "platform": "instagram.com"
            #     })

            # all_mentions.extend(instagram_mentions)
            openai.api_key = os.getenv('openai_key')
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": f"tell me exactly {request.limit} questions about {topic} people are asking on chatgpt. Give me a single text joined by commas."}]
            )
            chatgpt_mentions=[]
            chatgpt_response=response["choices"][0]["message"]["content"].split(",")
            chatgpt_response = chatgpt_response[:request.limit]
            for chatgpt_mention in chatgpt_response:
                chatgpt_mentions.append({
                    "title": f"{topic} Queries",
                    "content": chatgpt_mention,
                    "url": "N/A",
                    "score": "N/A",
                    "created_utc": "N/A",
                    "platform": "chatgpt.com"
                })
            all_mentions.extend(chatgpt_mentions)


        with open('mentions.json', 'w') as file:
            json.dump(all_mentions, file, indent=4)
        return {"mentions": all_mentions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pdf_generation/")
async def pdf_generation(request: RequestPdfData):
    try:
        with open("mentions.json", 'r') as file:
                data = json.load(file)
            
        mentions = data #

        if isinstance(mentions, list) and all(isinstance(m, dict) for m in mentions):
                # Extracting Titles & Content
            titles = [m.get("title", "") for m in mentions if m.get("title")]
            contents = [m.get("content", "") for m in mentions if m.get("content")]

            # Generate context summary
            debate_summary = analyze_context(titles, contents)

            # Total Mentions
            total_mentions = len(mentions)

            # Count platform occurrences
            platform_counts = Counter(m.get("platform", "unknown") for m in mentions)

            # Define social platforms
            social_platforms = {
                "Twitter": "X",
                "reddit.com": "Reddit",
                "youtube.com": "YouTube",
                "facebook.com": "Facebook",
                "instagram.com": "Instagram",
                "bluesky.com": "Bluesky"
            }

            # Count mentions for each platform
            social_mentions = sum(platform_counts.get(platform, 0) for platform in social_platforms)
            mentions_twitter = platform_counts.get("twitter.com", 0)
            mentions_reddit = platform_counts.get("reddit.com", 0)
            mentions_youtube = platform_counts.get("youtube.com", 0)
            mentions_facebook = platform_counts.get("facebook.com", 0)
            mentions_instagram = platform_counts.get("instagram.com", 0)
            mentions_bluesky = platform_counts.get("bluesky.com", 0)

            # Total Interactions (if available in API response)
            total_interactions = sum(int(m.get("score", 0)) for m in mentions if m.get("score", "N/A") != "N/A")

            # Sentiment Analysis using OpenAI
            positive_mentions = 0
            negative_mentions = 0

            for mention in mentions:
                sentiment = classify_sentiment(mention.get("title", ""), mention.get("content", ""))
                if sentiment == "Positive":
                    positive_mentions += 1
                elif sentiment == "Negative":
                    negative_mentions += 1

            # Non-Social Mentions
            non_social_mentions = total_mentions - social_mentions

            # Generate Output
            summary = f"""
            Numerical Summary
            -----------------
            Total Mentions: {total_mentions:,}
            
            Positive Mentions: {positive_mentions:,} ({(positive_mentions / total_mentions * 100):.1f}%)
            Negative Mentions: {negative_mentions:,} ({(negative_mentions / total_mentions * 100):.1f}%)
            Mentions on Social Networks: {social_mentions:,}
            Total Interactions: {total_interactions:,}
            Non-Social Mentions: {non_social_mentions:,}
            
            Platform-Specific Mentions:
            - X (Twitter): {mentions_twitter:,}
            - Reddit: {mentions_reddit:,}
            - YouTube: {mentions_youtube:,}
            - Facebook: {mentions_facebook:,}
            - Instagram: {mentions_instagram:,}
            - Bluesky: {mentions_bluesky:,}

            Context of the Debate:
            ----------------------
            {debate_summary}
            """
            neutral_mentions = total_mentions - (positive_mentions + negative_mentions)

            # Store the extracted values
            summary_text = summary.strip()  #
            platform_counts = dict(platform_counts)  
            # Call the improved PDF function with structured data
            pdf_file = save_summary_as_pdf(summary_text,request.title, positive_mentions, negative_mentions, neutral_mentions, platform_counts)
            return FileResponse(pdf_file, media_type='application/pdf', filename=f'{request.title}.pdf')



    except requests.exceptions.RequestException as e:
        print("Request failed:", e)

if __name__ == "__main__":
    uvicorn.run(app)
