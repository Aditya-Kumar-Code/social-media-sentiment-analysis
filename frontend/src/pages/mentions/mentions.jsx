import React, { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import axios from "axios";
import "./mentions.css"; // Import the CSS file

const FetchMentions = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const mentions = location.state?.mentions || [];

  // States for comments (YouTube, Bluesky, Reddit)
  const [youtubeComment, setYoutubeComment] = useState(""); // YouTube
  const [selectedYoutubeId, setSelectedYoutubeId] = useState(null);

  const [blueskyComment, setBlueskyComment] = useState(""); // Bluesky
  const [selectedBlueskyId, setSelectedBlueskyId] = useState(null);

  const [redditComment, setRedditComment] = useState(""); // Reddit
  const [selectedRedditId, setSelectedRedditId] = useState(null);

  const [instagramComment, setInstagramComment] = useState(""); // Reddit
  const [selectedInstagramId, setSelectedInstagramId] = useState(null);
  // Common states
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  // Handle YouTube comments
  const handleYoutubePostComment = async (youtubeUrl) => {
    if (!youtubeComment.trim()) {
      setError("Comment cannot be empty.");
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await axios.post("http://127.0.0.1:8000/post_comment_youtube/", {
        youtube_video_url:youtubeUrl ,
        comment_text: youtubeComment,
      });

      if (response.status === 200) {
        setSuccess("Comment posted successfully!");
        setYoutubeComment(""); // Clear input after success
      } else {
        setError("Failed to post comment. Please try again.");
      }
    } catch (err) {
      setError("Error posting comment: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Handle Bluesky comments
  const handlePostBlueskyComment = async (blueskyUrl) => {
    if (!blueskyComment.trim()) {
      setError("Comment cannot be empty.");
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await axios.post("http://127.0.0.1:8000/post_comment_bluesky/", {
        bluesky_url: blueskyUrl,
        bluesky_comment: blueskyComment, // ✅ Pass comment as 'v'
      });

      if (response.status === 200) {
        setSuccess("Bluesky comment posted successfully!");
        setBlueskyComment(""); // Clear input after success
      } else {
        setError("Failed to post Bluesky comment. Please try again.");
      }
    } catch (err) {
      setError("Error posting Bluesky comment: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Handle Reddit comments
  const handlePostRedditComment = async (redditUrl) => {
    if (!redditComment.trim()) {
      setError("Comment cannot be empty.");
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await axios.post("http://127.0.0.1:8000/post_comment_reddit/", {
        reddit_url: redditUrl,
        reddit_comment: redditComment, // ✅ Pass comment as 'comment'
      });

      if (response.status === 200) {
        setSuccess("Reddit comment posted successfully!");
        setRedditComment(""); // Clear input after success
      } else {
        setError("Failed to post Reddit comment. Please try again.");
      }
    } catch (err) {
      setError("Error posting Reddit comment: " + err.message);
    } finally {
      setLoading(false);
    }
  };
   // Handle Reddit comments
   const handlePostInstagramComment = async (instagramUrl) => {
    if (!redditComment.trim()) {
      setError("Comment cannot be empty.");
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await axios.post("/post_comment_instagram/", {
        instagram_url: instagramUrl,
        instagram_comment: instagramComment, 
      });

      if (response.status === 200) {
        setSuccess("Reddit comment posted successfully!");
        setRedditComment(""); 
      } else {
        setError("Failed to post Reddit comment. Please try again.");
      }
    } catch (err) {
      setError("Error posting Reddit comment: " + err.message);
    } finally {
      setLoading(false);
    }
  };
  // Separate social media and non-social media mentions
  const socialMediaMentions = mentions.filter((mention) =>
    ["reddit.com", "youtube.com", "bluesky.com","twitter.com"].includes(mention.platform)
  );
  const nonSocialMediaMentions = mentions.filter(
    (mention) => !["reddit.com", "youtube.com", "bluesky.com","twitter.com"].includes(mention.platform)
  );
  

  return (
    <div className="mentions-wrapper">
      <h1 className="mentions-heading">Mentions</h1>
      <button className="back-button" onClick={() => navigate("/")}>
        Back to Home
      </button>

      {mentions.length === 0 ? (
        <p className="no-mentions">No mentions found.</p>
      ) : (
        <div>
          {/* Social Media Mentions */}
          <section className="mentions-section">
            <h2 className="section-heading">Social Media Mentions</h2>
            <div className="mentions-grid">
              {socialMediaMentions.map((mention, index) => (
                <div key={index} className="mention-card">
                  <h2 className="mention-title">{mention.title}</h2>
                  <p className="mention-content">{mention.content}</p>
                  <a
                    href={mention.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="mention-link"
                  >
                    Link: {mention.url}
                  </a>
                  <p className="mention-metadata">
                    Created: {mention.created_utc !== "N/A"
                      ? new Date(mention.created_utc).toLocaleString()
                      : "N/A"}
                  </p>
                  <p className="mention-metadata">Platform: {mention.platform}</p>

                  {/* YouTube Comment Section */}
                  {mention.platform === "youtube.com" && (
                    <div className="comment-section">
                      <textarea
                        className="comment-input"
                        placeholder="Write a comment..."
                        value={selectedYoutubeId === mention.url ? youtubeComment : ""}
                        onChange={(e) => {
                          setYoutubeComment(e.target.value);
                          setSelectedYoutubeId(mention.url);
                        }}
                      />
                      
                      <button
                        className="comment-button"
                        onClick={() => handleYoutubePostComment(mention.url)}
                        disabled={loading}
                      >
                        {loading ? "Posting..." : "Posting Youtube Comment"}
                      </button>
                    </div>
                  )}

                  {/* Bluesky Comment Section */}
                  {mention.platform === "bluesky.com" && (
                    <div className="comment-section">
                      <textarea
                        className="comment-input"
                        placeholder="Write a comment..."
                        value={selectedBlueskyId === mention.url ? blueskyComment : ""}
                        onChange={(e) => {
                          setBlueskyComment(e.target.value);
                          setSelectedBlueskyId(mention.url);
                        }}
                      />
                      <button
                        className="comment-button"
                        onClick={() => handlePostBlueskyComment(mention.url)}
                        disabled={loading}
                      >
                        {loading ? "Posting..." : "Post on Bluesky"}
                      </button>
                    </div>
                  )}

                  {/* Reddit Comment Section */}
                  {mention.platform === "reddit.com" && (
                    <div className="comment-section">
                      <textarea
                        className="comment-input"
                        placeholder="Write a comment..."
                        value={selectedRedditId === mention.url ? redditComment : ""}
                        onChange={(e) => {
                          setRedditComment(e.target.value);
                          setSelectedRedditId(mention.url);
                        }}
                      />
                      <button
                        className="comment-button"
                        onClick={() => handlePostRedditComment(mention.url)}
                        disabled={loading}
                      >
                        {loading ? "Posting..." : "Post on Reddit"}
                      </button>
                    </div>
                  )}
                  {mention.platform === "instagram.com" && (
                    <div className="comment-section">
                      <textarea
                        className="comment-input"
                        placeholder="Write a comment..."
                        value={selectedInstagramId === mention.url ? instagramComment : ""}
                        onChange={(e) => {
                          setInstagramComment(e.target.value);
                          setSelectedInstagramId(mention.url);
                        }}
                      />
                      <button
                        className="comment-button"
                        onClick={() => handlePostInstagramComment(mention.url)}
                        disabled={loading}
                      >
                        {loading ? "Posting..." : "Post on Instagram..."}
                      </button>
                    </div>
                  )}
                  
                </div>
              ))}
            </div>
          </section>

          {/* Non-Social Media Mentions */}
          <section className="mentions-section">
            <h2 className="section-heading">Non-Social Media Mentions</h2>
            <div className="mentions-grid">
              {nonSocialMediaMentions.map((mention, index) => (
                <div key={index} className="mention-card">
                  <h2 className="mention-title">{mention.title}</h2>
                  <p className="mention-content">{mention.content}</p>
                  
                  <p className="mention-metadata">Platform: {mention.platform}</p>
                </div>
              ))}
            </div>
          </section>
        </div>
      )}

      {error && <p className="error-message">{error}</p>}
      {success && <p className="success-message">{success}</p>}
      <button 
        className="navigate-button" 
        onClick={() => navigate("/pdf-analysis")} >
        Get Pdf Analysis</button>
    </div>
  );
};

export default FetchMentions;
