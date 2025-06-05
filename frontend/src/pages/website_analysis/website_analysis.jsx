import React, { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import axios from "axios";
import "./website_analysis.css";

const WebsiteAnalysis = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const analysis = location.state?.analysis;
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const fetchMentions = async () => {
        setLoading(true);
        setError(null);

        try {
            console.log(Object.values(analysis)[0])
            const response = await axios.post("http://127.0.0.1:8000/fetch_mentions/", {
                topics:["brand24"], 
                limit: 1,
            });

            navigate("/mentions", { state: { mentions: response.data.mentions } });
        } catch (err) {
            setError("Failed to fetch mentions. Try again later.");
        } finally {
            setLoading(false);
        }
    };

    if (!analysis) {
        return (
            <div className="wrapper">
                <div className="main-box">
                    <p className="error">No analysis data found. Please analyze a website first.</p>
                    <button className="submit-button" onClick={() => navigate("/")}>
                        Go Back
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="wrapper">
            <div className="main-box">
                <h1 className="main-heading">Website Data Analysis</h1>
                {Object.entries(analysis).map(([key, value], index) => (
                    <div className="section" key={index}>
                        <h2>{key}</h2>
                        <div className="content-box">{value}</div>
                    </div>
                ))}
                <button className="submit-button" onClick={fetchMentions} disabled={loading}>
                    {loading ? "Fetching Mentions..." : "Find Mentions"}
                </button>
                {error && <p className="error">{error}</p>}
                <button className="submit-button" onClick={() => navigate("/")}>
                    Analyze Another
                </button>
            </div>
        </div>
    );
};

export default WebsiteAnalysis;
