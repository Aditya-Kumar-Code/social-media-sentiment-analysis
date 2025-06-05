import React, { useState, useEffect } from "react";
import './website_fetch.css';
import Header from '../../components/Header/Header';
import { useNavigate } from "react-router-dom";
const WebsiteFetch = () => {
    const [url, setUrl] = useState("");
    const [error, setError] = useState(null);
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (url.trim() === "") {
            setError("Please enter a valid URL.");
            return;
        }

        try {
            const response = await fetch("http://localhost:8000/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            navigate("/website-analysis", { state: { analysis: result.analysis } });
        } catch (err) {
            setError(err.message);
        }
    };
    return (
        <>
        <Header />
        <div className="wrapper">
            <div className="content">
                <p className="info-text">Analyze your entire website by simply entering its URL below.</p>
                <form className="input-form" onSubmit={handleSubmit}>
                    <input
                        type="text"
                        placeholder="Enter website URL"
                        value={url}
                        onChange={(e) => setUrl(e.target.value)}
                        className="url-input"
                    />
                    <button type="submit" className="submit-button">
                        Analyze
                    </button>
                </form>
                {error && <p className="error">Error: {error}</p>}
            </div>
        </div>
        </>
    );
};
export default WebsiteFetch;
