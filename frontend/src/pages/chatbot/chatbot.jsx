import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./chatbot.css";
import Header from '../../components/Header/Header';

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  // Updated Example Marketing Questions
  const exampleQuestions = [
    "How do I increase website traffic?",
    "What are the best digital marketing strategies?",
    "How can I improve social media engagement?",
    "What are the top SEO techniques?"
  ];

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async (text) => {
    if (!text.trim()) return;

    const userMessage = { role: "user", content: text };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await axios.post("http://localhost:8000/chat/", {
        message: text, 
      });

      const botMessage = { role: "assistant", content: response.data.response };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) => [...prev, { role: "assistant", content: "Something went wrong! Try again." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
    <Header />
    <div className="chat-container">
     
      
      {!messages.length && (
        <div className="examples-container">
          <h2>Try a marketing question</h2>
          <div className="examples-grid">
            {exampleQuestions.map((question, index) => (
              <button key={index} className="example-btn" onClick={() => sendMessage(question)}>
                {question}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="chat-box">
        {messages.map((msg, index) => (
          <div key={index} className={`chat-message ${msg.role}`}>
            {msg.content}
          </div>
        ))}
        {loading && <div className="chat-message assistant">Typing...</div>}
        <div ref={chatEndRef}></div>
      </div>

      <div className="input-box">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          onKeyDown={(e) => e.key === "Enter" && sendMessage(input)}
        />
        <button onClick={() => sendMessage(input)} disabled={loading}>Send</button>
      </div>
    </div>
    </>
  );
};

export default Chatbot;
