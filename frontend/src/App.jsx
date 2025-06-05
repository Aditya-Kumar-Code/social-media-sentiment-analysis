import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from './pages/Home/Home';
import Chatbot from './pages/chatbot/chatbot'
import WebsiteFetch from './pages/website_fetch/website_fetch';
import WebsiteAnalysis from './pages/website_analysis/website_analysis';
import FetchMentions from './pages/mentions/mentions';
import PdfAnalysis from "./pages/pdf_analysis/pdf_analysis";

function App() {
  return (
    <Router>
      <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/chatbot" element={<Chatbot/>} />
          <Route path="/website-fetch" element={<WebsiteFetch/>}/>
          <Route path="/website-analysis" element={<WebsiteAnalysis/>}/>
          <Route path="/mentions" element={<FetchMentions/>} />
          <Route path="/pdf-analysis" element={<PdfAnalysis/>} />
      </Routes>
    </Router>
  );
}

export default App;