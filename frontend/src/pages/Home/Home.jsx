import './Home.css';
import Header from '../../components/Header/Header';
import { Link } from 'react-router-dom'; 
function App() {
  return (
    <>
      <Header />
      <div className="box">
        <div className="powered-heading">
          <div className="title-heading">
            <h2>Supercharge Your Online Presence: Track, Analyze & Engage with AI-Powered Insights! 🚀</h2>
          </div>
          <div className="mention-heading">
            <h2>One Feed, Four Platforms:<br /> Monitor YouTube, Reddit, ChatGPT, Instagram, X & Bluesky in Real-Time!</h2>
          </div>
          <Link to="/website-fetch">
            <button className="analyze-button">Analyze Your Website By Giving URL</button>
          </Link>
          <h3 className="powered-text">Powered By</h3>
          <div className="powered-row">
            <img src={"src/assets/poweredby/anthropic.png"} width="170" height="50" alt="Anthropic" />
            <img src={"src/assets/poweredby/meta.png"} width="170" height="50" alt="Meta" />
            <img src={"src/assets/poweredby/openai.png"} width="170" height="50" alt="OpenAI" />
          </div>
        </div>
      </div>
    </>
  );
}

export default App;
