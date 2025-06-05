import React from "react";
import "./Header.css"; // Import the CSS file

function Header() {
    return (
        <>
            <header className="header">
                <div className="logo">
                <img src="src/assets/LogoOriginal.png" alt="Logo"/>
                
                </div>
                <nav className="tabs">
                <a href="#home">Home</a>
                <a href="#about">About</a>
                <a href="#contact">Contact</a>
                </nav>
                <div className="search-login">
                <div className="search-box">
                    <input type="text" placeholder="Search..."/>
                    <button>Search</button>
                </div>
                <button className="login-button">Login</button>
                </div>
            </header>
        </>
    );
}

export default Header;
