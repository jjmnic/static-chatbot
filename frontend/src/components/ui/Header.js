import React from "react";

function Header({ istoggleTheme }) {
  return (
    <>
      <header className="header">
        <div className="logo">
          <img
            className="img-fluid nationalEmblem"
            src={`./assets/images/${
              istoggleTheme === "light-theme"
                ? "national-emblem-india.png"
                : "national-emblem-india.png"
            }`}
            alt="national emblem"
          />
          <img
            className="img-fluid logoLine"
            src="./assets/images/logoline.png"
            alt="line"
          />
          <img
            className="img-fluid naljalSewa jjm-logo"
            src="./assets/images/jjm-logo.png"
            alt="Nal Jal Sewa"
          />
          <div className="newlogo">
            <strong>Water Scheme Reporting Tool</strong>
            <span> Department of Drinking </span>
            <span>Water & Sanitation </span>
            <span>Ministry of Jalshakti </span>
          </div>
        </div>
        <div className="jjmTxt">
          <h2>Water Scheme Reporting Tool</h2>
          <p>Operation &amp; Maintenance of rural water supply schemes</p>
        </div>
        <div className="swachhBharat">
          <img
            src={`./assets/images/swachh-bharat.png`}
            className="swachhBharatLogo"
            alt="swachh bharat"
          />
        </div>
      </header>
      <header className="headerMob">
        <div className="logo">
          {/* <img
            className="img-fluid nationalEmblem"
            src="./assets/images/national-emblem-india.png"
            alt="national emblem"
          /> */}
          {/* <img
            className="img-fluid naljalSewa"
            src="./assets/images/logo.png"
            alt="Nal Jal Sewa"
          /> */}
          <img
            className="img-fluid naljalSewa jjm-logo"
            src="./assets/images/jjm-logo.png"
            alt="Nal Jal Sewa"
          />
          <div className="newlogo">
            <strong>Water Scheme Query System </strong>
            <span> Department of Drinking </span>
            <span>Water & Sanitation </span>
            <span>Ministry of Jalshakti </span>
          </div>
          <img
            className="img-fluid swachhBharatLogo"
            src={`./assets/images/${
              istoggleTheme === "light-theme"
                ? "swachh-bharat.png"
                : "swachh-bharat.png"
            }`}
            alt="swachh bharat"
          />
        </div>
        <div className="jjmTxt">
          <h2>Water Scheme Query System</h2>
          {/* <p>Water Scheme Query (Select a state and ask your questions!)</p> */}
        </div>
      </header>
    </>
  );
}

export default Header;
