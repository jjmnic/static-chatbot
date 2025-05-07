

const TopHeader = ({ handleSkipMainContent, toggleTheme, istoggleTheme }) => {
  return (
    <section className="topHeader">
      <div className="left">
        <span>भारत सरकार</span>
        <span>|</span>
        <span>Government of India</span>
      </div>
      <div className="right">
        <span onClick={handleSkipMainContent} className="skiptomain">
          {/* <span>
            <img src="../assets/images/topArw.png" alt="skip to main" />
          </span> */}
          <span>Skip to main content</span>
        </span>
        {/* <span className="skiptomain">|</span> */}

        {/* <span>|</span>
        <span onClick={toggleTheme}>
          <span
            style={{
              color: istoggleTheme === "light-theme" ? "yellow" : "black",
              backgroundColor:
                istoggleTheme === "light-theme" ? "black" : "white",
              padding:
                istoggleTheme === "light-theme"
                  ? "2px 4px 2px 4px"
                  : "2px 4px 2px 4px",
            }}
          >
            A
          </span>
        </span> */}
      </div>
    </section>
  );
};

export default TopHeader;
