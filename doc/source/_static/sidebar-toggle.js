function connectSidebarToggle(visibleSelector, themeSelector) {
  const visibleButton = document.querySelector(visibleSelector);
  const themeButton = document.querySelector(themeSelector);

  if (!visibleButton || !themeButton) {
    console.error("Sidebar button not found:", {
      visibleSelector,
      themeSelector,
    });
    return;
  }

  visibleButton.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    themeButton.click();
  });
}

function connectSidebarToggles() {
  connectSidebarToggle(
    ".header-article-item .sidebar-toggle.primary-toggle",
    ".bd-header__inner .pst-navbar-icon.sidebar-toggle.primary-toggle"
  );

  connectSidebarToggle(
    ".article-header-buttons .sidebar-toggle.secondary-toggle",
    ".bd-header__inner .pst-navbar-icon.sidebar-toggle.secondary-toggle"
  );
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", connectSidebarToggles);
} else {
  connectSidebarToggles();
}