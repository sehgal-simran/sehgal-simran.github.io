(function () {
  var THEMES = ['light', 'dark', 'warm', 'lavender', 'blueprint', 'starry'];

  function setTheme(name) {
    document.documentElement.setAttribute('data-theme', name);
    localStorage.setItem('theme', name);
    document.querySelectorAll('.theme-dot').forEach(function (dot) {
      dot.classList.toggle('active', dot.getAttribute('data-theme') === name);
    });
  }

  // Apply saved theme immediately (before DOM ready to prevent flash)
  var saved = localStorage.getItem('theme');
  if (saved && THEMES.indexOf(saved) !== -1) {
    document.documentElement.setAttribute('data-theme', saved);
  }

  document.addEventListener('DOMContentLoaded', function () {
    // Mark active dot
    var current = document.documentElement.getAttribute('data-theme') || 'blueprint';
    document.querySelectorAll('.theme-dot').forEach(function (dot) {
      dot.classList.toggle('active', dot.getAttribute('data-theme') === current);
      dot.addEventListener('click', function () {
        setTheme(dot.getAttribute('data-theme'));
      });
    });
  });
})();
