const tabs = document.querySelectorAll(".tab");
const tabContents = document.querySelectorAll(".tab-content");
const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const fileName = document.getElementById("fileName");
const preview = document.getElementById("preview");
const urlInput = document.getElementById("image_url");
const urlPreview = document.getElementById("urlPreview");
const result = document.getElementById("result");
const queryText = document.getElementById("query_text");
const searchBtn = document.getElementById("searchBtn");
const clearBtn = document.getElementById("clearBtn");

let selectedFile = null;
let selectedUrl = null;
let mode = "upload";
let previewDataUrl = null;

// Cold start detection
let isFirstSearch = true;
const COLD_START_THRESHOLD = 5000; // 5 seconds

/* --- Tabs switching --- */
tabs.forEach(tab => {
  tab.addEventListener("click", () => {
    tabs.forEach(t => t.classList.remove("active"));
    tab.classList.add("active");

    tabContents.forEach(c => c.classList.remove("active"));

    if (tab.dataset.tab === "upload") {
      document.getElementById("uploadTab").classList.add("active");
      mode = "upload";
      urlPreview.style.display = "none";
    } else {
      document.getElementById("urlTab").classList.add("active");
      mode = "url";
      preview.style.display = "none";
    }
  });
});

/* --- Upload handling --- */
uploadBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) handleFile(file);
});

function handleFile(file) {
  selectedFile = file;
  selectedUrl = null;
  fileName.textContent = `Selected file: ${file.name}`;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewDataUrl = e.target.result;
    preview.src = previewDataUrl;
    preview.style.display = "block";
  };
  reader.readAsDataURL(file);
}

document.addEventListener("visibilitychange", () => {
  if (previewDataUrl && document.visibilityState === "visible") {
    preview.src = previewDataUrl;
    preview.style.display = "block";
  }
});

/* --- URL preview --- */
urlInput.addEventListener("input", () => {
  const url = urlInput.value.trim();
  if (url) {
    selectedUrl = url;
    urlPreview.src = url;
    urlPreview.style.display = "block";
  } else {
    selectedUrl = null;
    urlPreview.style.display = "none";
  }
});

/* --- Search --- */
searchBtn.addEventListener("click", async () => {
  const loadingIndicator = document.getElementById("loadingIndicator");
  const coldStartNotice = document.getElementById("coldStartNotice");
  
  const fd = new FormData();

  if (mode === "upload" && selectedFile) {
    fd.append("file", selectedFile);
  } else if (mode === "url" && selectedUrl) {
    fd.append("image_url", selectedUrl);
  } else {
    result.innerHTML = "<p class='error'>❌ Please provide an image first.</p>";
    return;
  }

  if (queryText.value.trim()) {
    fd.append("query_text", queryText.value.trim());
  }

  // Show loading indicator
  loadingIndicator.style.display = "block";
  result.innerHTML = "<p>⏳ Searching...</p>";
  
  // Show cold start notice on first search
  if (isFirstSearch) {
    coldStartNotice.style.display = "block";
  }

  const startTime = Date.now();

  try {
    const response = await fetch("/search", { 
      method: "POST", 
      body: fd 
    });
    
    const elapsedTime = Date.now() - startTime;
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    // Hide loading indicator
    loadingIndicator.style.display = "none";
    
    // Handle cold start notice
    if (isFirstSearch && elapsedTime > COLD_START_THRESHOLD) {
      // Keep notice for 3 seconds if it was a cold start
      setTimeout(() => {
        coldStartNotice.style.display = "none";
      }, 3000);
    } else {
      coldStartNotice.style.display = "none";
    }
    
    isFirstSearch = false;
    
    // Show results
    showResult(data);
    
  } catch (err) {
    console.error(err);
    loadingIndicator.style.display = "none";
    coldStartNotice.style.display = "none";
    result.innerHTML = `<p class='error'>❌ Error searching: ${err.message}</p>`;
  }
});

/* --- Clear --- */
clearBtn.addEventListener("click", () => {
  selectedFile = null;
  selectedUrl = null;
  previewDataUrl = null;
  fileInput.value = "";
  urlInput.value = "";
  fileName.textContent = "No file selected";
  preview.style.display = "none";
  urlPreview.style.display = "none";
  queryText.value = "";
  result.innerHTML = "<p>Cleared.</p>";
  
  // Hide notices if showing
  const loadingIndicator = document.getElementById("loadingIndicator");
  const coldStartNotice = document.getElementById("coldStartNotice");
  if (loadingIndicator) loadingIndicator.style.display = "none";
  if (coldStartNotice) coldStartNotice.style.display = "none";
});

/* --- Show result --- */
function showResult(data) {
  if (data.error) {
    result.innerHTML = `<p class='error'>❌ Error: ${data.error}</p>`;
    return;
  }
  
  if (data.results && data.results.length) {
    const best = data.results[0];
    const similarity = (best.similarity * 100).toFixed(1);
    
    result.innerHTML = `
      <h3>✨ Best Match</h3>
      <div style="max-width: 300px; margin: 0 auto; text-align: left;">
        <img src="${best.image_url}" alt="best-match" style="width: 100%; height: auto; display: block;">
        <p style="margin-top: 1rem; line-height: 1.6;">
          <b style="font-size: 1.1em;">${best.name}</b><br>
          <span style="color: #aaa;">Category:</span> ${best.category}<br>
          <span style="color: #aaa;">Similarity:</span> <span style="color: #22c55e; font-weight: bold;">${similarity}%</span>
        </p>
      </div>
    `;
  } else {
    result.innerHTML = "<p>No result found.</p>";
  }
}