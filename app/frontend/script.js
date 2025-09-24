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
const embedderSelect = document.getElementById("embedder");
const searchBtn = document.getElementById("searchBtn");
const clearBtn = document.getElementById("clearBtn");
const samples = document.querySelectorAll(".sample");

let selectedFile = null;
let selectedUrl = null;
let mode = "upload";

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
searchBtn.addEventListener("click", () => {
  const fd = new FormData();

  if (mode === "upload" && selectedFile) {
    fd.append("file", selectedFile);
  } else if (mode === "url" && selectedUrl) {
    fd.append("image_url", selectedUrl);
  } else {
    result.innerHTML = "<p>❌ Please provide an image first.</p>";
    return;
  }

  if (queryText.value.trim()) fd.append("query_text", queryText.value.trim());

  result.innerHTML = "<p>⏳ Searching...</p>";

  fetch("/search", { method: "POST", body: fd })
    .then(res => res.json())
    .then(data => showResult(data))
    .catch(err => {
      console.error(err);
      result.innerHTML = "<p>❌ Error searching.</p>";
    });
});

/* --- Clear --- */
clearBtn.addEventListener("click", () => {
  selectedFile = null;
  selectedUrl = null;
  fileInput.value = "";
  urlInput.value = "";
  fileName.textContent = "No file selected";
  preview.style.display = "none";
  urlPreview.style.display = "none";
  result.innerHTML = "<p>Cleared.</p>";
});

/* --- Show result --- */
function showResult(data) {
  if (data.results && data.results.length) {
    const best = data.results[0];
    result.innerHTML = `
      <h3>Best Match</h3>
      <img src="${best.image_url}" alt="best-match">
      <p><b>${best.name}</b><br>${best.category}<br>Price(€): ${best.price}<br>Distance: ${best.distance}</p>
    `;
  } else {
    result.innerHTML = "<p>No result found.</p>";
  }
}
