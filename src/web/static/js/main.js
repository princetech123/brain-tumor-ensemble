// Client-side interactions for NeuroEnsemble Dashboard

document.addEventListener("DOMContentLoaded", () => {
    // --- 1. Drag & Drop Setup ---
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const browseBtn = document.getElementById("browse-btn");
    const previewContainer = document.getElementById("preview-container");
    const filePreview = document.getElementById("file-preview");
    const filenameDisplay = document.getElementById("filename-display");
    const dropZoneContent = document.querySelector(".drop-zone-content");
    const uploadForm = document.getElementById("upload-form");
    const analyzeBtn = document.getElementById("analyze-btn");

    if (dropZone && fileInput) {
        // Prevent default browser drag behaviors
        ["dragenter", "dragover", "dragleave", "drop"].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        // Highlight drop zone when item is dragged over it
        ["dragenter", "dragover"].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.add("dragover");
            }, false);
        });

        ["dragleave", "drop"].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.remove("dragover");
            }, false);
        });

        // Handle dropped files
        dropZone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length) {
                fileInput.files = files;
                handleFiles(files[0]);
            }
        });

        // Open file dialog when clicking browse button or zone
        browseBtn.addEventListener("click", (e) => {
            e.stopPropagation(); // Avoid triggering dropZone click if nested
            fileInput.click();
        });

        dropZone.addEventListener("click", () => {
            fileInput.click();
        });

        // Handle file inputs via dialog
        fileInput.addEventListener("change", (e) => {
            if (fileInput.files.length) {
                handleFiles(fileInput.files[0]);
            }
        });

        // Display MRI preview
        function handleFiles(file) {
            if (!file.type.startsWith("image/")) {
                alert("Please upload an image file (PNG/JPG/JPEG).");
                return;
            }

            filenameDisplay.textContent = file.name;
            
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onloadend = () => {
                filePreview.src = reader.result;
                dropZoneContent.style.display = "none";
                previewContainer.style.display = "flex";
            };
        }
        
        // Loader on submit
        if (uploadForm && analyzeBtn) {
            uploadForm.addEventListener("submit", () => {
                analyzeBtn.disabled = true;
                analyzeBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing MRI...';
            });
        }
    }
});

// --- 2. Tab Switcher Logic ---
function switchTab(event, tabId) {
    // Get all tab contents and hide them
    const tabContents = document.querySelectorAll(".tab-content");
    tabContents.forEach(content => {
        content.classList.remove("active");
    });

    // Get all tab buttons and deactivate them
    const tabButtons = document.querySelectorAll(".tab-btn");
    tabButtons.forEach(btn => {
        btn.classList.remove("active");
    });

    // Show current tab content and mark button active
    const activeContent = document.getElementById(tabId);
    if (activeContent) {
        activeContent.classList.add("active");
    }
    
    event.currentTarget.classList.add("active");
}
